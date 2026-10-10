"""Multi-UAV mobile edge computing simulator (paper Section III).

One step lasts `dt` seconds. During a step:
  1. IoT devices generate tasks (Poisson arrivals).
  2. Each task is uploaded to the device's serving UAV (strongest signal).
  3. The serving UAV either runs it or forwards it to a neighbour (the offloading action).
  4. Each UAV runs its tasks first-in first-out on its CPU.
  5. Results travel back the same way and down to the device.
  6. UAVs fly with the chosen velocity (the trajectory action).
Every delay and energy term follows Eqs. 7-30.
"""

import numpy as np

from airfed.env import channel as ch

MB = 8e6  # bits in a megabyte (decimal, as networks count)


def kmeans(points, k, rng, iters=50):
    """Plain k-means with k-means++ seeding. Used for initial UAV positions (paper Sec. V)."""
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        label = np.argmin(((points[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        for c in range(k):
            if np.any(label == c):
                centers[c] = points[label == c].mean(0)
    return centers


class UAVEdgeEnv:
    def __init__(self, cfg, seed=0, graph_dt_base=20.0, alpha_speed=0.1, task_seed=None):
        self.cfg = cfg
        self.K, self.M = cfg.n_uavs, cfg.n_devices
        self.S = self.K - 1                    # offloading slots: local + up to K-1 neighbours
        self.graph_dt_base = graph_dt_base
        self.alpha_speed = alpha_speed

        # the scenario (devices, rates, UAV hardware) is fixed per seed
        rng = np.random.default_rng(seed)
        self.dev_pos = rng.uniform(0, cfg.area, size=(self.M, 2))
        self.dev_rate = rng.uniform(*cfg.rate_range, size=self.M)
        self.altitude = rng.uniform(*cfg.alt_range, size=self.K)
        self.cpu = rng.uniform(*cfg.cpu_range, size=self.K)
        self.home = kmeans(self.dev_pos, self.K, rng)

        # task randomness changes every episode; evaluation uses its own stream
        self.rng = np.random.default_rng(seed + 10_000 if task_seed is None else task_seed)

        # filled in by the federation: how reliable each UAV-UAV link has been
        self.link_reliability = np.ones((self.K, self.K))

    # ------------------------------------------------------------------ reset

    def reset(self):
        cfg = self.cfg
        self.t = 0
        self.pos = np.column_stack([self.home, self.altitude])  # x, y, fixed altitude
        self.vel = np.zeros((self.K, 2))
        self.energy = np.full(self.K, cfg.battery)
        self.cpu_free_at = np.zeros(self.K)                      # absolute time each CPU becomes idle

        self.last_tasks = np.zeros(self.M)                       # tasks per device in the last step
        self.last_deadline = np.full(self.M, np.mean(cfg.deadline_range))

        # cached graph views per UAV (Algorithm 1 refreshes them at a speed-dependent rate)
        self.graph_age = np.full(self.K, np.inf)
        self.view_coop = np.zeros((self.K, self.K), bool)
        self.view_cover = np.zeros((self.K, self.M), bool)

        # per-episode bookkeeping
        self.times, self.met, self.total_tasks, self.served = [], 0, 0, 0
        self.coverage_log = []
        self.used_energy = np.zeros(self.K)
        self.exec_total = np.zeros(self.K)
        self.exec_met = np.zeros(self.K)
        return self._observe()

    # ---------------------------------------------------------- observation

    def _links(self):
        """Distances, signals and associations at the current positions."""
        cfg = self.cfg
        self.d_ug = ch.ground_distance(self.pos, self.dev_pos)
        self.gain_ug = ch.ground_gain(cfg, self.d_ug)
        self.rssi_ug = ch.ground_rssi_dbm(cfg, self.d_ug)
        self.cover = self.rssi_ug >= cfg.rssi_min_dbm                  # Eq. 5
        best = np.argmax(self.rssi_ug, axis=0)                          # Eq. 6
        self.serving = np.where(self.cover.any(0), best, -1)

        self.d_uu = ch.uav_distance(self.pos)
        self.coop = self.d_uu <= cfg.r_comm                            # Eq. 43 (self included)
        self.rssi_uu = ch.air_rssi_dbm(cfg, self.d_uu)
        self.rate_uu = ch.air_rates(cfg, self.d_uu)

    def _refresh_views(self):
        """Rebuild each UAV's local graph when its refresh interval has passed."""
        speed = np.linalg.norm(self.vel, axis=1)
        interval = self.graph_dt_base / (1.0 + self.alpha_speed * speed)
        due = self.graph_age >= interval
        self.view_coop[due] = self.coop[due]
        self.view_cover[due] = self.cover[due]
        self.graph_age[due] = 0.0
        self.graph_age += self.cfg.dt

    def _observe(self):
        cfg, K, M, S = self.cfg, self.K, self.M, self.S
        self._links()
        self._refresh_views()

        now = self.t * cfg.dt
        backlog = np.maximum(self.cpu_free_at - now, 0.0)
        self.load = backlog / (cfg.load_max_steps * cfg.dt)            # 1.0 means "full"

        # UAV node features (Eq. 46): position, velocity, energy, load, CPU
        uav_feat = np.column_stack([
            self.pos[:, 0] / cfg.area, self.pos[:, 1] / cfg.area, self.pos[:, 2] / cfg.alt_range[1],
            self.vel / cfg.v_max,
            self.energy / cfg.battery,
            np.minimum(self.load, 2.0),
            self.cpu / cfg.cpu_range[1],
        ])

        # device node features (Eq. 48): location, queue, arrival rate, urgency
        urgency = self.last_deadline / cfg.deadline_range[1]
        dev_feat = np.column_stack([
            self.dev_pos / cfg.area,
            self.last_tasks / 10.0,
            self.dev_rate / cfg.rate_range[1],
            urgency,
        ])

        # edge features (Eqs. 47 and 49)
        coop_edge = np.stack([
            self.d_uu / cfg.r_comm,
            (self.rssi_uu + 90.0) / 60.0,
            np.full((K, K), cfg.bw_inter / 20e6),
            self.link_reliability,
        ], -1)
        clean_rate = cfg.bw * np.log2(1 + cfg.p_dev_tx * self.gain_ug / ch.dbm_to_watt(cfg.noise_dbm))
        serv_edge = np.stack([
            self.d_ug / 300.0,
            (self.rssi_ug + 90.0) / 60.0,
            clean_rate / 1e8,
        ], -1)

        # offloading slots: slot 0 = run locally, slot i = i-th nearest neighbour
        self.slots = -np.ones((K, S), int)
        slot_feat = np.zeros((K, S, 3))
        slot_mask = np.zeros((K, S + 1), bool)
        slot_mask[:, 0] = True
        for k in range(K):
            nbrs = [j for j in np.argsort(self.d_uu[k]) if j != k and self.view_coop[k, j]]
            for i, j in enumerate(nbrs[:S]):
                self.slots[k, i] = j
                slot_feat[k, i] = [min(self.load[j], 2.0), self.cpu[j] / cfg.cpu_range[1], self.rate_uu[k, j] / 5e8]
                slot_mask[k, i + 1] = self.load[j] < 1.0              # Eq. 42: don't push work onto a full UAV

        return {
            "uav": uav_feat.astype(np.float32),
            "dev": dev_feat.astype(np.float32),
            "coop": self.view_coop.copy(),
            "cover": self.view_cover.copy(),
            "coop_edge": coop_edge.astype(np.float32),
            "serv_edge": serv_edge.astype(np.float32),
            "urgency": urgency.astype(np.float32),
            "slot_feat": slot_feat.astype(np.float32),
            "slot_mask": slot_mask,
        }

    # ----------------------------------------------------------------- step

    def step(self, velocity, offload_probs):
        """velocity: K x 2 (m/s). offload_probs: K x (S+1) probabilities over slots."""
        cfg, K = self.cfg, self.K
        now = self.t * cfg.dt
        rng = self.rng

        # 1. new tasks this step (Poisson, Sec. III-A)
        counts = rng.poisson(self.dev_rate * cfg.dt)
        dev = np.repeat(np.arange(self.M), counts)
        n = len(dev)
        arrive = now + rng.uniform(0, cfg.dt, n)
        cycles = rng.uniform(*cfg.cycles_range, n)
        size_in = rng.uniform(*cfg.input_mb, n) * MB
        size_out = rng.uniform(*cfg.output_mb, n) * MB
        deadline = rng.uniform(*cfg.deadline_range, n)

        # 2. uplink / downlink rates. Interference is scaled by how long others transmit.
        serving = self.serving
        no_interference = ch.uplink_rates(cfg, self.gain_ug, serving, np.zeros(self.M))
        bits = np.bincount(dev, weights=size_in, minlength=self.M)
        airtime = np.where(no_interference > 0, np.minimum(bits / (no_interference * cfg.dt + 1e-9), 1.0), 0.0)
        up = ch.uplink_rates(cfg, self.gain_ug, serving, airtime)
        down = ch.downlink_rates(cfg, self.gain_ug, serving)

        # 3. offloading decision per task, sampled from the serving UAV's policy
        k1 = serving[dev]
        covered = k1 >= 0
        slot = np.zeros(n, int)
        slot_counts = np.zeros((K, self.S + 1))
        for k in range(K):
            mine = np.nonzero(k1 == k)[0]
            if len(mine):
                p = offload_probs[k] / offload_probs[k].sum()
                slot[mine] = rng.choice(self.S + 1, size=len(mine), p=p)
                slot_counts[k] = np.bincount(slot[mine], minlength=self.S + 1)

        target = k1.copy()
        fwd = covered & (slot > 0)
        target[fwd] = self.slots[k1[fwd], slot[fwd] - 1]
        # a stale view could point at a UAV that has drifted out of range: run locally instead
        bad = fwd & ((target < 0) | (self.d_uu[k1, np.maximum(target, 0)] > cfg.r_comm))
        target[bad] = k1[bad]
        hops = np.where(covered & (target != k1), 2, 1)

        energy = np.zeros(K)
        t_up = np.where(covered, size_in / np.maximum(up[np.maximum(dev, 0)], 1.0), 0.0)        # Eq. 8
        t_dec = hops * cfg.t_decision                                                           # Eq. 11
        rate_fwd = self.rate_uu[k1.clip(0), target.clip(0)]
        t_fwd = np.where(hops == 2, size_in / np.maximum(rate_fwd, 1.0), 0.0)                   # Eq. 12
        ready = arrive + t_up + t_dec + t_fwd

        # 4. run tasks first-in first-out on the executing UAV (Eq. 15)
        finish = np.zeros(n)
        queue = np.zeros(n)
        for j in range(K):
            idx = np.nonzero(covered & (target == j))[0]
            free = self.cpu_free_at[j]
            for i in idx[np.argsort(ready[idx])]:
                start = max(ready[i], free)
                queue[i] = start - ready[i]
                free = start + cycles[i] / self.cpu[j]
                finish[i] = free
            self.cpu_free_at[j] = max(free, now)

        # 5. results come back along the path and down to the device (Eqs. 16-17)
        t_ret = np.where(hops == 2, size_out / np.maximum(rate_fwd, 1.0), 0.0)
        t_down = np.where(covered, size_out / np.maximum(down[np.maximum(dev, 0)], 1.0), 0.0)
        total = finish - arrive + t_ret + t_down                                                # Eq. 7

        # task energy, charged to the UAV that spends it (Eqs. 21-30)
        comp_power = cfg.kappa * self.cpu**3                                                   # Eq. 27
        for i in np.nonzero(covered)[0]:
            a, b = k1[i], target[i]
            energy[a] += cfg.p_uav_rx * t_up[i] + cfg.p_uav_tx * t_down[i]                     # Eqs. 22, 30
            energy[a] += comp_power[a] * cfg.t_decision                                         # Eq. 24
            if hops[i] == 2:
                energy[b] += comp_power[b] * cfg.t_decision
                energy[a] += cfg.p_uav_tx * t_fwd[i] + cfg.p_uav_rx * t_ret[i]                 # Eqs. 25, 29
                energy[b] += cfg.p_uav_rx * t_fwd[i] + cfg.p_uav_tx * t_ret[i]
            energy[b] += cfg.p_idle * queue[i] + cfg.kappa * self.cpu[b] ** 2 * cycles[i]      # Eqs. 26, 28

        # 6. fly (Eqs. 36-37). Eq. 38 makes any velocity change reachable within one step.
        v = np.asarray(velocity, float)
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, cfg.v_max / np.maximum(speed, 1e-9))     # speed limit
        commanded = np.linalg.norm(v, axis=1)
        new_xy = self.pos[:, :2] + v * cfg.dt
        clipped = (new_xy < 0) | (new_xy > cfg.area)
        v[clipped] = 0.0                                                     # the geofence stops it at the border
        self.pos[:, :2] = np.clip(new_xy, 0, cfg.area)
        self.vel = v
        # energy follows the commanded speed: pushing against the border still burns power,
        # otherwise "fly into the wall" would cost the same as hovering and drones get stuck there
        flight = ch.flight_power(cfg, commanded) * cfg.dt                                    # Eqs. 19-20
        energy += flight
        self.energy -= energy
        self.used_energy += energy

        # rewards per UAV (Eqs. 67-70), credited to the serving UAV
        rewards = self._rewards(k1, covered, total, deadline, energy)

        # bookkeeping for metrics and for the reputation system
        met = covered & (total <= deadline)
        self.times.extend(total[covered].tolist())
        self.met += int(met.sum())
        self.total_tasks += n
        self.served += int(covered.sum())
        self.coverage_log.append(self.cover.any(0).mean())
        for j in range(K):
            on_j = covered & (target == j)
            self.exec_total[j] += on_j.sum()
            self.exec_met[j] += (met & on_j).sum()

        self.last_tasks = counts.astype(float)
        self.last_deadline = np.where(
            counts > 0,
            np.bincount(dev, weights=deadline, minlength=self.M) / np.maximum(counts, 1),
            np.mean(cfg.deadline_range),
        )

        self.t += 1
        done = self.t >= cfg.steps
        info = {"slot_counts": slot_counts, "step_energy": energy}
        return self._observe(), rewards, done, info

    def _rewards(self, k1, covered, total, deadline, energy):
        cfg = self.cfg
        e_low = cfg.p_hover * cfg.dt
        e_high = ch.flight_power(cfg, cfg.v_max) * cfg.dt * 1.1
        rewards = np.zeros(self.K)
        for k in range(self.K):
            mine = covered & (k1 == k)
            time_cost = min(total[mine].mean() / cfg.time_ref, 1.0) if mine.any() else 0.0
            energy_cost = np.clip((energy[k] - e_low) / (e_high - e_low), 0.0, 1.0)
            performance = -(cfg.alpha * time_cost + cfg.beta * energy_cost)            # Eq. 68
            late = np.maximum(total[mine] - deadline[mine], 0.0)
            penalty = -cfg.deadline_penalty * late.mean() if mine.any() else 0.0        # Eq. 69
            if cfg.coverage_credit == "covered":
                count = self.cover[k].sum()                  # every device this UAV covers (Eq. 70)
            else:
                count = (self.serving == k).sum()            # only devices it is the serving UAV for
            coverage = cfg.coverage_reward * count                                      # Eq. 70
            rewards[k] = performance + penalty + coverage
        return rewards

    # -------------------------------------------------------------- metrics

    def episode_metrics(self):
        return {
            "time": float(np.mean(self.times)) if self.times else 0.0,             # F_time, seconds
            "energy_kj": float(self.used_energy.mean() / 1e3),                      # F_energy per UAV
            "deadline_rate": self.met / max(self.served, 1),                       # among served tasks
            "dropped": 1.0 - self.served / max(self.total_tasks, 1),               # devices out of coverage
            "coverage": float(np.mean(self.coverage_log)),
            "tasks": self.total_tasks,
        }
