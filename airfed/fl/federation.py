"""Decentralised federated learning between UAVs (paper Section IV-C, Eqs. 76-89).

There is no server. Each UAV keeps a timer; when it fires, the UAV asks its
neighbours for their actor and critic weights, and averages what arrives,
weighting every model by its owner's reputation.
"""

import numpy as np
import torch

from airfed.fl.quantize import compress, full_precision
from airfed.fl.reputation import Reputation


class Federation:
    def __init__(self, n_uavs, fed_cfg, variant, seed=0):
        self.cfg, self.variant = fed_cfg, variant
        self.K = n_uavs
        self.rng = np.random.default_rng(seed + 20_000)
        self.reputation = Reputation(n_uavs, fed_cfg.alpha_succ, fed_cfg.alpha_stab, fed_cfg.rho)
        self.timer = np.zeros(n_uavs)
        self.sent_bits = np.zeros(n_uavs)
        self.rounds = 0

    def new_episode(self):
        self.sent_bits[:] = 0.0

    def _payload(self, agent):
        """What one UAV transmits: its shared weights, quantized unless the ablation says not to."""
        tensors, grads = [], []
        for module in agent.federated_modules().values():
            for p in module.parameters():
                tensors.append(p.detach())
                g = agent.last_grad.get(id(p)) if hasattr(agent, "last_grad") else None
                grads.append(g if g is not None else torch.zeros_like(p))
        if self.variant.use_quant:
            return compress(tensors, grads, self.cfg.b_min, self.cfg.b_max)
        return full_precision(tensors)

    def _drop_chance(self, dist, r_comm):
        return self.cfg.drop_base + self.cfg.drop_far * (dist / r_comm) ** 2

    def step(self, agents, env, elapsed):
        """Advance timers by `elapsed` seconds and run any aggregation that is due."""
        if not self.variant.use_fed:
            return
        self.timer += elapsed
        speed = np.linalg.norm(env.vel, axis=1)
        period = 1.0 / (self.cfg.f_base * (1.0 + self.cfg.alpha_mobility * speed))     # Eq. 85
        due = np.nonzero(self.timer >= period)[0]
        if len(due) == 0:
            return

        trust = self.reputation.update(env.exec_total, env.exec_met)
        own = [[p.detach().clone() for m in a.federated_modules().values() for p in m.parameters()] for a in agents]
        payloads = {}
        new_weights = {}

        for k in due:
            # partners: in range and loud enough (Eq. 76)
            partners = [j for j in range(self.K)
                        if j != k and env.d_uu[k, j] <= env.cfg.r_comm and env.rssi_uu[k, j] >= self.cfg.rssi_fl_dbm]
            received = []
            for j in partners:
                if j not in payloads:
                    payloads[j] = self._payload(agents[j])
                self.sent_bits[j] += payloads[j][1]
                ok = self.rng.random() > self._drop_chance(env.d_uu[k, j], env.cfg.r_comm)
                self.reputation.record_exchange(j, ok)
                env.link_reliability[k, j] = 0.9 * env.link_reliability[k, j] + 0.1 * ok
                if ok:
                    received.append(j)

            # best effort: average whatever arrived, including our own model (Eqs. 81, 86)
            members = [k] + received
            w = np.array([trust[m] for m in members]) if self.variant.use_reputation else np.ones(len(members))
            w = w / w.sum()
            models = [own[k]] + [payloads[j][0] for j in received]
            new_weights[k] = [sum(float(wi) * model[i] for wi, model in zip(w, models)) for i in range(len(own[k]))]
            self.timer[k] = 0.0

        # everyone swaps in their new weights at once (Eqs. 87-89)
        with torch.no_grad():
            for k, weights in new_weights.items():
                params = [p for m in agents[k].federated_modules().values() for p in m.parameters()]
                for p, v in zip(params, weights):
                    p.copy_(v)
        self.rounds += len(new_weights)
