"""Training loop (paper Algorithm 1) and a runner for the baselines.

Usage:
    python -m airfed.train --variant airfed --episodes 100 --seed 0
    python -m airfed.train --baseline static_balanced --episodes 100
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch

from airfed.baselines import BASELINES
from airfed.config import VARIANTS, Config
from airfed.env.uav_env import UAVEdgeEnv
from airfed.fl.federation import Federation
from airfed.models.agent import Agent, offload_terms, sample_velocity

FIELDS = ["method", "uavs", "devices", "seed", "episode", "time", "energy_kj", "deadline_rate",
          "dropped", "coverage", "reward", "comm_mb", "tasks", "seconds"]


def to_torch(obs):
    return {k: torch.as_tensor(v) for k, v in obs.items()}


def make_optimizer(agent, tcfg):
    g = agent.groups()
    return torch.optim.Adam([
        {"params": g["encoder"], "lr": tcfg.lr_encoder},
        {"params": g["vel"], "lr": tcfg.lr_vel},
        {"params": g["off"], "lr": tcfg.lr_off},
        {"params": g["critic"], "lr": tcfg.lr_critic},
    ])


def update(agent, opt, window, bootstrap, tcfg):
    """One-step actor-critic update over a short window (Algorithm 1, lines 41-55).

    Advantage A(t) = r(t) + gamma * V(t+1) - V(t), with V(t+1) treated as a constant.
    """
    values = torch.stack([w["value"] for w in window])
    rewards = torch.tensor([w["reward"] for w in window], dtype=torch.float32)
    next_values = torch.cat([values[1:].detach(), bootstrap.view(1)])
    delta = rewards + tcfg.gamma * next_values - values.detach()        # one-step TD error

    # lambda = 0 keeps the paper's one-step advantage; lambda > 0 mixes in later TD errors
    advantage = torch.zeros_like(delta)
    running = 0.0
    for i in reversed(range(len(delta))):
        running = delta[i] + tcfg.gamma * tcfg.gae_lambda * running
        advantage[i] = running
    target = advantage + values.detach()

    vel_loss = -(torch.stack([w["vel_logp"] for w in window]) * advantage).mean()
    off_loss = -(torch.stack([w["off_logp"] for w in window]) * advantage).mean()
    critic_loss = (target - values).pow(2).mean()
    entropy = torch.stack([w["entropy"] for w in window]).mean()
    loss = vel_loss + off_loss + critic_loss - tcfg.entropy * entropy          # Eq. 74

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), tcfg.grad_clip)
    # keep |gradient| per shared parameter: the quantizer uses it as sensitivity
    agent.last_grad = {id(p): p.grad.detach().abs().clone()
                       for m in agent.federated_modules().values() for p in m.parameters() if p.grad is not None}
    opt.step()


def train(variant="airfed", episodes=100, seed=0, cfg=None, log=print):
    cfg = cfg or Config()
    var = VARIANTS[variant]
    torch.manual_seed(seed)
    env = UAVEdgeEnv(cfg.env, seed, cfg.model.graph_dt_base, cfg.model.alpha_speed)
    K = env.K
    # every UAV owns its own networks and optimizer (training is distributed)
    agents = [Agent(K, cfg.model, var.use_gat) for _ in range(K)]
    opts = [make_optimizer(a, cfg.train) for a in agents]
    fed = Federation(K, cfg.fed, var, seed)
    rows = []
    best_score, best_state = -np.inf, None

    for ep in range(1, episodes + 1):
        start = time.time()
        obs = env.reset()
        fed.new_episode()
        hidden = [torch.zeros(cfg.model.gru_hidden) for _ in range(K)]
        windows = [[] for _ in range(K)]
        total_reward, done, step = 0.0, False, 0

        while not done:
            # each UAV looks at its local graph and picks its actions (Algorithm 1, lines 8-27)
            o = to_torch(obs)
            outs = [agents[k](o, k, hidden[k]) for k in range(K)]

            velocity, vel_terms, probs = [], [], []
            for mean, log_std, logits, _, _ in outs:
                v, logp, ent = sample_velocity(mean, log_std, cfg.env.v_max)
                velocity.append(v.detach().numpy())
                vel_terms.append((logp, ent))
                probs.append(torch.softmax(logits, -1).detach().numpy())

            # all UAVs act together, the environment returns one reward per UAV (lines 30-37)
            obs, rewards, done, info = env.step(np.array(velocity), np.array(probs))
            total_reward += rewards.mean()
            step += 1

            # remember what each UAV did and got, for its next update
            for k in range(K):
                counts = torch.as_tensor(info["slot_counts"][k], dtype=torch.float32)
                off_logp, off_ent, _ = offload_terms(outs[k][2], counts)
                windows[k].append({
                    "vel_logp": vel_terms[k][0], "off_logp": off_logp,
                    "entropy": vel_terms[k][1] + off_ent,
                    "value": outs[k][3], "reward": float(rewards[k]),
                })
                hidden[k] = outs[k][4]

            if step % cfg.train.update_every == 0 or done:
                # value of the state we just reached, used as the bootstrap (no gradient)
                with torch.no_grad():
                    o = to_torch(obs)
                    boot = [agents[k](o, k, hidden[k])[3] for k in range(K)]
                # local learning on each UAV (lines 41-55)
                for k in range(K):
                    update(agents[k], opts[k], windows[k], boot[k], cfg.train)
                    windows[k] = []
                    hidden[k] = hidden[k].detach()          # cut the memory's gradient history here
                # then share weights with neighbours if a UAV's federated timer has fired
                fed.step(agents, env, cfg.train.update_every * cfg.env.dt)

        m = env.episode_metrics()
        row = {"method": variant, "seed": seed, "episode": ep, **m, "reward": total_reward,
               "comm_mb": fed.sent_bits.mean() / 8e6, "seconds": time.time() - start}
        rows.append(row)

        # on-policy actor-critic swings up and down, so remember the best stretch of training
        recent = np.mean([r["reward"] for r in rows[-5:]])
        if cfg.train.keep_best and ep >= 5 and recent > best_score:
            best_score, best_state = recent, [copy.deepcopy(a.state_dict()) for a in agents]

        if ep % 10 == 0 or ep == 1:
            log(f"[{variant} s{seed}] ep {ep:3d}  reward {total_reward:7.2f}  time {m['time']:.3f}s  "
                f"energy {m['energy_kj']:.1f}kJ  deadline {m['deadline_rate']:.3f}  "
                f"coverage {m['coverage']:.3f}  comm {row['comm_mb']:.2f}MB")

    if best_state is not None:
        for agent, state in zip(agents, best_state):
            agent.load_state_dict(state)
    return rows, agents


EVAL_TASKS = 50_000  # evaluation episodes draw tasks from a stream training never saw


def evaluate(agents, method, seed, cfg, episodes=5):
    """Run trained agents without learning: mean velocity, learned offloading probabilities."""
    env = UAVEdgeEnv(cfg.env, seed, cfg.model.graph_dt_base, cfg.model.alpha_speed, task_seed=seed + EVAL_TASKS)
    rows = []
    for ep in range(1, episodes + 1):
        obs, done, total_reward = env.reset(), False, 0.0
        hidden = [torch.zeros(cfg.model.gru_hidden) for _ in agents]
        while not done:
            o = to_torch(obs)
            velocity, probs = [], []
            with torch.no_grad():
                for k, agent in enumerate(agents):
                    mean, _, logits, _, hidden[k] = agent(o, k, hidden[k])
                    velocity.append((torch.tanh(mean) * cfg.env.v_max).numpy())
                    probs.append(torch.softmax(logits, -1).numpy())
            obs, rewards, done, _ = env.step(np.array(velocity), np.array(probs))
            total_reward += rewards.mean()
        rows.append({"method": method, "seed": seed, "episode": ep, **env.episode_metrics(),
                     "reward": total_reward, "comm_mb": 0.0, "seconds": 0.0})
    return rows


def run_baseline(name, episodes=100, seed=0, cfg=None, task_seed=None):
    cfg = cfg or Config()
    env = UAVEdgeEnv(cfg.env, seed, cfg.model.graph_dt_base, cfg.model.alpha_speed, task_seed=task_seed)
    policy = BASELINES[name](env, seed) if name == "random" else BASELINES[name](env)
    rows = []
    for ep in range(1, episodes + 1):
        start = time.time()
        obs, done, total_reward = env.reset(), False, 0.0
        while not done:
            velocity, probs = policy.act(obs)
            obs, rewards, done, _ = env.step(velocity, probs)
            total_reward += rewards.mean()
        rows.append({"method": name, "seed": seed, "episode": ep, **env.episode_metrics(),
                     "reward": total_reward, "comm_mb": 0.0, "seconds": time.time() - start})
    return rows


def save(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Train AirFed (or a variant) or run a baseline.")
    ap.add_argument("--variant", default="airfed", choices=list(VARIANTS))
    ap.add_argument("--baseline", choices=list(BASELINES))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uavs", type=int, default=6)
    ap.add_argument("--devices", type=int, default=40)
    ap.add_argument("--out", default="results/runs")
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = Config().with_env(n_uavs=args.uavs, n_devices=args.devices)
    name = args.baseline or args.variant
    if args.baseline:
        rows = run_baseline(args.baseline, args.episodes, args.seed, cfg)
        final = run_baseline(args.baseline, 5, args.seed, cfg, task_seed=args.seed + EVAL_TASKS)
    else:
        rows, agents = train(args.variant, args.episodes, args.seed, cfg)
        final = evaluate(agents, name, args.seed, cfg)
        model_path = Path(args.out) / f"{name}_k{args.uavs}_m{args.devices}_s{args.seed}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save([a.state_dict() for a in agents], model_path)
    save(rows, Path(args.out) / f"{name}_k{args.uavs}_m{args.devices}_s{args.seed}.csv")
    mean = lambda key: np.mean([r[key] for r in final])
    print(f"done: {name} (evaluation, 5 episodes) -> time {mean('time'):.3f}s, energy {mean('energy_kj'):.2f}kJ, "
          f"coverage {mean('coverage'):.3f}, deadline met {mean('deadline_rate'):.3f}")


if __name__ == "__main__":
    main()
