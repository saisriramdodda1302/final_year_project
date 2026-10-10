"""Non-learning reference policies.

These are not the paper's baselines (IF-DDPG, MUTO, JTORA, BCD-SCA); see the README.
They give a floor (random) and a strong hand-tuned reference (static + balanced).
"""

import numpy as np


class RandomPolicy:
    """Random velocity inside the speed limit, random valid offloading target."""

    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        K, v_max = self.env.K, self.env.cfg.v_max
        angle = self.rng.uniform(0, 2 * np.pi, K)
        speed = v_max * np.sqrt(self.rng.uniform(0, 1, K))
        velocity = np.column_stack([np.cos(angle), np.sin(angle)]) * speed[:, None]
        probs = obs["slot_mask"].astype(float)
        return velocity, probs / probs.sum(1, keepdims=True)


class StaticLocal:
    """Hover at the k-means positions and run every task on the serving UAV."""

    def __init__(self, env):
        self.env = env

    def act(self, obs):
        probs = np.zeros_like(obs["slot_mask"], dtype=float)
        probs[:, 0] = 1.0
        return np.zeros((self.env.K, 2)), probs


class StaticBalanced:
    """Hover at the k-means positions and spread tasks by spare CPU capacity.

    Score per slot = CPU speed x (1 - load). Running locally gets a small bonus
    because it skips the forwarding hop.
    """

    def __init__(self, env, local_bonus=1.5):
        self.env = env
        self.local_bonus = local_bonus

    def act(self, obs):
        uav, slots = obs["uav"], obs["slot_feat"]
        own = self.local_bonus * uav[:, 7] * np.clip(1.0 - uav[:, 6], 0.05, None)
        others = slots[:, :, 1] * np.clip(1.0 - slots[:, :, 0], 0.05, None)
        score = np.concatenate([own[:, None], others], axis=1) * obs["slot_mask"]
        return np.zeros((self.env.K, 2)), score / score.sum(1, keepdims=True)


BASELINES = {
    "random": RandomPolicy,
    "static_local": StaticLocal,
    "static_balanced": StaticBalanced,
}
