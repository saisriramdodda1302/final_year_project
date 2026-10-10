"""One UAV's brain: graph encoder -> GRU -> shared layer -> two actors and one critic.

Paper Section IV-B, Eqs. 56-66.
  * velocity actor   (continuous): where to fly, a Gaussian squashed by tanh
  * offloading actor (discrete):   run locally or hand the task to the i-th nearest neighbour
  * critic:                        how good the current situation is, V(s)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from airfed.models.gat import DualLayerGAT, MLPEncoder

UAV_DIM, DEV_DIM, COOP_EDGE_DIM, SERV_EDGE_DIM, SLOT_DIM = 8, 5, 4, 3, 3


def mlp(dim_in, hidden, dim_out):
    layers, d = [], dim_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, dim_out))
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, n_uavs, model_cfg, use_gat=True):
        super().__init__()
        self.slots = n_uavs - 1
        if use_gat:
            self.encoder = DualLayerGAT(UAV_DIM, DEV_DIM + SERV_EDGE_DIM, COOP_EDGE_DIM,
                                        model_cfg.gat_hidden, model_cfg.heads, model_cfg.gamma_urgency)
        else:
            self.encoder = MLPEncoder(UAV_DIM, DEV_DIM + SERV_EDGE_DIM)

        self.gru = nn.GRUCell(self.encoder.out_dim, model_cfg.gru_hidden)                 # Eq. 56
        self.shared = nn.Linear(model_cfg.gru_hidden, model_cfg.shared)                    # Eq. 61
        self.vel = mlp(model_cfg.shared, model_cfg.actor_hidden, 4)                         # mean + log std
        # the paper feeds only the shared features; we also show each slot's load,
        # CPU and link rate so the actor knows which neighbour a slot stands for
        self.off = mlp(model_cfg.shared + self.slots * SLOT_DIM, model_cfg.actor_hidden, self.slots + 1)
        self.critic = mlp(model_cfg.shared, model_cfg.critic_hidden, 1)                     # Eq. 66

        # start close to hovering: tiny mean, modest spread
        nn.init.uniform_(self.vel[-1].weight, -1e-3, 1e-3)
        nn.init.constant_(self.vel[-1].bias, 0.0)
        with torch.no_grad():
            self.vel[-1].bias[2:] = model_cfg.init_log_std

    def forward(self, obs, k, hidden):
        """Run one step for UAV k. `hidden` is its GRU memory from the previous step."""
        spatial = self.encoder(obs, k)
        hidden = self.gru(spatial[None], hidden[None])[0]
        feat = F.relu(self.shared(hidden))

        mean, log_std = self.vel(feat).chunk(2)
        log_std = log_std.clamp(-4.0, 0.5)

        logits = self.off(torch.cat([feat, obs["slot_feat"][k].flatten()]))
        logits = logits.masked_fill(~obs["slot_mask"][k], float("-inf"))                    # invalid targets

        value = self.critic(feat).squeeze(-1)
        return mean, log_std, logits, value, hidden

    # parameter groups, so each part gets the paper's learning rate
    def groups(self):
        encoder = list(self.encoder.parameters()) + list(self.gru.parameters()) + list(self.shared.parameters())
        return {"encoder": encoder, "vel": list(self.vel.parameters()),
                "off": list(self.off.parameters()), "critic": list(self.critic.parameters())}

    def federated_modules(self):
        """What gets shared with neighbours in federated learning (Eqs. 87-89)."""
        return {"vel": self.vel, "off": self.off, "critic": self.critic}


def sample_velocity(mean, log_std, v_max):
    """Tanh-squashed Gaussian. Returns velocity (m/s) and its log-probability."""
    std = log_std.exp()
    raw = mean + std * torch.randn_like(mean)
    squashed = torch.tanh(raw)
    log_prob = torch.distributions.Normal(mean, std).log_prob(raw).sum()
    log_prob = log_prob - torch.log(1 - squashed.pow(2) + 1e-6).sum()      # change-of-variables term
    entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum()
    return squashed * v_max, log_prob, entropy


def offload_terms(logits, counts):
    """Average log-probability of the slots actually chosen this step, plus entropy.

    The environment samples one slot per task; `counts` says how many tasks went to each slot.
    """
    log_p = torch.log_softmax(logits, dim=-1)
    probs = log_p.exp()
    safe = torch.where(torch.isfinite(log_p), log_p, torch.zeros_like(log_p))
    n = counts.sum()
    log_prob = (counts * safe).sum() / n if n > 0 else torch.zeros(())
    entropy = -(probs * safe).sum()
    return log_prob, entropy, probs
