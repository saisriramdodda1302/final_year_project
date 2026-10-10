"""Dual-layer dynamic graph attention network (paper Section IV-A, Eqs. 46-56).

Each UAV builds two small graphs around itself:
  * cooperation layer: the UAV and the neighbours it can talk to (within R_comm)
  * service layer:     the IoT devices it currently covers
Attention runs inside each layer, and the service layer feeds the cooperation
layer through an urgency-weighted coverage matrix (Eq. 54).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    """Multi-head scaled dot-product attention restricted to graph edges (Eq. 55)."""

    def __init__(self, dim_in, dim_out, heads, edge_dim=0):
        super().__init__()
        assert dim_out % heads == 0
        self.heads, self.head_dim = heads, dim_out // heads
        self.query = nn.Linear(dim_in, dim_out, bias=False)
        self.key = nn.Linear(dim_in, dim_out, bias=False)
        self.value = nn.Linear(dim_in, dim_out, bias=False)        # W^(l) in Eqs. 52-53
        # edge features (distance, RSSI, ...) nudge the attention score of each link
        self.edge = nn.Linear(edge_dim, heads) if edge_dim else None

    def forward(self, x, mask, edge=None):
        n = x.shape[0]
        split = lambda t: t.view(n, self.heads, self.head_dim).transpose(0, 1)   # heads x n x d
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if self.edge is not None and edge is not None:
            score = score + self.edge(edge).permute(2, 0, 1)
        score = score.masked_fill(~mask, float("-inf"))           # only attend along edges
        weights = torch.softmax(score, dim=-1)
        return (weights @ v).transpose(0, 1).reshape(n, -1)


class DualLayerGAT(nn.Module):
    def __init__(self, uav_dim, dev_dim, coop_edge_dim, hidden=(128, 64), heads=4, gamma_urgency=0.5):
        super().__init__()
        self.gamma_urgency = gamma_urgency
        self.coop, self.serv, self.cross = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c_in, s_in = uav_dim, dev_dim
        for h in hidden:
            self.coop.append(GraphAttention(c_in, h, heads, coop_edge_dim))
            self.serv.append(GraphAttention(s_in, h, heads))
            self.cross.append(nn.Linear(s_in, h, bias=False))      # W_cross in Eq. 52
            c_in = s_in = h
        self.out_dim = 2 * hidden[-1]

    def forward(self, obs, k):
        """Embedding of UAV k built only from what UAV k can see."""
        K = obs["uav"].shape[0]
        eye_c = torch.eye(K, dtype=torch.bool)
        eye_s = torch.eye(obs["dev"].shape[0], dtype=torch.bool)

        # local graph of UAV k
        in_coop = obs["coop"][k]                                   # k and its neighbours
        in_serv = obs["cover"][k]                                  # devices k covers
        coop_mask = (in_coop[:, None] & in_coop[None, :] & obs["coop"]) | eye_c
        serv_mask = (in_serv[:, None] & in_serv[None, :]) | eye_s

        # device features + the link quality from UAV k to each device (Eq. 49)
        h_serv = torch.cat([obs["dev"], obs["serv_edge"][k]], dim=-1)
        h_coop = obs["uav"]

        # cross-layer matrix (Eq. 54): urgent devices weigh more
        weight = torch.exp(-self.gamma_urgency * obs["urgency"])
        visible = (obs["cover"] & in_serv[None, :] & in_coop[:, None]).float() * weight
        cross = visible / visible.sum(-1, keepdim=True).clamp_min(1e-6)

        for coop, serv, lin in zip(self.coop, self.serv, self.cross):
            new_serv = F.relu(serv(h_serv, serv_mask))                                         # Eq. 53
            h_coop = F.relu(coop(h_coop, coop_mask, obs["coop_edge"]) + lin(cross @ h_serv))  # Eq. 52
            h_serv = new_serv

        # fuse the two layers: UAV k's own node + its urgency-weighted service summary
        return torch.cat([h_coop[k], cross[k] @ h_serv], dim=-1)


class MLPEncoder(nn.Module):
    """Ablation 'w/o GATs': same inputs, but neighbours and devices are just averaged."""

    def __init__(self, uav_dim, dev_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * uav_dim + dev_dim + 2, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, obs, k):
        K, M = obs["uav"].shape[0], obs["dev"].shape[0]
        nbrs = obs["coop"][k].clone()
        nbrs[k] = False
        devs = obs["cover"][k]
        dev_in = torch.cat([obs["dev"], obs["serv_edge"][k]], dim=-1)
        mean = lambda x, m: (x * m[:, None]).sum(0) / m.sum().clamp_min(1)
        x = torch.cat([
            obs["uav"][k],
            mean(obs["uav"], nbrs.float()),
            mean(dev_in, devs.float()),
            torch.stack([nbrs.sum() / K, devs.sum() / M]).float(),
        ])
        return self.net(x)
