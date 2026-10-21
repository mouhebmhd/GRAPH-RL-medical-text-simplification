"""Component 3a: GAT encoder + actor-critic with a pointer head for MOVE actions."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import STATIC_DIM, EDGE_TYPES, N_MAX
from .env import GLOBAL_DIM, DYN_DIM, MOVE0, N_ACTIONS, INVOKE, STOP

NEG = -1e9


class DenseGATLayer(nn.Module):
    """Multi-head graph attention on padded dense graphs with a learned per-edge-type attention bias
    (so the policy can distinguish e.g. 'contrast' from 'elaboration' links)."""

    def __init__(self, d_in, d_out, heads=4, n_edge_types=len(EDGE_TYPES)):
        super().__init__()
        assert d_out % heads == 0
        self.h, self.dk = heads, d_out // heads
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.edge_bias = nn.Embedding(n_edge_types, heads)
        nn.init.zeros_(self.edge_bias.weight)
        self.res = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h, adj, nmask):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.h, self.dk)
        es = (Wh * self.a_src).sum(-1)                  # [B,N,H]
        ed = (Wh * self.a_dst).sum(-1)
        e = F.leaky_relu(es.unsqueeze(2) + ed.unsqueeze(1), 0.2)           # [B,N,N,H]
        e = e + self.edge_bias(adj.clamp(min=0))
        mask = (adj >= 0) & nmask.unsqueeze(1) & nmask.unsqueeze(2)
        e = e.masked_fill(~mask.unsqueeze(-1), NEG)
        alpha = torch.softmax(e, dim=2)
        alpha = alpha * mask.unsqueeze(-1)              # rows of padded nodes -> 0
        out = torch.einsum("bijh,bjhd->bihd", alpha, Wh).reshape(B, N, -1)
        return self.norm(F.elu(out) + self.res(h))


class SIPGPolicy(nn.Module):
    def __init__(self, hidden=64, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(STATIC_DIM + DYN_DIM, hidden), nn.ReLU())
        self.gat = nn.ModuleList([DenseGATLayer(hidden, hidden, heads) for _ in range(layers)])
        self.glob = nn.Sequential(nn.Linear(GLOBAL_DIM, 16), nn.ReLU())
        sd = hidden * 2 + 16
        self.state = nn.Sequential(nn.Linear(sd, hidden), nn.ReLU())
        self.q, self.k = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.head_inv = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.head_stop = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.hidden = hidden

    def forward(self, b):
        x, adj, nmask, glob, amask, cur = (b[k] for k in ("x", "adj", "nmask", "glob", "amask", "cur"))
        h = self.inp(x)
        for layer in self.gat:
            h = layer(h, adj, nmask)
        B = h.shape[0]
        h_cur = h[torch.arange(B), cur]
        pooled = (h * nmask.unsqueeze(-1)).sum(1) / nmask.sum(1, keepdim=True).clamp(min=1)
        s = self.state(torch.cat([h_cur, pooled, self.glob(glob)], -1))
        move = torch.einsum("bd,bnd->bn", self.q(s), self.k(h)) / math.sqrt(self.hidden)
        logits = torch.cat([self.head_inv(s), self.head_stop(s), move], -1)   # [B, 2+N_MAX]
        logits = logits.masked_fill(~amask, NEG)
        return logits, self.value(s).squeeze(-1)


def collate(obs_list, device="cpu"):
    out = {}
    for k in obs_list[0]:
        arr = np.stack([o[k] for o in obs_list])
        t = torch.as_tensor(arr, device=device)
        out[k] = t
    return out
