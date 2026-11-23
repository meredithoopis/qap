"""Distance heads on top of embeddings z = f(s).

IQE   interval quasimetric embedding (Wang & Isola 2022a), maxmean reduction as in QRL.
      Each of K components holds D numbers; component distance = Lebesgue measure of
      the union of intervals [x_i, max(x_i, y_i)]. Asymmetric, triangle inequality for
      every parameter value, d(x, x) = 0. Only parameter: the max/mean mixing weight.
MRN   metric residual network (Liu et al. 2022): ||sym(x)-sym(y)|| + max_i relu(asym(x)-asym(y)).
L2    symmetric Hilbert distance ||x - y|| (HILP; Park et al. 2024).

Reference implementation: quasimetric-learning/torch-quasimetric (IQE/MRN). The code
here is self-contained so it runs inside the IG environment without extra installs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def iqe_component_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x, y: [..., K, D] -> [..., K] union length of intervals [x_i, max(x_i, y_i)]."""
    left = x
    right = torch.maximum(x, y)
    ls, order = torch.sort(left, dim=-1)
    rs = torch.gather(right, -1, order)
    run_max = torch.cummax(rs, dim=-1).values
    prev = torch.cat([ls[..., :1], run_max[..., :-1]], dim=-1)   # max right end of earlier intervals
    start = torch.maximum(ls, prev)
    return (rs - start).clamp_min(0).sum(-1)


class IQEHead(nn.Module):
    def __init__(self, in_dim: int = 2048, dim_per_component: int = 32):
        super().__init__()
        assert in_dim % dim_per_component == 0
        self.k = in_dim // dim_per_component
        self.dpc = dim_per_component
        self.raw_alpha = nn.Parameter(torch.zeros(()))        # sigmoid(0) = 0.5 max / 0.5 mean

    def forward(self, zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        zx, zy = torch.broadcast_tensors(zx, zy)
        shp = zx.shape[:-1] + (self.k, self.dpc)
        dk = iqe_component_distance(zx.reshape(shp), zy.reshape(shp))
        a = torch.sigmoid(self.raw_alpha)
        return a * dk.max(-1).values + (1 - a) * dk.mean(-1)


class MRNHead(nn.Module):
    def __init__(self, in_dim: int = 2048, sym_dim: int = 256, asym_dim: int = 256):
        super().__init__()
        self.sym = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, sym_dim))
        self.asym = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, asym_dim))

    def forward(self, zx, zy):
        zx, zy = torch.broadcast_tensors(zx, zy)
        ds = torch.sqrt(((self.sym(zx) - self.sym(zy)) ** 2).sum(-1) + 1e-8)
        da = F.relu(self.asym(zx) - self.asym(zy)).max(-1).values
        return ds + da


class L2Head(nn.Module):
    """Symmetric: d(x, y) = d(y, x). Optionally project first (HILP uses the raw phi)."""

    def __init__(self, in_dim: int = 2048, proj_dim: int = 0):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim) if proj_dim else nn.Identity()

    def forward(self, zx, zy):
        zx, zy = torch.broadcast_tensors(zx, zy)
        return torch.sqrt(((self.proj(zx) - self.proj(zy)) ** 2).sum(-1) + 1e-8)


def make_head(kind: str, in_dim: int, dim_per_component: int = 32) -> nn.Module:
    kind = kind.lower()
    if kind == "iqe":
        return IQEHead(in_dim, dim_per_component)
    if kind == "mrn":
        return MRNHead(in_dim)
    if kind in ("l2", "hilbert"):
        return L2Head(in_dim)
    raise ValueError(kind)
