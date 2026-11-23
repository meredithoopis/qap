"""d_theta = head(f_theta(s), f_theta(g)), with a K-member ensemble and pessimistic pricing."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from qap.quasimetric.encoder import EncoderConfig, ObjectTokenEncoder
from qap.quasimetric.heads import make_head


@dataclass
class ModelConfig:
    head: str = "iqe"                 # iqe | mrn | l2
    dim_per_component: int = 32       # IQE: 64 components x 32
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


class QuasimetricModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ObjectTokenEncoder(cfg.encoder)
        self.head = make_head(cfg.head, cfg.encoder.out_dim, cfg.dim_per_component)

    def embed(self, s) -> torch.Tensor:
        return self.encoder(s)

    def dist_z(self, zs: torch.Tensor, zg: torch.Tensor) -> torch.Tensor:
        return self.head(zs, zg)

    def forward(self, s, g) -> torch.Tensor:
        return self.head(self.encoder(s), self.encoder(g))


class Ensemble(nn.Module):
    """K independently initialised members. Edge price = mean + kappa * std (paper: kappa = 1)."""

    def __init__(self, cfg: ModelConfig, k: int = 5):
        super().__init__()
        self.cfg = cfg
        self.members = nn.ModuleList([QuasimetricModel(cfg) for _ in range(k)])

    def __len__(self) -> int:
        return len(self.members)

    @torch.no_grad()
    def distances(self, s, g) -> torch.Tensor:
        return torch.stack([m(s, g) for m in self.members], 0)        # [K, B]

    @torch.no_grad()
    def pessimistic(self, s, g, kappa: float = 1.0) -> torch.Tensor:
        d = self.distances(s, g)
        if d.shape[0] == 1:
            return d[0]
        return d.mean(0) + kappa * d.std(0)

    @torch.no_grad()
    def embed(self, s) -> List[torch.Tensor]:
        return [m.embed(s) for m in self.members]

    @torch.no_grad()
    def pessimistic_z(self, zs: List[torch.Tensor], zg: List[torch.Tensor], kappa: float = 1.0) -> torch.Tensor:
        d = torch.stack([m.dist_z(a, b) for m, a, b in zip(self.members, zs, zg)], 0)
        return d[0] if d.shape[0] == 1 else d.mean(0) + kappa * d.std(0)


def ema_copy(model: nn.Module) -> nn.Module:
    t = copy.deepcopy(model)
    for p in t.parameters():
        p.requires_grad_(False)
    return t


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(1 - tau).add_(po.detach(), alpha=tau)


def save_checkpoint(path: Path, ens: Ensemble, extra: Optional[Dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(ens.cfg)
    torch.save(dict(cfg=cfg, k=len(ens), state_dict=ens.state_dict(), extra=extra or {}), path)


def load_checkpoint(path: Path, device="cpu") -> Ensemble:
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["cfg"]
    cfg = ModelConfig(head=c["head"], dim_per_component=c["dim_per_component"], encoder=EncoderConfig(**c["encoder"]))
    ens = Ensemble(cfg, ck["k"])
    ens.load_state_dict(ck["state_dict"])
    ens.extra = ck.get("extra", {})
    return ens.to(device).eval()
