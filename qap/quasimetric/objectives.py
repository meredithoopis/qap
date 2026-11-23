"""Training objectives for d_theta and its selection-study variants (Week 2).

qrl      QRL (Wang et al. 2023), proposal eq. (3):
         max E_{s,g}[phi(d(s,g))]  s.t.  E_{(s,s')}[relu(d(s,s') - 1)^2] <= eps^2,
         phi(d) = -softplus(offset - d, beta), Lagrange multiplier by dual ascent.
tmd      TMD-style (Myers et al. 2025), state-only port of the loss *structure*:
         InfoNCE with logits -d(s_i, g_j) on future-state positives, plus a temporal
         backup that matches exp(-d(s,g)) to gamma * exp(-d(s',g)) under the Bregman
         divergence of exp (a LINEX-type loss). Verify constants against the official
         code (tmd-website.github.io) before reporting; all weights are flags.
hilp     HILP (Park et al. 2024): V(s,g) = -||phi(s) - phi(g)||, IQL expectile regression
         on r + gamma * V_target(s', g), EMA target network. Symmetric by construction.
distill  Fallback (execution plan): supervised regression of planner path lengths into
         the head (in-episode pairs of planner episodes, targets (u-t)/k in log space)
         + hinge pushing d(later, earlier) above D_far when an irreversible event happened
         in between + the QRL local constraint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qap.quasimetric.model import QuasimetricModel, ema_copy, ema_update
from qap.state.schema import EVENT_INDEX, LATCHED_EVENTS


@dataclass
class ObjConfig:
    # QRL
    eps: float = 0.25
    step_cost: float = 1.0
    softplus_offset: float = 200.0
    softplus_beta: float = 0.01
    lambda_init: float = 0.01
    # TMD
    gamma: float = 0.98
    tmd_zeta: float = 1.0
    nce_batch: int = 256
    # HILP
    expectile: float = 0.95
    target_tau: float = 0.005
    p_goal_is_state: float = 0.1
    # distill
    d_far: float = 300.0


def _softplus_inv(x: float) -> float:
    return math.log(math.expm1(x))


class Objective(nn.Module):
    needs = ("transitions",)

    def __init__(self, model: QuasimetricModel, cfg: ObjConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def extra_parameters(self):
        return []

    def after_step(self) -> None:
        pass


class QRL(Objective):
    needs = ("transitions", "random")

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.raw_lambda = nn.Parameter(torch.tensor(_softplus_inv(cfg.lambda_init)))

    def extra_parameters(self):
        return [self.raw_lambda]

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        c = self.cfg
        tr, goals = batch["transitions"], batch["random"]
        zs = self.model.embed(tr["s"])
        zs2 = self.model.embed(tr["s_next"])
        zg = self.model.embed(goals)
        d_local = self.model.dist_z(zs, zs2)
        violation = F.relu(d_local - c.step_cost).pow(2).mean()
        d_global = self.model.dist_z(zs, zg)
        phi = -F.softplus(c.softplus_offset - d_global, beta=c.softplus_beta).mean()
        lam = F.softplus(self.raw_lambda)
        loss = -phi + lam.detach() * violation - lam * (violation.detach() - c.eps ** 2)
        return dict(loss=loss, phi=phi.detach(), violation=violation.detach(), lam=lam.detach(),
                    d_local=d_local.mean().detach(), d_global=d_global.mean().detach())


class TMD(Objective):
    needs = ("future",)

    def forward(self, batch):
        c = self.cfg
        fb = batch["future"]
        n = min(c.nce_batch, fb["s"]["robot"].shape[0])
        cut = lambda st: {k: v[:n] for k, v in st.items()}
        zs = self.model.embed(cut(fb["s"]))
        zg = self.model.embed(cut(fb["g"]))
        D = self.model.dist_z(zs[:, None], zg[None])                      # [n, n]
        logits = -D
        labels = torch.arange(n, device=D.device)
        l_nce = F.cross_entropy(logits, labels)
        with torch.no_grad():
            zs2 = self.model.embed(cut(fb["s_next"]))
            y = math.log(c.gamma) - self.model.dist_z(zs2[:, None], zg[None])   # log(gamma * e^{-d(s',g)})
            # goal reached exactly: d(s_next, g) = 0 already handled by IQE (d(x,x)=0)
        x = logits
        diff = (x - y).clamp(max=20.0)
        l_t = (torch.exp(y) * (torch.exp(diff) - 1.0 - diff)).mean()     # Bregman divergence of exp
        loss = l_nce + c.tmd_zeta * l_t
        acc = (logits.argmax(1) == labels).float().mean()
        return dict(loss=loss, nce=l_nce.detach(), backup=l_t.detach(), nce_acc=acc.detach(),
                    d_pos=D.diag().mean().detach(), d_neg=D.mean().detach())


class HILP(Objective):
    needs = ("future",)

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.target = ema_copy(model)

    def after_step(self):
        ema_update(self.target, self.model, self.cfg.target_tau)

    def forward(self, batch):
        c = self.cfg
        fb = batch["future"]
        s, s2, g = fb["s"], fb["s_next"], fb["g"]
        B = s["robot"].shape[0]
        same = torch.rand(B, device=s["robot"].device) < c.p_goal_is_state
        g = {k: torch.where(same.view(-1, *([1] * (v.dim() - 1))), s[k], v) for k, v in g.items()}
        v = -self.model(s, g)
        with torch.no_grad():
            v_next = -self.target(s2, g)
            r = torch.where(same, torch.zeros_like(v), -torch.ones_like(v))
            target = r + c.gamma * (~same).float() * v_next
        adv = target - v
        w = torch.abs(c.expectile - (adv < 0).float())
        loss = (w * adv.pow(2)).mean()
        return dict(loss=loss, v=v.mean().detach(), target=target.mean().detach())


class Distill(Objective):
    needs = ("pairs", "transitions")

    def forward(self, batch):
        c = self.cfg
        pb, tr = batch["pairs"], batch["transitions"]
        zs, zg = self.model.embed(pb["s"]), self.model.embed(pb["g"])
        d_fwd = self.model.dist_z(zs, zg)
        l_fwd = F.smooth_l1_loss(torch.log1p(d_fwd), torch.log1p(pb["steps"]))
        latched = torch.tensor([EVENT_INDEX[e] for e in LATCHED_EVENTS], device=d_fwd.device)
        new_event = ((pb["ev_g"][:, latched] > 0.5) & (pb["ev_s"][:, latched] < 0.5)).any(1)
        d_bwd = self.model.dist_z(zg, zs)
        l_irr = (F.relu(c.d_far - d_bwd) * new_event.float()).sum() / new_event.float().sum().clamp_min(1.0) / c.d_far
        d_local = self.model(tr["s"], tr["s_next"])
        l_loc = F.relu(d_local - c.step_cost).pow(2).mean()
        loss = l_fwd + l_irr + l_loc
        return dict(loss=loss, fwd=l_fwd.detach(), irr=l_irr.detach(), local=l_loc.detach(),
                    frac_irr=new_event.float().mean().detach())


OBJECTIVES = {"qrl": QRL, "tmd": TMD, "hilp": HILP, "distill": Distill}


def make_objective(name: str, model: QuasimetricModel, cfg: Optional[ObjConfig] = None) -> Objective:
    return OBJECTIVES[name](model, cfg or ObjConfig())
