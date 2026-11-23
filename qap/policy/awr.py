"""Goal-conditioned policy pi(a | s, g) extracted from V = -d_theta by advantage-weighted
regression with hindsight goal relabelling (proposal Sec. 4.4; Week 2).

    python -m qap.policy.awr --dist-ckpt qap_runs/runs/qrl_iqe_k5/ckpt.pt --run-name awr_qrl

Advantage of the logged action a_t at s_t for goal g, in d_theta units (1 unit = k control
steps, k = frame skip of the distance model):
    A(s_t, a_t, g) = d(s_t, g) - d(s_{t+k}, g) - 1
    weight = min(exp(beta * A), w_max)
Goals: future states of the same episode (geometric, gamma) with prob 1 - p_random,
random dataset states otherwise. Actions are pd_joint_pos targets; the network predicts
joint deltas (target - current qpos) and the gripper command, per arm.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qap.collect.heldout import heldout_tasks, keep_episode
from qap.common.paths import episodes_dir, runs_dir
from qap.quasimetric.data import EpisodeStore, StoreConfig
from qap.quasimetric.encoder import EncoderConfig, ObjectTokenEncoder
from qap.quasimetric.model import load_checkpoint
from qap.state.categories import CategoryVocab

ACT_DIM = 16          # 2 x (7 joints + gripper)


def action_to_delta(action: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
    a = action.clone()
    for arm in (0, 1):
        a[:, arm * 8: arm * 8 + 7] = action[:, arm * 8: arm * 8 + 7] - robot[:, arm, :7]
    return a


def delta_to_action(delta: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
    a = delta.clone()
    for arm in (0, 1):
        a[:, arm * 8: arm * 8 + 7] = delta[:, arm * 8: arm * 8 + 7] + robot[:, arm, :7]
    return a


@dataclass
class PolicyConfig:
    num_categories: int = 512
    d_model: int = 256
    n_layers: int = 4
    emb: int = 256
    delta_scale: float = 0.05     # normalise joint deltas (rad) by this


class GoalPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        ecfg = EncoderConfig(num_categories=cfg.num_categories, d_model=cfg.d_model, n_layers=cfg.n_layers,
                             out_dim=cfg.emb)
        self.enc = ObjectTokenEncoder(ecfg)            # shared for state and goal
        self.mlp = nn.Sequential(nn.Linear(3 * cfg.emb, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                 nn.Linear(512, ACT_DIM))

    def forward(self, s, g) -> torch.Tensor:
        zs, zg = self.enc(s), self.enc(g)
        out = self.mlp(torch.cat([zs, zg, zg - zs], -1))
        scale = torch.ones(ACT_DIM, device=out.device)
        for arm in (0, 1):
            scale[arm * 8: arm * 8 + 7] = self.cfg.delta_scale
        return out * scale                                   # delta joints (rad) + gripper

    @torch.no_grad()
    def act(self, s, g) -> torch.Tensor:
        d = self.forward(s, g)
        a = delta_to_action(d, s["robot"])
        for arm in (0, 1):
            a[:, arm * 8 + 7] = torch.where(a[:, arm * 8 + 7] >= 0, 1.0, -1.0)   # gripper open/close
        return a


def save_policy(path: Path, pol: GoalPolicy, extra=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cfg=asdict(pol.cfg), state_dict=pol.state_dict(), extra=extra or {}), path)


def load_policy(path: Path, device="cpu") -> GoalPolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = GoalPolicy(PolicyConfig(**ck["cfg"]))
    pol.load_state_dict(ck["state_dict"])
    pol.extra = ck.get("extra", {})
    return pol.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-ckpt", type=Path, required=True)
    ap.add_argument("--run-name", default="awr")
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--include-heldout", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--w-max", type=float, default=100.0)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--p-random-goal", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=10_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dist = load_checkpoint(args.dist_ckpt, device)
    k = int(getattr(dist, "extra", {}).get("frame_skip", 5))
    dirs = [Path(d) for d in args.data] if args.data else [episodes_dir("mp"), episodes_dir("play")]
    held = set() if args.include_heldout else heldout_tasks()
    store = EpisodeStore(dirs, StoreConfig(frame_skip=k, max_episodes=args.max_episodes),
                         keep=(lambda m: keep_episode(m, held)) if held else None)

    vocab = CategoryVocab()
    pol = GoalPolicy(PolicyConfig(num_categories=max(64, len(vocab) + 8))).to(device)
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr)
    out = runs_dir(args.run_name)
    json.dump(dict(args={k_: str(v) for k_, v in vars(args).items()}, frame_skip=k), open(out / "config.json", "w"), indent=2)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = store.hindsight(args.batch, args.gamma, args.p_random_goal, device)
        with torch.no_grad():
            d_now = dist.distances(b["s"], b["g"]).mean(0)
            d_next = dist.distances(b["s_next"], b["g"]).mean(0)
            adv = d_now - d_next - 1.0
            w = torch.exp(args.beta * adv).clamp(max=args.w_max)
        target = action_to_delta(b["action"], b["s"]["robot"])
        pred = pol(b["s"], b["g"])
        scale = torch.ones(ACT_DIM, device=device)
        for arm in (0, 1):
            scale[arm * 8: arm * 8 + 7] = pol.cfg.delta_scale
        err = ((pred - target) / scale).pow(2).sum(-1)
        loss = (w * err).mean() / w.mean().clamp_min(1e-6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 10.0)
        opt.step()
        if step % args.log_every == 0:
            rec = dict(step=step, loss=float(loss), adv_mean=float(adv.mean()), adv_pos=float((adv > 0).float().mean()),
                       w_mean=float(w.mean()), w_max=float(w.max()), sec=round(time.time() - t0, 1))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_policy(out / "policy.pt", pol, extra=dict(dist_ckpt=str(args.dist_ckpt), frame_skip=k, step=step))
    print(f"done -> {out/'policy.pt'}")


if __name__ == "__main__":
    main()
