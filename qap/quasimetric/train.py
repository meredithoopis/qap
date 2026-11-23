"""Train d_theta (and the selection-study variants) on robot episodes + play.

Main model (proposal Sec. 4.2 / Week 2):
    python -m qap.quasimetric.train --objective qrl --head iqe --ensemble 5 --run-name qrl_iqe_k5
Variants for the selection study (same encoder):
    --objective tmd  --head iqe   (TMD losses)
    --objective qrl  --head mrn   (MRN head)
    --objective hilp --head l2    (symmetric Hilbert, HILP objective)
    --objective distill --head iqe  (fallback: planner path-length distillation)
Visual cosine (DINOv2) needs no training: qap.quasimetric.visual_cosine.

Data: $QAP_DATA_ROOT/episodes/{mp,play} by default. Robot data only (never human video).
L3 / graded G2-G4 scenes of calibration+test tasks are excluded unless --include-heldout.
Output: $QAP_DATA_ROOT/runs/<run-name>/{ckpt.pt, log.jsonl, config.json}
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from qap.collect.heldout import heldout_tasks, keep_episode
from qap.common.paths import episodes_dir, runs_dir
from qap.quasimetric.data import EpisodeStore, StoreConfig
from qap.quasimetric.encoder import EncoderConfig
from qap.quasimetric.model import Ensemble, ModelConfig, save_checkpoint
from qap.quasimetric.objectives import ObjConfig, make_objective
from qap.state.categories import CategoryVocab


def build_store(args) -> EpisodeStore:
    dirs = [Path(d) for d in args.data] if args.data else [episodes_dir("mp"), episodes_dir("play")]
    held = set() if args.include_heldout else heldout_tasks()
    keep = (lambda m: keep_episode(m, held)) if held else None
    cfg = StoreConfig(max_objects=args.max_objects, frame_skip=args.frame_skip,
                      max_episodes=args.max_episodes, obj_dtype=args.obj_dtype)
    print(f"loading episodes from {dirs}; held-out tasks (L3/G2-G4 excluded): {sorted(held)}")
    return EpisodeStore(dirs, cfg, keep=keep)


def make_batch(store: EpisodeStore, needs, B: int, gamma: float, device):
    batch = {}
    if "transitions" in needs:
        batch["transitions"] = store.transitions(B, device)
    if "random" in needs:
        batch["random"] = store.random_states(B, device)
    if "future" in needs:
        batch["future"] = store.future(B, gamma, device)
    if "pairs" in needs:
        batch["pairs"] = store.same_episode_pairs(B, device)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["qrl", "tmd", "hilp", "distill"], default="qrl")
    ap.add_argument("--head", choices=["iqe", "mrn", "l2"], default="iqe")
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--include-heldout", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-objects", type=int, default=16)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--obj-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--components", type=int, default=64)
    ap.add_argument("--dim-per-component", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--softplus-offset", type=float, default=200.0)
    ap.add_argument("--softplus-beta", type=float, default=0.01)
    ap.add_argument("--tmd-zeta", type=float, default=1.0)
    ap.add_argument("--nce-batch", type=int, default=256)
    ap.add_argument("--expectile", type=float, default=0.95)
    ap.add_argument("--d-far", type=float, default=300.0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    run = args.run_name or f"{args.objective}_{args.head}_k{args.ensemble}"
    out = runs_dir(run)
    store = build_store(args)

    vocab = CategoryVocab()
    enc = EncoderConfig(num_categories=max(64, len(vocab) + 8), d_model=args.d_model, n_layers=args.layers,
                        n_heads=args.heads, out_dim=args.components * args.dim_per_component)
    mcfg = ModelConfig(head=args.head, dim_per_component=args.dim_per_component, encoder=enc)
    ocfg = ObjConfig(eps=args.eps, softplus_offset=args.softplus_offset, softplus_beta=args.softplus_beta,
                     gamma=args.gamma, tmd_zeta=args.tmd_zeta, nce_batch=args.nce_batch,
                     expectile=args.expectile, d_far=args.d_far)
    ens = Ensemble(mcfg, args.ensemble).to(device)
    objs = [make_objective(args.objective, m, ocfg).to(device) for m in ens.members]
    opts = []
    for m, o in zip(ens.members, objs):
        groups = [dict(params=m.parameters(), lr=args.lr, weight_decay=args.wd)]
        if o.extra_parameters():
            groups.append(dict(params=o.extra_parameters(), lr=args.lambda_lr, weight_decay=0.0))
        opts.append(torch.optim.AdamW(groups))

    cfg_dump = dict(args=vars(args), model=asdict(mcfg), objective=asdict(ocfg),
                    n_states=store.num_states, n_episodes=len(store.metas), vocab_size=len(vocab))
    json.dump(cfg_dump, open(out / "config.json", "w"), indent=2)
    log = open(out / "log.jsonl", "a")
    needs = objs[0].needs
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = make_batch(store, needs, args.batch, args.gamma, device)   # shared across members
        stats = []
        for o, opt in zip(objs, opts):
            o.train()
            res = o(batch)
            opt.zero_grad(set_to_none=True)
            res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(o.model.parameters(), 10.0)
            opt.step()
            o.after_step()
            stats.append({k: float(v) for k, v in res.items()})
        if step % args.log_every == 0:
            mean = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
            mean.update(step=step, sec=round(time.time() - t0, 1))
            log.write(json.dumps(mean) + "\n")
            log.flush()
            print(json.dumps(mean), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(out / "ckpt.pt", ens, extra=dict(objective=args.objective, frame_skip=args.frame_skip,
                                                             step=step, run=run))
    print(f"done -> {out/'ckpt.pt'}")


if __name__ == "__main__":
    main()
