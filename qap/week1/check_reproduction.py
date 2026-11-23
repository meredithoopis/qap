"""Week-1 go/no-go: reproduced success within +-10 points of the reported numbers.

    python -m qap.week1.check_reproduction --model dp --results runs/dp_eval/<stamp>/dp_45 \
        --tasks StirSpoon PlaceMugRack
    python -m qap.week1.check_reproduction --model act --results <act eval dir> --tasks StirSpoon PlaceMugRack

Reads the JSON records written by the IG evaluators (fields env_id, success_once_mean or
success_at_end_mean) and compares each (task, level) to a reference. The default reference
is IG Table 11 (per-task, per-level success, DP/ACT); pass --reference ref.json
({"L0_TwoRobotStirSpoon-v1": 0.9, ...}, success in [0,1]) to compare against another table
(e.g. the Table-3 seen-task numbers of the corpus you trained on).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from qap.common.tasks import dataset_id, task_from_any
from qap.week1.ig_reference import TABLE11, level_success


def load_results(root: Path, metric: str) -> Dict[str, float]:
    best: Dict[str, Dict] = {}
    for f in root.rglob("*.json"):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for r in data if isinstance(data, list) else [data]:
            if not isinstance(r, dict) or r.get("status") != "success" or "env_id" not in r:
                continue
            if f"{metric}_mean" not in r:
                continue
            if r["env_id"] not in best or r.get("timestamp", "") >= best[r["env_id"]].get("timestamp", ""):
                best[r["env_id"]] = r
    return {k: float(v[f"{metric}_mean"]) for k, v in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--model", choices=["dp", "act"], required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    ap.add_argument("--metric", default="success_once")
    ap.add_argument("--tolerance", type=float, default=10.0, help="points")
    ap.add_argument("--reference", type=Path, default=None)
    args = ap.parse_args()

    got = load_results(args.results, args.metric)
    ref = json.load(open(args.reference)) if args.reference else None
    ok_all = True
    print(f"{'env':40s} {'ours':>6s} {'ref':>6s} {'diff':>6s}")
    for t in args.tasks:
        t = task_from_any(t)
        for l in args.levels:
            key = dataset_id(t, l)
            ours = got.get(key)
            if ref is not None:
                r = ref.get(key)
                r = None if r is None else 100 * float(r)
            else:
                r = level_success(t, l, args.model) if t in TABLE11 else None
            if ours is None or r is None:
                print(f"{key:40s} {'-' if ours is None else f'{100*ours:6.1f}':>6s} {'-' if r is None else f'{r:6.1f}':>6s}")
                ok_all = False
                continue
            diff = 100 * ours - r
            flag = "" if abs(diff) <= args.tolerance else "  <-- outside tolerance"
            ok_all &= abs(diff) <= args.tolerance
            print(f"{key:40s} {100*ours:6.1f} {r:6.1f} {diff:+6.1f}{flag}")
    print("GO" if ok_all else "NO-GO (or missing entries)")


if __name__ == "__main__":
    main()
