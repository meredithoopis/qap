"""Screening rule + frozen task list + by-task splits (Week 1 deliverables).

    python -m qap.week1.screen_tasks            # writes qap/data/frozen_tasks.json, splits.json
    python -m qap.week1.screen_tasks --dry-run  # print only

Rule (execution plan, Week 1): retain an L3 task if the motion primitive changes and IG's
per-task L3 success is below L0. Made operational as:
  (R1) motion_changed in the curated inventory (grasp-only changes do not count unless
       --count-grasp-change);
  (R2) mean(DP, ACT) success at L0 minus at L3 >= --min-gap points (default 10: one
       episode out of ten per model, so '5 points' is within noise);
  (R3) the L3 variant keeps the demonstrated intent (drops PressStapler, PressJuicer);
  (R4) an L3 environment exists.
With the defaults this keeps 13 tasks (target 10-15). Every exclusion is logged with its
reason in frozen_tasks.json.

Splits are by task (no task shared between train / calibration / test). The default
assignment is written out explicitly below (PROPOSED_SPLIT) so that it is a reviewable
decision, stratified by reversibility (H3 needs irreversible steps in test) and by
substitution type. If the screened set differs from it (non-default flags), a
deterministic stratified split is generated instead.

Both files are hashed; the script refuses to overwrite frozen files unless --force.
The shortcut filter (DP trained on L0-L2 must succeed < 50% on L3) is applied later by
qap.week1.shortcut_filter, which can only *remove* tasks; record the outcome with
`--apply-shortcut-results` before any QAP evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from qap.common.io import load_json, save_json
from qap.common.paths import FROZEN_DIR, TASKS_DIR
from qap.week1.ig_reference import TABLE11, level_success
from qap.week1.inventory_spec import INVENTORY

PROPOSED_SPLIT: Dict[str, List[str]] = {
    # 5 tasks: two irreversible (H3), two with an added step (counterweight / unblocking), one tool->fixture
    "test": ["PourKetchupFries", "PlacePillBox", "ScanPillBottle", "PlaceFoodScale", "PlaceBookBookcase"],
    # 3 tasks: used only to calibrate tau (conformal) and to choose lambda
    "calibration": ["TransFood", "PutBox", "PlaceMagazineFolder"],
    # 5 tasks: goal head / lambda tuning / any IG-Sub-specific training
    "train": ["PlaceClothBasket", "PlaceCommodityRack", "PlacePlateRack", "PutCubeOnScale", "OpenLiquidCap"],
}


def _l3_env_exists(name: str) -> bool:
    import re

    for p in (TASKS_DIR / "dual_tasks_l3").glob("_0*.py"):
        if re.search(rf"TwoRobot{name}L3-v1", p.read_text(encoding="utf-8", errors="replace")):
            return True
    return False


def screen(min_gap: float, require_intent: bool, count_grasp: bool, model: str) -> Dict:
    kept, dropped = [], []
    for e in INVENTORY:
        reasons = []
        changed = e.motion_changed or (count_grasp and e.grasp_changed)
        if not changed:
            reasons.append("R1: primitive unchanged" + (" (grasp-only change)" if e.grasp_changed else ""))
        if e.name not in TABLE11:
            reasons.append("no IG Table-11 entry")
            gap = None
        else:
            gap = level_success(e.name, "L0", model) - level_success(e.name, "L3", model)
            if gap < min_gap:
                reasons.append(f"R2: L0-L3 gap {gap:.1f} < {min_gap}")
        if require_intent and not e.intent_preserved:
            reasons.append("R3: intent not preserved at L3")
        if not _l3_env_exists(e.name):
            reasons.append("R4: no L3 env")
        rec = dict(task=e.name, gap=gap, demo=e.demo, sub=e.sub, prim_l0=e.prim_l0, prim_l3=e.prim_l3,
                   reversibility=e.reversibility, role=e.role,
                   ig_L0=level_success(e.name, "L0", model) if e.name in TABLE11 else None,
                   ig_L3=level_success(e.name, "L3", model) if e.name in TABLE11 else None)
        if reasons:
            rec["excluded_because"] = reasons
            dropped.append(rec)
        else:
            kept.append(rec)
    return dict(retained=kept, excluded=dropped)


def stratified_split(tasks: List[Dict], seed: int = 0, frac=(0.4, 0.2, 0.4)) -> Dict[str, List[str]]:
    """Round-robin over reversibility strata (irreversible first) with per-split quotas."""
    rng = random.Random(seed)
    n = len(tasks)
    n_cal = max(1, round(frac[1] * n))
    n_test = round(frac[2] * n)
    quota = {"test": n_test, "calibration": n_cal, "train": n - n_cal - n_test}
    strata: Dict[str, List[str]] = {}
    for t in sorted(tasks, key=lambda r: r["task"]):
        strata.setdefault(t["reversibility"], []).append(t["task"])
    ordered: List[str] = []
    for key in sorted(strata, key=lambda k: (k != "irreversible", k)):
        names = strata[key]
        rng.shuffle(names)
        ordered += names
    split: Dict[str, List[str]] = {"train": [], "calibration": [], "test": []}
    cycle, i = ["test", "train", "calibration"], 0
    for name in ordered:
        for _ in range(3):
            k = cycle[i % 3]
            i += 1
            if len(split[k]) < quota[k]:
                split[k].append(name)
                break
    return {k: sorted(v) for k, v in split.items()}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gap", type=float, default=10.0)
    ap.add_argument("--no-intent-check", action="store_true")
    ap.add_argument("--count-grasp-change", action="store_true")
    ap.add_argument("--model", choices=["mean", "dp", "act"], default="mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=FROZEN_DIR)
    ap.add_argument("--force", action="store_true", help="overwrite existing frozen files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply-shortcut-results", type=Path, default=None,
                    help="shortcut_filter summary JSON; removes tasks that fail the filter (DP L3 success >= 50%%)")
    args = ap.parse_args()

    ft, sp = args.out / "frozen_tasks.json", args.out / "splits.json"

    if args.apply_shortcut_results:
        frozen, splits = load_json(ft), load_json(sp)
        res = load_json(args.apply_shortcut_results)
        missing = [t for t, r in res["tasks"].items() if not r.get("evaluated")]
        if missing:
            raise SystemExit(f"shortcut results missing for {missing}; finish the DP evaluations first")
        removed = {t for t, r in res["tasks"].items() if not r["passes"]}
        frozen["shortcut_filter"] = dict(source=str(args.apply_shortcut_results), removed=sorted(removed),
                                         results=res["tasks"], applied_utc=datetime.now(timezone.utc).isoformat())
        frozen["retained_after_shortcut"] = [t["task"] for t in frozen["retained"] if t["task"] not in removed]
        for k in ("train", "calibration", "test"):
            splits["splits"][k] = [t for t in splits["splits"][k] if t not in removed]
        splits["shortcut_removed"] = sorted(removed)
        frozen["sha256_16"] = _digest({k: v for k, v in frozen.items() if k != "sha256_16"})
        splits["sha256_16"] = _digest(splits["splits"])
        save_json(frozen, ft)
        save_json(splits, sp)
        print(f"shortcut filter removed {sorted(removed)}; retained {frozen['retained_after_shortcut']}")
        return

    res = screen(args.min_gap, not args.no_intent_check, args.count_grasp_change, args.model)
    names = sorted(r["task"] for r in res["retained"])
    proposed = sorted(sum(PROPOSED_SPLIT.values(), []))
    if names == proposed:
        split, how = {k: sorted(v) for k, v in PROPOSED_SPLIT.items()}, "PROPOSED_SPLIT (hand-stratified)"
    else:
        split, how = stratified_split(res["retained"], args.seed), f"stratified_split(seed={args.seed})"

    rule = dict(min_gap=args.min_gap, require_intent=not args.no_intent_check,
                count_grasp_change=args.count_grasp_change, model=args.model,
                source="IG Table 11 (DP/ACT, 10 episodes per cell)")
    frozen = dict(rule=rule, retained=res["retained"], excluded=res["excluded"],
                  n_retained=len(names), frozen_utc=datetime.now(timezone.utc).isoformat())
    frozen["sha256_16"] = _digest({k: v for k, v in frozen.items() if k != "frozen_utc"})
    splits = dict(splits=split, method=how, by="task", rule_sha=frozen["sha256_16"],
                  notes="no task shared between splits; calibration is used only for tau (and lambda)")
    splits["sha256_16"] = _digest(split)

    print(f"retained {len(names)} tasks: {names}")
    for r in res["excluded"]:
        if r["gap"] is not None and r["gap"] > 0 and len(r["excluded_because"]) == 1:
            print(f"  near miss: {r['task']}: {r['excluded_because'][0]}")
    print(f"split ({how}): " + json.dumps(split))
    if args.dry_run:
        return
    if (ft.exists() or sp.exists()) and not args.force:
        raise SystemExit(f"{ft} / {sp} exist (frozen). Use --force only before any evaluation has been looked at.")
    save_json(frozen, ft)
    save_json(splits, sp)
    print(f"wrote {ft} and {sp}")


if __name__ == "__main__":
    main()
