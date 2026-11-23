"""Simulator-side inventory (run once on the server, CPU only, ~10 min).

    python -m qap.week1.runtime_inventory --levels L0 L1 L2 L3

For every task and level: instantiate the env, reset(seed), list the objects the state
extractor sees (names, normalised categories, kinds, sizes), the boolean predicates of
evaluate(), and whether an IG motion-planning solution is registered. Then:
  * qap_runs/inventory/runtime_inventory.json   (per task/level dump)
  * qap/data/category_vocab.json                (frozen int ids, built from all scenes)
  * qap/data/category_overrides.draft.json      (actor -> category guesses to review;
                                                 copy to category_overrides.json when done)
Run qap.week1.inventory --runtime ... afterwards to merge it into the inventory table.
Each (task, level) runs in a fresh subprocess because the level flags are globals.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
from typing import Dict, List

from qap.common.io import save_json
from qap.common.paths import FROZEN_DIR, inventory_dir
from qap.common.tasks import all_tasks


def _probe(task: str, level: str, seed: int) -> Dict:
    from qap.common.envs import has_mp_solution, make_env
    from qap.state.extractor import ObjectCentricExtractor

    out: Dict = dict(task=task, level=level, ok=False)
    try:
        env = make_env(task, level)
        env.reset(seed=seed)
        ex = ObjectCentricExtractor(env, task=task)
        ex.reset()
        s = ex.extract(None)
        d = ex.describe()
        out.update(d)
        out["obj_half_extents"] = [sl.half_extents.round(4).tolist() for sl in ex.slots]
        out["obj_static"] = [bool(sl.is_static) for sl in ex.slots]
        out["obj_rest_xyz"] = [s["obj"][i, :3].round(4).tolist() for i in range(len(ex.slots))]
        out["all_actor_names"] = sorted(env.unwrapped.scene.actors.keys())
        out["all_articulation_names"] = sorted(env.unwrapped.scene.articulations.keys())
        out["robot_base_xyz"] = [a.robot.pose.p.cpu().numpy().reshape(-1)[:3].round(3).tolist()
                                 for a in env.unwrapped.agent.agents]
        out["has_mp_solution"] = has_mp_solution(task, level)
        out["ok"] = True
        env.close()
    except Exception as e:  # keep going; failures are part of the inventory
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc(limit=5)
    return out


def _worker(args):
    return _probe(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--write-frozen", action="store_true",
                    help="write qap/data/category_vocab.json even when --tasks is a subset")
    args = ap.parse_args()

    tasks = args.tasks or all_tasks()
    jobs = [(t, l, args.seed) for t in tasks for l in args.levels]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs, maxtasksperchild=1) as pool:
        results = pool.map(_worker, jobs)

    by_task: Dict[str, Dict] = {}
    for r in results:
        by_task.setdefault(r["task"], {})[r["level"]] = r

    # category vocabulary over every scene (frozen ids; <unk> = 0). Only written for a full
    # run (all tasks) so that a partial/smoke run cannot overwrite the frozen vocabulary.
    write_frozen = args.tasks is None or args.write_frozen
    cats = sorted({c for r in results if r.get("ok") for c in r.get("obj_categories", [])} - {"<unk>"})
    vocab = {"<unk>": 0}
    vocab.update({c: i + 1 for i, c in enumerate(cats)})
    if write_frozen:
        save_json(vocab, FROZEN_DIR / "category_vocab.json")

    # substitute coverage at the category level: L3-only categories seen in other tasks
    cat_sets = {t: {l: set(d[l].get("obj_categories", [])) for l in d if d[l].get("ok")} for t, d in by_task.items()}
    for t, d in by_task.items():
        c0 = cat_sets[t].get("L0", set())
        c3 = cat_sets[t].get("L3", set())
        cov: List[str] = []
        for c in sorted(c3 - c0):
            for t2, lv in cat_sets.items():
                if t2 != t and any(c in s for s in lv.values()):
                    cov.append(f"{c}@{t2}")
        d["sub_category_coverage"] = cov

    # draft overrides: every actor name with its automatic category, flagged if <unk>
    draft = {"global": {}, "per_task": {}, "_review_needed": []}
    for t, d in by_task.items():
        for l, r in d.items():
            if not isinstance(r, dict) or not r.get("ok"):
                continue
            for n, c in zip(r.get("obj_names", []), r.get("obj_categories", [])):
                draft["per_task"].setdefault(t, {})[n] = c
                if c == "<unk>" or len(c) <= 2:
                    draft["_review_needed"].append(f"{t}/{l}/{n}")
    if write_frozen:
        save_json(draft, FROZEN_DIR / "category_overrides.draft.json")
    else:
        save_json(draft, inventory_dir() / "category_overrides.partial.json")

    out = inventory_dir() / "runtime_inventory.json"
    save_json(by_task, out)
    n_fail = sum(1 for r in results if not r.get("ok"))
    print(f"probed {len(results)} (task, level) scenes, {n_fail} failed -> {out}")
    print(f"category vocab: {len(vocab)} entries; review {FROZEN_DIR/'category_overrides.draft.json'}")
    if n_fail:
        for r in results:
            if not r.get("ok"):
                print(f"  FAIL {r['task']} {r['level']}: {r.get('error')}")


if __name__ == "__main__":
    main()
