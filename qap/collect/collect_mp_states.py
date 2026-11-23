"""Robot episodes of the IG corpus as object-centric states.

The released IG-10K LeRobot data stores only images, joint states and actions (no object
poses), so the 45-task robot corpus is regenerated here with IG's own motion-planning
solutions while recording object-centric states (qap.state.recorder). Same solutions, same
seeds as two_robot_run, CPU physics, no rendering.

    python -m qap.collect.collect_mp_states --corpus 45 --levels L0 L1 L2 L3 \
        --n-per-scene 50 --procs 32

Output: $QAP_DATA_ROOT/episodes/mp/<task>/<level>/*.npz (+ .json). Resumable: seeds whose
episode already exists are skipped. Failed planner attempts are kept (flag
meta.final_success=False) because they are valid robot transitions for d_theta; only
successes count towards --n-per-scene. L3 scenes of calibration/test tasks are skipped
unless --include-heldout.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import List, Tuple

from qap.collect.heldout import heldout_tasks, is_heldout_scene
from qap.common.paths import episodes_dir
from qap.common.tasks import all_tasks, corpus_tasks


def _done_seeds(out: Path) -> Tuple[set, int]:
    seeds, succ = set(), 0
    for p in out.glob("*.json"):
        try:
            m = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        seeds.add(int(m.get("seed", -1)))
        succ += int(bool(m.get("final_success")))
    return seeds, succ


def run_scene(task: str, level: str, n_success: int, max_attempts: int, seed0: int,
              keep_failures: bool, sim_state_every: int, out_root: str) -> dict:
    from qap.common.envs import make_env, mp_solution
    from qap.state.recorder import StateRecorder

    out = Path(out_root) / task / level
    out.mkdir(parents=True, exist_ok=True)
    done, succ = _done_seeds(out)
    stats = dict(task=task, level=level, success=succ, attempts=0, errors=0, transitions=0)
    if succ >= n_success:
        return stats
    try:
        solve = mp_solution(task, level)
        env = StateRecorder(make_env(task, level), task=task, sim_state_every=sim_state_every, level=level)
    except Exception as e:
        stats["fatal"] = f"{type(e).__name__}: {e}"
        return stats
    seed = seed0
    t0 = time.time()
    while stats["success"] < n_success and stats["attempts"] < max_attempts:
        if seed in done:
            seed += 1
            continue
        stats["attempts"] += 1
        ok = False
        try:
            left, right = solve(env, seed=seed, debug=False, vis=False)
            ev = env.unwrapped.evaluate()
            ok = bool(ev["success"].reshape(-1)[0].item()) and left != -1 and right != -1
        except Exception:
            stats["errors"] += 1
            stats.setdefault("last_error", traceback.format_exc(limit=3))
        meta = dict(source="mp", task=task, level=level, seed=seed, planner_success=ok,
                    env_id=getattr(getattr(env, "spec", None), "id", ""))
        path = env.flush(out, meta, save=ok or keep_failures)
        if path is not None:
            stats["transitions"] += int(json.load(open(path.with_suffix(".json")))["num_transitions"])
        stats["success"] += int(ok)
        seed += 1
    stats["seconds"] = round(time.time() - t0, 1)
    env.close()
    return stats


def _worker(job):
    return run_scene(*job)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="45", help="15 | 30 | 45 | all")
    ap.add_argument("--tasks", nargs="*", default=None, help="overrides --corpus")
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    ap.add_argument("--n-per-scene", type=int, default=50, help="successful episodes per (task, level)")
    ap.add_argument("--max-attempts-factor", type=float, default=3.0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--no-failures", action="store_true", help="drop failed planner attempts")
    ap.add_argument("--sim-state-every", type=int, default=5)
    ap.add_argument("--include-heldout", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tasks = args.tasks or (all_tasks() if args.corpus == "all" else corpus_tasks(args.corpus))
    held = set() if args.include_heldout else heldout_tasks()
    out_root = str(args.out or episodes_dir("mp"))
    jobs: List[tuple] = []
    for t in tasks:
        for l in args.levels:
            if is_heldout_scene(t, l, held):
                print(f"skip held-out scene {t}/{l}")
                continue
            jobs.append((t, l, args.n_per_scene, int(args.n_per_scene * args.max_attempts_factor),
                         args.seed0, not args.no_failures, args.sim_state_every, out_root))
    print(f"{len(jobs)} scenes -> {out_root}")
    ctx = mp.get_context("spawn")
    total = 0
    with ctx.Pool(args.procs, maxtasksperchild=1) as pool:
        for st in pool.imap_unordered(_worker, jobs):
            total += st.get("transitions", 0)
            print(json.dumps({k: v for k, v in st.items() if k != "last_error"}), flush=True)
            if "last_error" in st or "fatal" in st:
                print(st.get("last_error", st.get("fatal")), flush=True)
    print(f"new transitions this run: {total}")


if __name__ == "__main__":
    main()
