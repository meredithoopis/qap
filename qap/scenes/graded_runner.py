"""Graded scenes G0-G4 for retained tasks: validate, generate episodes, planner feasibility.

    # 1) check every (task, grade) instantiates and how often its solver succeeds
    python -m qap.scenes.graded_runner validate --trials 5
    # 2) 50 robot episodes per training configuration (task x grade), object-centric states
    python -m qap.scenes.graded_runner episodes --n-per-config 50 --procs 32
    # 3) planner-certified feasibility for every (step, affordance) in every graded scene
    python -m qap.scenes.graded_runner feasibility --trials 10 --procs 32

Outputs under $QAP_DATA_ROOT: graded/validation.json, episodes/graded/<task>/<grade>/,
labels/planner_feasibility.json. Single-step tasks have one step (step 0); the affordances
are 'copy' (demonstrated object, IG L0 solver) and 'substitute' (L3 object, IG L3 solver).
A solver that cannot even run in a scene (missing object attributes) is recorded as
status 'n/a' (not applicable), which is distinct from a planning failure.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from qap.common.io import load_json, save_json
from qap.common.paths import FROZEN_DIR, episodes_dir, labels_dir, sub
from qap.scenes.graded_specs import GRADES, grade_spec

GT_LABEL = {"G0": "C", "G1": "C", "G2": "S", "G3": "S", "G4": "A"}


def retained_tasks() -> List[str]:
    fr = load_json(FROZEN_DIR / "frozen_tasks.json")
    return list(fr.get("retained_after_shortcut") or [t["task"] for t in fr["retained"]])


def _success(env) -> bool:
    ev = env.unwrapped.evaluate()
    return bool(ev["success"].reshape(-1)[0].item())


def run_trials(task: str, grade: str, solver_level: Optional[str], seeds: List[int],
               record_dir: Optional[str] = None, n_success_stop: Optional[int] = None,
               keep_failures: bool = True) -> Dict:
    from qap.common.envs import mp_solution
    from qap.scenes.graded import make_graded_env
    from qap.state.recorder import StateRecorder

    from qap.common.timeout import TrialTimeout, time_limit, trial_timeout

    out = dict(task=task, grade=grade, solver=solver_level, trials=0, success=0, errors=0, timeouts=0,
               error_msgs=[])
    try:
        env, solve, spec = make_graded_env(task, grade)
    except Exception as e:
        out.update(status="build_error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=4))
        return out
    out["spec"] = {k: v for k, v in spec.items() if k != "task"}
    if solver_level is not None:
        try:
            solve = mp_solution(task, solver_level)
        except KeyError as e:
            out.update(status="no_solver", error=str(e))
            return out
    rec = StateRecorder(env, task=task, level=str(spec["level"])) if record_dir else env
    t0 = time.time()
    for seed in seeds:
        if n_success_stop is not None and out["success"] >= n_success_stop:
            break
        out["trials"] += 1
        ok = False
        try:
            with time_limit(trial_timeout()):
                left, right = solve(rec, seed=seed, debug=False, vis=False)
            ok = _success(rec) and not (isinstance(left, int) or isinstance(right, int))
        except TrialTimeout:
            out["timeouts"] += 1          # a planning failure, not 'solver cannot run'
        except Exception as e:
            out["errors"] += 1
            if len(out["error_msgs"]) < 3:
                out["error_msgs"].append(f"{type(e).__name__}: {e}")
        out["success"] += int(ok)
        if record_dir:
            rec.flush(Path(record_dir) / task / grade.upper(),
                      dict(source="graded", task=task, grade=grade.upper(), level=spec["level"], seed=seed,
                           solver=solver_level or spec["solver"], planner_success=ok,
                           gt_label=GT_LABEL.get(grade.upper())), save=ok or keep_failures)
    out["seconds"] = round(time.time() - t0, 1)
    out["status"] = "n/a" if out["errors"] == out["trials"] and out["trials"] > 0 else "ok"
    try:
        env.close()
    except Exception:
        pass
    return out


def _worker(job):
    return run_trials(**job)


def _pool_run(jobs: List[Dict], procs: int) -> List[Dict]:
    ctx = mp.get_context("spawn")
    res = []
    with ctx.Pool(procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            print(json.dumps({k: v for k, v in r.items() if k not in ("traceback", "spec")}), flush=True)
            res.append(r)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "episodes", "feasibility"):
        p = sp.add_parser(name)
        p.add_argument("--tasks", nargs="*", default=None)
        p.add_argument("--grades", nargs="+", default=list(GRADES))
        p.add_argument("--procs", type=int, default=16)
        p.add_argument("--seed0", type=int, default=1000)
        if name == "validate":
            p.add_argument("--trials", type=int, default=5)
        if name == "episodes":
            p.add_argument("--n-per-config", type=int, default=50)
            p.add_argument("--max-attempts-factor", type=float, default=3.0)
            p.add_argument("--no-failures", action="store_true")
        if name == "feasibility":
            p.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    tasks = args.tasks or retained_tasks()
    pairs = [(t, g) for t in tasks for g in args.grades if grade_spec(t, g) is not None]
    skipped = [(t, g) for t in tasks for g in args.grades if grade_spec(t, g) is None]
    if skipped:
        print(f"no spec (skipped): {skipped}")

    if args.cmd == "validate":
        jobs = [dict(task=t, grade=g, solver_level=None, seeds=list(range(args.seed0, args.seed0 + args.trials)))
                for t, g in pairs]
        res = _pool_run(jobs, args.procs)
        out = sub("graded") / "validation.json"
        save_json(res, out)
        print(f"\n{'task':24s} " + " ".join(f"{g:>6s}" for g in args.grades))
        table: Dict[str, Dict[str, str]] = {}
        for r in res:
            cell = r.get("status", "?") if r.get("status") != "ok" else f"{r['success']}/{r['trials']}"
            table.setdefault(r["task"], {})[r["grade"]] = cell
        for t in tasks:
            print(f"{t:24s} " + " ".join(f"{table.get(t, {}).get(g, '-'):>6s}" for g in args.grades))
        print(f"-> {out}\nExpected: G0-G3 mostly succeed; G4 must fail (no reachable substitute).")

    elif args.cmd == "episodes":
        n = args.n_per_config
        jobs = [dict(task=t, grade=g, solver_level=None,
                     seeds=list(range(args.seed0, args.seed0 + int(n * args.max_attempts_factor))),
                     record_dir=str(episodes_dir("graded")),
                     n_success_stop=(n if g.upper() != "G4" else None), keep_failures=not args.no_failures)
                for t, g in pairs]
        # G4 has no successes by design: record n attempts (the robot's failed tries) instead
        for j in jobs:
            if j["grade"].upper() == "G4":
                j["seeds"] = j["seeds"][:n]
        _pool_run(jobs, args.procs)

    else:  # feasibility
        seeds = list(range(args.seed0, args.seed0 + args.trials))
        jobs = []
        for t, g in pairs:
            for aff, lvl in (("copy", "L0"), ("substitute", "L3")):
                jobs.append(dict(task=t, grade=g, solver_level=lvl, seeds=seeds))
        res = _pool_run(jobs, args.procs)
        from qap.week1.inventory_spec import INVENTORY_BY_NAME

        table = []
        for r in res:
            inv = INVENTORY_BY_NAME.get(r["task"])
            aff = "copy" if r["solver"] == "L0" else "substitute"
            table.append(dict(task=r["task"], grade=r["grade"], step=0, affordance=aff,
                              affordance_object=(inv.demo if aff == "copy" else inv.sub) if inv else "",
                              trials=r["trials"], success=r["success"], status=r.get("status"),
                              feasible=bool(r["success"] > 0), success_rate=r["success"] / max(1, r["trials"]),
                              gt_label=GT_LABEL.get(r["grade"].upper()), errors=r.get("error_msgs", [])))
        out = labels_dir() / "planner_feasibility.json"
        save_json(table, out)
        print(f"-> {out} ({len(table)} (task, grade, step, affordance) records)")


if __name__ == "__main__":
    main()
