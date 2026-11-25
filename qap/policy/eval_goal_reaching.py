"""Goal reaching with oracle goals on single-step tasks (Week 2 go/no-go).

For each (task, grade, seed): the IG planner solves the scene once to produce the oracle
subgoals - the entry state w (first gripper-contact onset) and the final goal state g. The
scene is then reset to the same seed and the learned policy pi(a | s, subgoal) is rolled out:
first towards w, then towards g (switching when w is reached or its step budget runs out).

Reached w : the arm that made contact in the oracle run touches the same object.
Reached g : the task's success predicate (env.evaluate()['success']) holds.
Go/no-go  : majority of trials reach the oracle subgoals.

    python -m qap.policy.eval_goal_reaching --policy qap_runs/runs/awr_qrl/policy.pt \
        --grades G0 G3 --seeds 10 --procs 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

from qap.common.io import load_json, save_json
from qap.common.paths import FROZEN_DIR, labels_dir
from qap.common.timeout import time_limit, trial_timeout
from qap.scenes.graded_specs import grade_spec


def rollout(task: str, grade: str, seeds: List[int], policy_path: str, budget_w: int, budget_g: int,
            device: str) -> Dict:
    import torch

    from qap.policy.awr import load_policy
    from qap.quasimetric.data import state_from_arrays
    from qap.scenes.graded import make_graded_env
    from qap.state.recorder import StateRecorder

    res = dict(task=task, grade=grade, trials=[])
    try:
        env, solve, spec = make_graded_env(task, grade)
    except Exception as e:
        res["fatal"] = f"{type(e).__name__}: {e}"
        return res
    rec = StateRecorder(env, task=task, level=str(spec["level"]))
    pol = load_policy(Path(policy_path), device)
    keys = [f"panda_wristcam-{i}" for i in (0, 1)]

    def to_state(robot, obj, events, cat):
        return state_from_arrays(robot, obj, cat, events, device=device)

    for seed in seeds:
        tr = dict(seed=seed)
        try:
            # 1) oracle run with the IG planner
            with time_limit(trial_timeout()):
                left, right = solve(rec, seed=seed, debug=False, vis=False)
            arr = rec.episode_arrays()
            ok = bool(rec.unwrapped.evaluate()["success"].reshape(-1)[0].item())
            rec.flush(Path("."), {}, save=False)
            tr["oracle_success"] = ok
            if not ok or not arr:
                res["trials"].append(tr)
                continue
            T = arr["robot"].shape[0] - 1
            onset = arr["onsets"][0] if len(arr["onsets"]) else None
            cat = arr["obj_cat"]
            goal_g = to_state(arr["robot"][T], arr["obj"][T], arr["events"][T].astype(np.float32), cat)
            goal_w = to_state(arr["robot"][int(onset[0])], arr["obj"][int(onset[0])],
                              arr["events"][int(onset[0])].astype(np.float32), cat) if onset is not None else None
            # 2) policy rollout from the same initial state
            rec.reset(seed=seed)
            reached_w = onset is None
            steps_w = 0
            reached_g = False
            for t in range(budget_w + budget_g):
                s = rec.ex.extract(rec.last_info)
                st = to_state(s["robot"], s["obj"], s["events"].astype(np.float32), cat)
                if not reached_w:
                    if bool(s["contact"][int(onset[1]), int(onset[2])]):
                        reached_w = True
                    elif steps_w >= budget_w:
                        pass                         # give up on w, continue towards g
                goal = goal_w if (not reached_w and steps_w < budget_w) else goal_g
                steps_w += int(not reached_w)
                a = pol.act(st, goal)[0].cpu().numpy()
                rec.step({keys[0]: a[:8], keys[1]: a[8:]})
                if bool(rec.unwrapped.evaluate()["success"].reshape(-1)[0].item()):
                    reached_g = True
                    break
            rec.flush(Path("."), {}, save=False)
            tr.update(reached_w=bool(reached_w), reached_g=reached_g, steps=t + 1)
        except Exception as e:
            tr["error"] = f"{type(e).__name__}: {e}"
            tr["traceback"] = traceback.format_exc(limit=3)
        res["trials"].append(tr)
    env.close()
    return res


def _worker(job):
    return rollout(*job)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--grades", nargs="+", default=["G0", "G3"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--budget-w", type=int, default=200)
    ap.add_argument("--budget-g", type=int, default=400)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="policy device inside worker processes")
    args = ap.parse_args()
    if args.tasks:
        tasks = args.tasks
    else:
        fr = load_json(FROZEN_DIR / "frozen_tasks.json")
        tasks = fr.get("retained_after_shortcut") or [t["task"] for t in fr["retained"]]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    jobs = [(t, g, seeds, args.policy, args.budget_w, args.budget_g, args.device)
            for t in tasks for g in args.grades if grade_spec(t, g) is not None]
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            valid = [t for t in r.get("trials", []) if t.get("oracle_success") and "error" not in t]
            rw = np.mean([t["reached_w"] for t in valid]) if valid else float("nan")
            rg = np.mean([t["reached_g"] for t in valid]) if valid else float("nan")
            print(f"{r['task']:24s} {r['grade']}  oracle-ok {len(valid)}/{len(r.get('trials', []))}  "
                  f"reach_w {rw:.2f}  reach_g {rg:.2f}  {r.get('fatal', '')}", flush=True)
            results.append(r)
    allv = [t for r in results for t in r.get("trials", []) if t.get("oracle_success") and "error" not in t]
    summary = dict(n=len(allv), reach_w=float(np.mean([t["reached_w"] for t in allv])) if allv else None,
                   reach_g=float(np.mean([t["reached_g"] for t in allv])) if allv else None)
    summary["go_no_go_majority"] = bool(summary["reach_w"] is not None and summary["reach_w"] > 0.5)
    out = labels_dir() / f"goal_reaching_{Path(args.policy).parent.name}.json"
    save_json(dict(summary=summary, results=results), out)
    print(json.dumps(summary, indent=1), f"\n-> {out}")


if __name__ == "__main__":
    main()
