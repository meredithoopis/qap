"""Planner-certified realisation feasibility for (state, affordance, goal) triples (Week 2).

SoRB-style protocol (Eysenbach et al. 2019) adapted to affordances. For every retained task,
graded scene G0-G4 and seed:
  planner      s = reset state, run the scene's IG solver; w = first gripper-contact onset
               (entry state of the affordance actually used), g = final state.
               Successful runs give positives; failed runs are stored with goal_valid=False
               (their final state is not a goal) and only serve as G4 start states.
  planner_mid  restart the same solver from a random mid-episode state (restored simulator
               state) of the first run; label = success from there; if it fails, the goal is
               the first run's (successful) final state.
  reverse_irr  (g -> s) when the run latched an irreversible event (poured, cut, ...):
               label 0 by construction (cannot un-pour).
  g4_vs_g3     start of the G4 scene (substitute parked out of reach) with the goal reached
               in G3 for the same seed: label 0 (no feasible affordance).
Labels are planner certificates: label 1 = the IG motion planner realised it.

    python -m qap.feasibility.label_triples --procs 32 --seeds 20 [--images]

Output: $QAP_DATA_ROOT/labels/triples/<task>_<grade>.npz (+ .json) and an index.
--images also stores 224x224 zed2i RGB of s and g for the DINOv2 visual-cosine baseline
(needs a renderer on the machine).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from qap.common.io import load_json, save_json
from qap.common.paths import FROZEN_DIR, labels_dir
from qap.common.timeout import time_limit, trial_timeout
from qap.scenes.graded_specs import GRADES, grade_spec
from qap.state.schema import EVENT_INDEX, LATCHED_EVENTS, OBJ_DIM

MAXN = 16
SOURCES = ["planner", "planner_mid", "reverse_irr", "g4_vs_g3"]


def _pad(obj: np.ndarray) -> np.ndarray:
    o = np.zeros((MAXN, OBJ_DIM), np.float32)
    n = min(MAXN, obj.shape[0])
    o[:n] = obj[:n]
    return o


def _state(arr: Dict[str, np.ndarray], t: int) -> Dict[str, np.ndarray]:
    return dict(robot=arr["robot"][t], obj=_pad(arr["obj"][t]), events=arr["events"][t].astype(np.float32))


def _latched(ev: np.ndarray) -> np.ndarray:
    idx = [EVENT_INDEX[e] for e in LATCHED_EVENTS]
    return ev[idx] > 0.5


def label_scene(task: str, grade: str, seeds: List[int], images: bool, mid: bool, out_dir: str) -> Dict:
    from qap.feasibility.start_state import StartStateOverride, capture_rgb
    from qap.scenes.graded import make_graded_env
    from qap.state.recorder import StateRecorder

    res = dict(task=task, grade=grade, n=0, pos=0, errors=0)
    try:
        env, solve, spec = make_graded_env(task, grade, obs_mode="rgb" if images else "none")
    except Exception as e:
        res["fatal"] = f"{type(e).__name__}: {e}"
        return res
    over = StartStateOverride(env)
    rec = StateRecorder(over, task=task, sim_state_every=3, level=str(spec["level"]))
    rows: List[Dict] = []
    finals: Dict[int, Dict] = {}

    def run(seed: int, start: Optional[np.ndarray]):
        over.set_next_start(start)
        ok, err = False, None
        img_s = img_g = None
        try:
            with time_limit(trial_timeout()):
                left, right = solve(rec, seed=seed, debug=False, vis=False)
            ev = rec.unwrapped.evaluate()
            ok = bool(ev["success"].reshape(-1)[0].item()) and not (isinstance(left, int) or isinstance(right, int))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        arr = rec.episode_arrays()
        if images and arr:
            img_g = capture_rgb(rec)
        cats = arr.get("obj_cat") if arr else None
        rec.flush(Path(out_dir), {}, save=False)
        return ok, err, arr, cats, img_g

    for seed in seeds:
        try:
            ok, err, arr, cats, img_g = run(seed, None)
            if not arr:
                res["errors"] += 1
                continue
            img_s = None
            if images:   # re-render the start state: reset to the same seed without running
                over.set_next_start(None)
                rec.reset(seed=seed)
                img_s = capture_rgb(rec)
                rec.flush(Path(out_dir), {}, save=False)
            T = arr["robot"].shape[0] - 1
            t_w = int(arr["onsets"][0, 0]) if len(arr["onsets"]) else -1
            s, g = _state(arr, 0), _state(arr, T)
            w = _state(arr, t_w) if t_w >= 0 else None
            cat = np.zeros(MAXN, np.int64)
            cat[: min(MAXN, len(cats))] = cats[:MAXN]
            # A failed run's final state is not in the goal set, so its row is kept only as a
            # start-state carrier (goal_valid=False; used for g4_vs_g3, excluded from PR-AUC).
            rows.append(dict(s=s, w=w, g=g, cat=cat, label=int(ok), source="planner", seed=seed, err=err,
                             img_s=img_s, img_g=img_g, goal_valid=ok))
            if ok:
                finals[seed] = dict(g=g, img=img_g)
            # reverse of an irreversible run
            if ok and (_latched(g["events"]) & ~_latched(s["events"])).any():
                rows.append(dict(s=g, w=w, g=s, cat=cat, label=0, source="reverse_irr", seed=seed, err=None,
                                 img_s=img_g, img_g=img_s))
            # restart from a mid-episode state. Skipped after a failed G4 run: G4 has no reachable
            # substitute, so the restart fails too and (without a successful final state to use as
            # goal) yields no row, costing a full trial timeout per seed.
            if mid and (ok or grade != "G4") and len(arr["sim_t"]) > 2:
                cands = [i for i, t in enumerate(arr["sim_t"]) if 0 < t < 0.6 * T]
                if cands:
                    i = cands[np.random.default_rng(seed).integers(len(cands))]
                    ok2, err2, arr2, _, img_g2 = run(seed, arr["sim_state"][i])
                    if arr2:
                        T2 = arr2["robot"].shape[0] - 1
                        t_w2 = int(arr2["onsets"][0, 0]) if len(arr2["onsets"]) else -1
                        g2 = _state(arr2, T2) if ok2 else (finals[seed]["g"] if seed in finals else None)
                        if g2 is not None:
                            rows.append(dict(s=_state(arr2, 0), w=_state(arr2, t_w2) if t_w2 >= 0 else None,
                                             g=g2, cat=cat, label=int(ok2), source="planner_mid", seed=seed,
                                             err=err2, img_s=None, img_g=img_g2 if ok2 else img_g))
        except Exception:
            res["errors"] += 1
            res["last_error"] = traceback.format_exc(limit=3)

    if rows:
        _save_rows(rows, Path(out_dir) / f"{task}_{grade}", dict(task=task, grade=grade, spec_level=spec["level"]))
    res["n"] = len(rows)
    res["pos"] = sum(r["label"] for r in rows)
    try:
        env.close()
    except Exception:
        pass
    return res


def _save_rows(rows: List[Dict], stem: Path, meta: Dict) -> None:
    n = len(rows)
    zero = dict(robot=np.zeros((2, 16), np.float32), obj=np.zeros((MAXN, OBJ_DIM), np.float32),
                events=np.zeros_like(rows[0]["s"]["events"]))
    arrays = {}
    for key in ("s", "w", "g"):
        for f in ("robot", "obj", "events"):
            arrays[f"{key}_{f}"] = np.stack([(r[key] or zero)[f] for r in rows])
    arrays["has_w"] = np.array([r["w"] is not None for r in rows])
    arrays["cat"] = np.stack([r["cat"] for r in rows])
    arrays["label"] = np.array([r["label"] for r in rows], np.int8)
    arrays["source"] = np.array([SOURCES.index(r["source"]) for r in rows], np.int8)
    arrays["seed"] = np.array([r["seed"] for r in rows], np.int32)
    arrays["goal_valid"] = np.array([r.get("goal_valid", True) for r in rows], bool)
    imgs_s = [r.get("img_s") for r in rows]
    imgs_g = [r.get("img_g") for r in rows]
    if any(x is not None for x in imgs_s + imgs_g):
        ref = next(x for x in imgs_s + imgs_g if x is not None)
        blank = np.zeros_like(ref)
        arrays["img_s"] = np.stack([x if x is not None else blank for x in imgs_s])
        arrays["img_g"] = np.stack([x if x is not None else blank for x in imgs_g])
        arrays["has_img"] = np.array([(a is not None and b is not None) for a, b in zip(imgs_s, imgs_g)])
    np.savez_compressed(f"{stem}.npz", **arrays)
    save_json(dict(meta, sources=SOURCES, n=n), Path(f"{stem}.json"))


def add_g4_vs_g3(out_dir: Path) -> int:
    """Cross-grade negatives: G4 start + G3 goal of the same task and seed (label 0)."""
    added = 0
    for f3 in out_dir.glob("*_G3.npz"):
        task = f3.name[: -len("_G3.npz")]
        f4 = out_dir / f"{task}_G4.npz"
        if not f4.exists():
            continue
        a3, a4 = dict(np.load(f3)), dict(np.load(f4))
        rows = []
        for i in np.nonzero((a3["source"] == 0) & (a3["label"] == 1))[0]:
            j = np.nonzero((a4["source"] == 0) & (a4["seed"] == a3["seed"][i]))[0]
            if len(j) == 0:
                continue
            j = j[0]
            rows.append((j, i))
        if not rows:
            continue
        out = {}
        for k in ("s_robot", "s_obj", "s_events", "cat"):
            out[k] = np.stack([a4[k][j] for j, _ in rows])
        for k in ("w_robot", "w_obj", "w_events", "g_robot", "g_obj", "g_events"):
            out[k] = np.stack([a3[k][i] for _, i in rows])
        out["has_w"] = np.array([a3["has_w"][i] for _, i in rows])
        out["label"] = np.zeros(len(rows), np.int8)
        out["source"] = np.full(len(rows), SOURCES.index("g4_vs_g3"), np.int8)
        out["seed"] = np.array([a3["seed"][i] for _, i in rows], np.int32)
        out["goal_valid"] = np.ones(len(rows), bool)
        if "img_s" in a4 and "img_g" in a3:
            out["img_s"] = np.stack([a4["img_s"][j] for j, _ in rows])
            out["img_g"] = np.stack([a3["img_g"][i] for _, i in rows])
            out["has_img"] = np.ones(len(rows), bool)
        np.savez_compressed(out_dir / f"{task}_G4xG3.npz", **out)
        save_json(dict(task=task, grade="G4xG3", sources=SOURCES, n=len(rows)), out_dir / f"{task}_G4xG3.json")
        added += len(rows)
    return added


def _worker(job):
    return label_scene(*job)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--grades", nargs="+", default=list(GRADES))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=5000, help="disjoint from training-episode seeds")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--no-mid", action="store_true")
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    if args.tasks:
        tasks = args.tasks
    else:
        fr = load_json(FROZEN_DIR / "frozen_tasks.json")
        tasks = fr.get("retained_after_shortcut") or [t["task"] for t in fr["retained"]]
    out = labels_dir() / "triples"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    jobs = [(t, g, seeds, args.images, not args.no_mid, str(out))
            for t in tasks for g in args.grades if grade_spec(t, g) is not None]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            print(json.dumps({k: v for k, v in r.items() if k != "last_error"}), flush=True)
    n = add_g4_vs_g3(out)
    print(f"added {n} g4_vs_g3 negatives; labels in {out}")


if __name__ == "__main__":
    main()
