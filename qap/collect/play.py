"""Robot play with the IG motion planner: random waypoint compositions over all task assets.

    python -m qap.collect.play --target-transitions 1000000 --procs 48
    python -m qap.collect.play --count            # how many transitions exist so far

Each play episode resets a task scene (any IG task, level L0-L3, random seed) and executes
3-8 random primitives (pick / place / tilt / push / press / drag articulation / reach /
home) with the nearest arm, choosing objects and targets at random among the scene's
assets. States are recorded with qap.state.recorder, so contact onsets of play episodes
also provide entry states W_a for the test-time graph.

Jobs are (task, level, n_episodes); one fresh process per job (level flags are globals).
Resumable: re-running adds episodes until --target-transitions is reached. L3 scenes of
calibration/test tasks are excluded unless --include-heldout.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import traceback
import zlib
from pathlib import Path
from typing import List

import numpy as np

from qap.collect.heldout import heldout_tasks, is_heldout_scene
from qap.common.io import count_transitions
from qap.common.paths import episodes_dir
from qap.common.tasks import all_tasks

PRIMS_FREE = ["pick", "pick", "pick", "push", "press", "articulate", "reach", "home"]
PRIMS_HELD = ["place", "place", "place_on", "tilt", "reach_carry", "drop"]


def _targets(env):
    from qap.collect.primitives import Target
    from qap.state.extractor import ObjectCentricExtractor

    ex = ObjectCentricExtractor(env)
    ex.reset()
    return [Target(s.name, s.kind, s.handle, dynamic=(s.kind == "actor" and not s.is_static)) for s in ex.slots]


def _workspace(arms, targets):
    ps = np.array([arms.pos(t) for t in targets]) if targets else np.zeros((1, 3))
    lo = ps.min(0) - np.array([0.12, 0.12, 0.0])
    hi = ps.max(0) + np.array([0.12, 0.12, 0.0])
    lo[2], hi[2] = max(0.03, ps[:, 2].min() + 0.03), ps[:, 2].max() + 0.25
    return lo, hi


def play_episode(env, rng: np.random.Generator, n_prims: int, max_steps: int) -> List[str]:
    from qap.collect.primitives import PlayArms

    arms = PlayArms(env, rng)
    targets = _targets(env)
    box = _workspace(arms, targets)
    log: List[str] = []
    for _ in range(n_prims):
        if env.num_steps >= max_steps:
            break
        held = [i for i in (0, 1) if arms.holding[i] is not None]
        try:
            if held:
                i = held[0]
                prim = rng.choice(PRIMS_HELD)
                if prim == "place":
                    xyz = rng.uniform(box[0], box[1])
                    xyz[2] = box[0][2] + 0.05
                    ok = arms.place(i, xyz)
                elif prim == "place_on":            # put / stack onto (or into) another object
                    others = [t for t in targets if t is not arms.holding[i]]
                    t2 = others[rng.integers(len(others))] if others else None
                    ok = t2 is not None and arms.place(i, arms.pos(t2) + np.array([0, 0, 0.12]))
                elif prim == "tilt":
                    ok = arms.tilt(i, float(rng.uniform(np.deg2rad(50), np.deg2rad(110)) * rng.choice([-1, 1])))
                elif prim == "reach_carry":
                    ok = arms.reach_random(i, box)
                else:  # drop
                    arms.open(i)
                    ok = True
            else:
                prim = rng.choice(PRIMS_FREE)
                t = targets[rng.integers(len(targets))] if targets else None
                if prim in ("reach", "home") or t is None:
                    i = int(rng.integers(2))
                    ok = arms.go_home(i) if prim == "home" else arms.reach_random(i, box)
                else:
                    i = arms.nearest_arm(arms.pos(t))
                    if prim == "pick":
                        movable = [x for x in targets if x.dynamic]
                        t = movable[rng.integers(len(movable))] if movable else t
                        i = arms.nearest_arm(arms.pos(t))
                        ok = arms.pick(i, t)
                    elif prim == "push":
                        ok = arms.push(i, t)
                    elif prim == "press":
                        ok = arms.press(i, t)
                    else:
                        arts = [x for x in targets if x.kind == "articulation"]
                        t = arts[rng.integers(len(arts))] if arts else t
                        ok = arms.articulate(arms.nearest_arm(arms.pos(t)), t)
            log.append(f"{prim}:{int(bool(ok))}")
        except Exception as e:
            log.append(f"{type(e).__name__}")
    for i in (0, 1):
        if arms.holding[i] is not None and rng.random() < 0.5:
            arms.open(i)
    return log


def run_job(task: str, level: str, n_episodes: int, seed0: int, max_steps: int, out_root: str) -> dict:
    from qap.common.envs import make_env
    from qap.state.recorder import StateRecorder

    out = Path(out_root) / task / level
    out.mkdir(parents=True, exist_ok=True)
    st = dict(task=task, level=level, episodes=0, transitions=0, errors=0)
    try:
        env = StateRecorder(make_env(task, level), task=task, max_steps=max_steps, level=level)
    except Exception as e:
        st["fatal"] = f"{type(e).__name__}: {e}"
        return st
    t0 = time.time()
    for k in range(n_episodes):
        seed = seed0 + k
        rng = np.random.default_rng(seed * 7919 + zlib.crc32(f"{task}/{level}".encode()) % 1000)
        try:
            env.reset(seed=seed)
            log = play_episode(env, rng, int(rng.integers(3, 9)), max_steps)
        except Exception:
            st["errors"] += 1
            st["last_error"] = traceback.format_exc(limit=3)
            log = ["episode_error"]
        p = env.flush(out, dict(source="play", task=task, level=level, seed=seed, program=log), min_transitions=10)
        if p is not None:
            st["episodes"] += 1
            st["transitions"] += json.load(open(p.with_suffix(".json")))["num_transitions"]
    st["seconds"] = round(time.time() - t0, 1)
    env.close()
    return st


def _worker(job):
    return run_job(*job)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-transitions", type=int, default=1_000_000)
    ap.add_argument("--est-episode-len", type=int, default=250)
    ap.add_argument("--episodes-per-job", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--include-heldout", action="store_true")
    ap.add_argument("--seed-base", type=int, default=100_000, help="kept disjoint from MP seeds")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_root = args.out or episodes_dir("play")
    have = count_transitions(out_root)
    print(f"existing play transitions: {have}")
    if args.count:
        return
    need = args.target_transitions - have
    if need <= 0:
        print("target reached")
        return
    held = set() if args.include_heldout else heldout_tasks()
    scenes = [(t, l) for t in (args.tasks or all_tasks()) for l in args.levels if not is_heldout_scene(t, l, held)]
    n_eps = math.ceil(need / args.est_episode_len)
    n_jobs = math.ceil(n_eps / args.episodes_per_job)
    rng = np.random.default_rng(int(time.time()))
    # resume-safe seeds: each run starts from a fresh, time-derived block
    block = args.seed_base + int(rng.integers(1, 10_000_000)) * 1000
    jobs = []
    for j in range(n_jobs):
        t, l = scenes[j % len(scenes)]
        jobs.append((t, l, args.episodes_per_job, block + j * args.episodes_per_job, args.max_steps, str(out_root)))
    print(f"{len(scenes)} scenes, {len(jobs)} jobs x {args.episodes_per_job} episodes (~{n_eps * args.est_episode_len} transitions)")
    ctx = mp.get_context("spawn")
    got = 0
    with ctx.Pool(args.procs, maxtasksperchild=1) as pool:
        for st in pool.imap_unordered(_worker, jobs):
            got += st.get("transitions", 0)
            print(json.dumps({k: v for k, v in st.items() if k != "last_error"}) + f"  total_new={got}", flush=True)
    print(f"play transitions now: {count_transitions(out_root)}")


if __name__ == "__main__":
    main()
