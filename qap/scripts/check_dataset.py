"""Check that the downloaded IG-10K dataset is complete and in the LeRobot layout the IG loaders expect.

    python -m qap.scripts.check_dataset                     # human + sim (what Weeks 1-2 use)
    python -m qap.scripts.check_dataset --robot --levels    # also real-robot and human-levels folders
    python -m qap.scripts.check_dataset --deep              # also count parquet rows vs meta/info.json

Expected layout (one LeRobot dataset per folder):
    demos/imitator_human_v1/human_H<N>/                  human videos (+ MANO), one per task
    demos/imitator_sim_v1_zed2i/L<k>_TwoRobot<Task>-v1/  simulated robot demos, one per task x level
    demos/imitator_robot_v1/robot_H<N>_L<k>/             real-robot demos (not used in Weeks 1-2)
    demos/imitator_human_v1_levels/...                   human videos per imitative level (not used in Weeks 1-2)
and inside each dataset folder:
    meta/info.json  meta/tasks.parquet  meta/episodes/  data/chunk-*/file-*.parquet  videos/<feature>/...

Which datasets are expected comes from examples/baselines/lerobot_dataset/task_mapping.json, the
same file the IG loaders and qap use. Read-only: it never modifies or downloads anything.
Exit code 1 if any expected dataset has a problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from qap.common.paths import TASK_MAPPING_FILE


def _mapping() -> List[Dict]:
    with open(TASK_MAPPING_FILE, encoding="utf-8") as f:
        return json.load(f)["task_mappings"]


def check_lerobot(d: Path, min_episodes: int, deep: bool) -> Tuple[List[str], Dict]:
    """Problems found in one LeRobot dataset folder, plus a few stats."""
    probs: List[str] = []
    st: Dict = {}
    if not d.is_dir():
        return ["missing folder"], st
    info_p = d / "meta" / "info.json"
    if not info_p.is_file():
        return ["no meta/info.json"], st
    try:
        info = json.loads(info_p.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"meta/info.json unreadable ({type(e).__name__})"], st
    st.update(version=info.get("codebase_version"), episodes=info.get("total_episodes"),
              frames=info.get("total_frames"), fps=info.get("fps"))
    if not (d / "meta" / "tasks.parquet").is_file() and not (d / "meta" / "tasks.jsonl").is_file():
        probs.append("no meta/tasks.parquet")
    if not (d / "meta" / "episodes").exists() and not (d / "meta" / "episodes.jsonl").is_file():
        probs.append("no meta/episodes/")
    parquets = sorted((d / "data").rglob("*.parquet")) if (d / "data").is_dir() else []
    if not parquets:
        probs.append("no data/*.parquet")
    eps = st["episodes"]
    if not isinstance(eps, int) or eps < min_episodes:
        probs.append(f"total_episodes={eps} < {min_episodes} (configs use episodes 0:{min_episodes})")
    # every video feature must have files under videos/<feature>/
    vkeys = [k for k, v in (info.get("features") or {}).items() if isinstance(v, dict) and v.get("dtype") == "video"]
    st["video_features"] = len(vkeys)
    missing_v = [k for k in vkeys if not any((d / "videos" / k).rglob("*.mp4"))]
    if missing_v:
        probs.append(f"no videos for {len(missing_v)}/{len(vkeys)} video features, e.g. {missing_v[0]}")
    # truncated downloads show up as empty files
    empty = [p for p in d.rglob("*") if p.is_file() and p.stat().st_size == 0]
    if empty:
        probs.append(f"{len(empty)} empty file(s), e.g. {empty[0].relative_to(d)}")
    if deep and parquets:
        try:
            import pyarrow.parquet as pq

            rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquets)
            st["rows"] = rows
            if isinstance(st["frames"], int) and rows != st["frames"]:
                probs.append(f"parquet rows {rows} != total_frames {st['frames']}")
        except Exception as e:
            probs.append(f"parquet read failed ({type(e).__name__}: {e})")
    return probs, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("demos"))
    ap.add_argument("--human-dir", default="imitator_human_v1")
    ap.add_argument("--sim-dir", default="imitator_sim_v1_zed2i")
    ap.add_argument("--robot-dir", default="imitator_robot_v1")
    ap.add_argument("--levels-dir", default="imitator_human_v1_levels")
    ap.add_argument("--min-episodes", type=int, default=50)
    ap.add_argument("--robot", action="store_true", help="also check real-robot datasets")
    ap.add_argument("--levels", action="store_true", help="also list the human-levels folder")
    ap.add_argument("--deep", action="store_true", help="count parquet rows (needs pyarrow; slower)")
    ap.add_argument("--verbose", action="store_true", help="print every dataset, not only problems")
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        sys.exit(f"{root} not found; run from the repository root or pass --root")

    expected: List[Tuple[str, Path]] = []      # (group, path)
    tasks = []
    for e in _mapping():
        sims = e.get("sim_task_id") or []
        if not sims:
            continue                          # human-only entries (no sim task): not used by qap
        tasks.append(sims[0])
        expected.append(("human", root / args.human_dir / e["human_task_id"]))
        expected += [("sim", root / args.sim_dir / s) for s in sims]
        if args.robot:
            expected += [("robot", root / args.robot_dir / r) for r in e.get("robot_task_id", [])]

    n_bad = 0
    per_group: Dict[str, List[int]] = {}
    for group, d in expected:
        probs, st = check_lerobot(d, args.min_episodes, args.deep)
        ok_bad = per_group.setdefault(group, [0, 0])
        ok_bad[1 if probs else 0] += 1
        if probs:
            n_bad += 1
            print(f"[BAD] {group:5s} {d.relative_to(root)}: " + "; ".join(probs))
        elif args.verbose:
            print(f"[ok]  {group:5s} {d.relative_to(root)}: v{st.get('version')} {st.get('episodes')} eps, "
                  f"{st.get('frames')} frames, {st.get('fps')} fps, {st.get('video_features')} video features"
                  + (f", {st['rows']} rows" if "rows" in st else ""))

    # folders present on disk but not expected (e.g. human-only tasks H25/H37/H38)
    known = {d for _, d in expected}
    extra = []
    for sub in [args.human_dir, args.sim_dir] + ([args.robot_dir] if args.robot else []):
        p = root / sub
        if p.is_dir():
            extra += [c for c in sorted(p.iterdir()) if c.is_dir() and c not in known]
    if args.levels:
        p = root / args.levels_dir
        kids = sorted(c.name for c in p.iterdir()) if p.is_dir() else []
        print(f"\n{args.levels_dir}: {len(kids)} entries, e.g. {kids[:5]} (not used in Weeks 1-2)")

    incomplete = list((root / ".cache").rglob("*.incomplete")) if (root / ".cache").is_dir() else []

    print(f"\n{len(tasks)} sim tasks in task_mapping.json")
    for g, (ok, bad) in per_group.items():
        print(f"  {g:5s}: {ok} ok, {bad} with problems")
    if extra:
        print(f"  extra folders not used by qap ({len(extra)}): {[str(x.relative_to(root)) for x in extra[:10]]}")
    if incomplete:
        print(f"  WARNING: {len(incomplete)} unfinished download(s) in {root/'.cache'}; re-run the hf download command")
    print("DATASET OK" if n_bad == 0 and not incomplete else f"DATASET HAS PROBLEMS ({n_bad} dataset(s))")
    sys.exit(1 if n_bad or incomplete else 0)


if __name__ == "__main__":
    main()
