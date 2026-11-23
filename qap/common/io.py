"""Episode storage: one compressed .npz per episode + a JSON sidecar.

Arrays (T = number of actions, states are T+1):
    robot      float32 [T+1, 2, ROBOT_DIM]   per-arm proprioception (see qap.state.schema)
    obj        float32 [T+1, N, OBJ_DIM]     per-object token features
    obj_cat    int32   [N]                   category ids (qap.state.categories vocab)
    events     uint8   [T+1, E]              latched scene events (poured, cut, ...)
    preds      uint8   [T+1, P]              raw boolean task predicates from env.evaluate()
    success    uint8   [T+1]
    action     float32 [T, A]                pd_joint_pos action (16 = 2 x (7 joints + gripper))
    contact    uint8   [T+1, 2, N]           gripper (fingers+hand) in contact with object, per arm
    onsets     int32   [K, 3]                contact onsets (t, arm, obj) => entry states W_a
    sim_t      int32   [S]                   time indices with a saved simulator state
    sim_state  float32 [S, D]                env.get_state() at those indices (for set_state)
Sidecar JSON: env_id, task, level/grade, seed, source, obj_names, pred_names, ...
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


def write_episode(out_dir: Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any],
                  stem: Optional[str] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{meta.get('task', 'ep')}_{meta.get('level', meta.get('grade', 'x'))}_" \
                   f"s{meta.get('seed', 0)}_{uuid.uuid4().hex[:8]}"
    npz = out_dir / f"{stem}.npz"
    tmp = out_dir / f".{stem}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz)
    meta = dict(meta)
    meta["num_transitions"] = int(arrays["action"].shape[0]) if "action" in arrays else 0
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return npz


def read_episode(npz_path: Path) -> Dict[str, Any]:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    side = npz_path.with_suffix(".json")
    ep["meta"] = json.load(open(side, encoding="utf-8")) if side.exists() else {}
    return ep


def iter_episode_files(*dirs: Path, pattern: str = "*.npz") -> Iterator[Path]:
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.rglob(pattern)):
            if not p.name.startswith("."):
                yield p


def count_transitions(*dirs: Path) -> int:
    n = 0
    for p in iter_episode_files(*dirs, pattern="*.json"):
        try:
            n += int(json.load(open(p, encoding="utf-8")).get("num_transitions", 0))
        except Exception:
            pass
    return n


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_lines(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]
