"""Which (task, level) scenes must stay out of d_theta / policy training data.

Held-out substitutes (proposal Sec. 6.1 and Sec. 8): the learned quasimetric may only
know connectivity it has seen. To keep the calibration and test splits honest, the L3
scenes (and IG-Sub graded scenes G2-G4) of calibration/test tasks are excluded from
training data by default. Their L0-L2 scenes stay in (they are part of the IG 45-task
corpus that the paper trains d_theta on).
"""

from __future__ import annotations

from typing import Dict, Iterable, Set

from qap.common.io import load_json
from qap.common.paths import FROZEN_DIR


def heldout_tasks(splits=("calibration", "test")) -> Set[str]:
    path = FROZEN_DIR / "splits.json"
    if not path.exists():
        return set()
    sp = load_json(path)["splits"]
    return {t for s in splits for t in sp.get(s, [])}


def is_heldout_scene(task: str, level_or_grade: str, held: Iterable[str]) -> bool:
    lg = level_or_grade.upper()
    return task in set(held) and lg in {"L3", "G2", "G3", "G4"}


def keep_episode(meta: Dict, held: Set[str]) -> bool:
    lg = str(meta.get("grade") or meta.get("level") or "")
    return not is_heldout_scene(str(meta.get("task")), lg, held)
