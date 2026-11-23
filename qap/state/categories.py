"""Object categories (cat(a), kappa_j) and the integer vocabulary used by the encoder.

IG actor names are not uniform: some are asset ids ('076_breadbasket-0'), some are role
names ('apple_out', 'fruit_a', 'food', 'cube'). We normalise automatically and let a
reviewed override file fix the rest:

    qap/data/category_overrides.json
        {"global": {"apple_out": "apple", ...},
         "per_task": {"TransFood": {"food": "bowl"}, ...}}

`python -m qap.week1.runtime_inventory` writes a draft of this file (normalised guesses
for every actor of every task scene) for you to review once in Week 1.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from qap.common.paths import FROZEN_DIR

UNK = "<unk>"
VOCAB_FILE = FROZEN_DIR / "category_vocab.json"
OVERRIDES_FILE = FROZEN_DIR / "category_overrides.json"

_SUFFIXES = re.compile(r"(-\d+)+$|_(in|out|left|right|src|dst|a|b|\d)$")
_PREFIX_ID = re.compile(r"^(ycb:)?\d{3}_")


def normalize_name(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"^ycb:", "", n)
    n = _SUFFIXES.sub("", n)
    n = _SUFFIXES.sub("", n)          # e.g. 'box1-0' -> 'box1' ; 'apple_out' -> 'apple'
    n = _PREFIX_ID.sub("", n)         # '076_breadbasket' -> 'breadbasket'
    n = re.sub(r"\d+$", "", n)        # 'box1' -> 'box'
    n = n.strip("_- ")
    if re.fullmatch(r"\d*", n or ""):
        return UNK                     # pure numeric (e.g. bare partnet id) -> needs override
    return n


class CategoryResolver:
    def __init__(self, overrides_file: Path = OVERRIDES_FILE):
        self.global_map: Dict[str, str] = {}
        self.per_task: Dict[str, Dict[str, str]] = {}
        if Path(overrides_file).exists():
            data = json.load(open(overrides_file, encoding="utf-8"))
            self.global_map = data.get("global", {})
            self.per_task = data.get("per_task", {})

    def __call__(self, actor_name: str, task: Optional[str] = None) -> str:
        if task and actor_name in self.per_task.get(task, {}):
            return self.per_task[task][actor_name]
        if actor_name in self.global_map:
            return self.global_map[actor_name]
        n = normalize_name(actor_name)
        return self.global_map.get(n, n)


class CategoryVocab:
    """Stable string -> int mapping. Id 0 is <unk>. Grows only when allow_new=True."""

    def __init__(self, path: Path = VOCAB_FILE):
        self.path = Path(path)
        self.stoi: Dict[str, int] = {UNK: 0}
        if self.path.exists():
            self.stoi = json.load(open(self.path, encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.stoi)

    def id(self, cat: str, allow_new: bool = False) -> int:
        if cat not in self.stoi:
            if not allow_new:
                return 0
            self.stoi[cat] = len(self.stoi)
        return self.stoi[cat]

    def itos(self) -> Dict[int, str]:
        return {v: k for k, v in self.stoi.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.stoi, open(self.path, "w", encoding="utf-8"), indent=1, sort_keys=True)
