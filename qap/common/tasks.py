"""Canonical IG task table, built from the repository's own config files.

A task is identified by its short name, e.g. ``StirSpoon``. From it:
    base env id (L0/L1/L2) : TwoRobotStirSpoon-v1
    L3 env id              : TwoRobotStirSpoonL3-v1
    dataset ids            : L0_TwoRobotStirSpoon-v1 ... L3_TwoRobotStirSpoon-v1
    human id               : human_H1
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from qap.common.paths import EXP_CONFIG_DIR, TASK_MAPPING_FILE

LEVELS = ("L0", "L1", "L2", "L3")
_SIM_RE = re.compile(r"^(L[0-3])_TwoRobot(?P<name>[A-Za-z]+)-v1$")


def base_env_id(task: str) -> str:
    return f"TwoRobot{task}-v1"


def l3_env_id(task: str) -> str:
    return f"TwoRobot{task}L3-v1"


def env_id_for(task: str, level: str) -> str:
    return l3_env_id(task) if level.upper() == "L3" else base_env_id(task)


def dataset_id(task: str, level: str) -> str:
    return f"{level.upper()}_TwoRobot{task}-v1"


def task_from_any(s: str) -> str:
    """Accept 'StirSpoon', 'TwoRobotStirSpoon-v1', 'TwoRobotStirSpoonL3-v1', 'L2_TwoRobotStirSpoon-v1'."""
    s = s.strip()
    m = _SIM_RE.match(s)
    if m:
        return m.group("name")
    s = re.sub(r"^L[0-3]_", "", s)
    s = re.sub(r"-v\d+$", "", s)
    s = re.sub(r"^TwoRobot", "", s)
    s = re.sub(r"L3$", "", s)
    return s


@dataclass
class TaskInfo:
    name: str
    human_id: str
    robot_ids: List[str] = field(default_factory=list)
    corpus: str = "unseen"          # C15 | C30 | C45 | unseen
    eval_role: str = "none"         # seen | unseen | none

    @property
    def base_env_id(self) -> str:
        return base_env_id(self.name)

    @property
    def l3_env_id(self) -> str:
        return l3_env_id(self.name)


def _corpus_members(tag: str) -> set:
    path = EXP_CONFIG_DIR / f"sim_train_config_{tag}.json"
    with open(path, encoding="utf-8") as f:
        return {task_from_any(x["repo_id"]) for x in json.load(f)}


def _eval_members(split: str) -> set:
    path = EXP_CONFIG_DIR / f"sim_test_config_{split}.json"
    with open(path, encoding="utf-8") as f:
        return {task_from_any(x["repo_id"]) for x in json.load(f)}


@lru_cache(maxsize=1)
def load_task_table() -> Dict[str, TaskInfo]:
    with open(TASK_MAPPING_FILE, encoding="utf-8") as f:
        mapping = json.load(f)["task_mappings"]
    c15, c30, c45 = _corpus_members("15"), _corpus_members("30"), _corpus_members("45")
    seen, unseen = _eval_members("seen"), _eval_members("unseen")
    table: Dict[str, TaskInfo] = {}
    for entry in mapping:
        sims = entry.get("sim_task_id") or []
        if not sims:
            continue
        name = task_from_any(sims[0])
        corpus = "C15" if name in c15 else "C30" if name in c30 else "C45" if name in c45 else "unseen"
        role = "seen" if name in seen else "unseen" if name in unseen else "none"
        table[name] = TaskInfo(name=name, human_id=entry["human_task_id"],
                               robot_ids=list(entry.get("robot_task_id", [])),
                               corpus=corpus, eval_role=role)
    return table


def all_tasks() -> List[str]:
    return sorted(load_task_table().keys())


def corpus_tasks(tag: str = "45") -> List[str]:
    """Tasks in the nested pre-training corpus C15 / C30 / C45."""
    keep = {"15": {"C15"}, "30": {"C15", "C30"}, "45": {"C15", "C30", "C45"}}[str(tag)]
    return sorted(n for n, t in load_task_table().items() if t.corpus in keep)


def human_id(task: str) -> Optional[str]:
    t = load_task_table().get(task)
    return t.human_id if t else None
