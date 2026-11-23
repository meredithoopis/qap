"""Filesystem layout shared by all QAP scripts.

Everything heavy (episodes, checkpoints, labels) goes under QAP_DATA_ROOT, which
defaults to <repo>/qap_runs. Small frozen deliverables (task list, splits) live in
qap/data and are meant to be committed.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QAP_PKG = REPO_ROOT / "qap"
FROZEN_DIR = QAP_PKG / "data"           # committed deliverables
IG_LEROBOT_DIR = REPO_ROOT / "examples" / "baselines" / "lerobot_dataset"
TASK_MAPPING_FILE = IG_LEROBOT_DIR / "task_mapping.json"
EXP_CONFIG_DIR = IG_LEROBOT_DIR / "config" / "exp_configs"
SIM_DESC_FILE = IG_LEROBOT_DIR / "task_desc" / "sim_desc.json"
TASKS_DIR = REPO_ROOT / "mani_skill" / "envs" / "tasks" / "tabletop"


def data_root() -> Path:
    root = Path(os.environ.get("QAP_DATA_ROOT", REPO_ROOT / "qap_runs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def sub(*parts: str) -> Path:
    p = data_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# Canonical sub-directories under QAP_DATA_ROOT
def episodes_dir(source: str) -> Path:
    """source in {'mp', 'play', 'graded'}"""
    return sub("episodes", source)


def runs_dir(name: str) -> Path:
    return sub("runs", name)


def labels_dir() -> Path:
    return sub("labels")


def inventory_dir() -> Path:
    return sub("inventory")
