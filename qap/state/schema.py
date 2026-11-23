"""Feature layout of the object-centric robot state s (paper Sec. 3, 'Robot reachability').

A state is a set of tokens:
    2 robot tokens  : proprioception of each arm (index 0 = agents[0], 1 = agents[1])
    N object tokens : pose, size, category id (separate int), discrete flags
    1 event token   : latched scene events (multi-hot), e.g. 'poured' is not visible in
                      object poses because liquid is not simulated, but it makes the state
                      irreversible and therefore must be in s.
"""

from __future__ import annotations

import re
from typing import Dict, List

# ---------------- robot token ----------------
ROBOT_FEATURES: List[str] = (
    [f"q{i}" for i in range(7)]          # arm joints
    + ["finger_l", "finger_r"]            # finger joints
    + ["tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz"]
)
ROBOT_DIM = len(ROBOT_FEATURES)  # 16

# ---------------- object token ----------------
OBJ_CONT: List[str] = ["x", "y", "z", "qw", "qx", "qy", "qz", "hx", "hy", "hz"]
OBJ_FLAGS: List[str] = [
    "is_static",       # static/kinematic body (fixture)
    "is_articulated",  # articulation (box lid, folder, juicer...)
    "grasped_0",       # agents[0].is_grasping(obj)
    "grasped_1",
    "contact_0",       # any contact between arm-0 fingers/hand and obj
    "contact_1",
    "lifted",          # z > rest z + LIFT_EPS
    "upright",         # object z-axis within ~25 deg of world z
    "open_frac",       # articulations: mean normalised joint position in [0,1]
    "is_open",         # open_frac > 0.5
    "filled",          # latched, from task predicates via flag_map (e.g. cup after pouring)
    "cut",             # latched
    "dispensed",       # latched (sauce/pills dispensed onto/into this object)
    "cleaned",         # latched
]
OBJ_FEATURES = OBJ_CONT + OBJ_FLAGS
OBJ_DIM = len(OBJ_FEATURES)  # 24
FLAG_INDEX: Dict[str, int] = {n: len(OBJ_CONT) + i for i, n in enumerate(OBJ_FLAGS)}

LIFT_EPS = 0.02
UPRIGHT_COS = 0.9

# ---------------- event token ----------------
EVENT_TYPES: List[str] = [
    "poured", "cut", "dispensed", "cleaned", "ground", "stirred", "scanned",
    "pressed", "opened", "closed", "placed", "success",
]
EVENT_INDEX = {e: i for i, e in enumerate(EVENT_TYPES)}

# Regexes that map raw evaluate() keys onto event types. Keys are lower-cased first.
EVENT_PATTERNS: Dict[str, str] = {
    "poured": r"pour|liquid_ever|is_bowl_src_ever_in_bowl_dst|is_spoon_ever_in_bowl_dst",
    "cut": r"cut",
    "dispensed": r"dispens|ketchup_ever",
    "cleaned": r"clean|wipe",
    "ground": r"grind",
    "stirred": r"stir",
    "scanned": r"scan",
    "pressed": r"press|stapler_ever|screwdriver_ever",
    "opened": r"open|is_folder_open|is_cap_opened",
    "closed": r"close_enough|_closed$|^closed",   # not 'is_fork_close_to_bowl_*'
    "placed": r"placed|in_box|in_basket|on_scale|on_stand|in_plate|in_region|on_rack|cleared|in_the_cup|contact_rack",
    "success": r"^success$",
}
# Physically irreversible events are latched for the whole episode even if the raw
# predicate later turns false (e.g. 'is_tilted' while pouring).
LATCHED_EVENTS = {"poured", "cut", "dispensed", "cleaned", "ground", "stirred", "scanned", "pressed"}
EVENT_RE = {e: re.compile(p) for e, p in EVENT_PATTERNS.items()}

# Some irreversible effects are only visible through the task's success predicate (e.g.
# pills tipped out of the bottle at PlacePillBox L3). Success of these scenes is latched as
# the given event. Key 'Task@Level' takes precedence over 'Task'.
TASK_SUCCESS_EVENT: Dict[str, str] = {
    "PlacePillBox@L3": "dispensed",
    "PourKetchupFries": "dispensed",
    "TransFood": "poured",
    "PourKettle": "poured", "PourCup": "poured", "PourLiquidCup": "poured",
    "PourLiquidMug": "poured", "PourLiquidFilter": "poured",
    "CutFruit": "cut",
    "WipePot": "cleaned", "CleanDesk": "cleaned", "CleanCup": "cleaned",
    "GrindFood": "ground",
}


def success_event(task: str | None, level: str | None) -> str | None:
    if not task:
        return None
    return TASK_SUCCESS_EVENT.get(f"{task}@{(level or '').upper()}") or TASK_SUCCESS_EVENT.get(task)

# Raw predicate keys we never treat as events (they are distances / bookkeeping).
IGNORED_PRED_RE = re.compile(r"dist|pos$|_pos_|steps|progress|height|qpos|cos$|center|goal_pose|xy$|_z$|motion")


def events_for_key(key: str) -> List[str]:
    k = key.lower()
    if IGNORED_PRED_RE.search(k) or k.startswith("is_grasp") or "grasped" in k or "static" in k:
        return ["success"] if k == "success" else []
    return [e for e, r in EVENT_RE.items() if r.search(k)]
