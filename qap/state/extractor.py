"""Object-centric state extractor from the ManiSkill simulator (Week 1 deliverable).

    ex = ObjectCentricExtractor(env, task="PourKettle")
    ex.reset()                      # after env.reset(); rediscovers objects
    s = ex.extract(info)            # dict of numpy arrays for one time step

Produces, per step:
    robot   [2, ROBOT_DIM]   dual-arm proprioception (joint qpos, finger qpos, TCP pose)
    obj     [N, OBJ_DIM]     per-object tokens: pose, half extents, discrete flags
    events  [E]              latched scene events derived from env.evaluate() predicates
    preds   [P]              raw boolean predicates (task predicates for L_pred)
    contact [2, N]           arm-object contact (fingers or hand), from one pass over contacts
and contact *onsets* (no contact at t-1, contact at t) which define the entry states W_a.

Only CPU physics is supported (the IG motion planner runs on CPU anyway).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qap.common.paths import FROZEN_DIR
from qap.state.categories import CategoryResolver
from qap.state.schema import (
    EVENT_INDEX,
    EVENT_TYPES,
    FLAG_INDEX,
    LATCHED_EVENTS,
    LIFT_EPS,
    OBJ_DIM,
    ROBOT_DIM,
    UPRIGHT_COS,
    events_for_key,
    success_event,
)

EXCLUDE_NAME_RE = re.compile(r"^(table|ground|wall|goal|grasp_pose_visual|camera|light)", re.I)
FLAG_MAP_FILE = FROZEN_DIR / "flag_map.json"


def _np(x) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _scalar_bool(v) -> Optional[bool]:
    a = _np(v)
    if a.dtype == np.bool_ and a.size == 1:
        return bool(a.reshape(-1)[0])
    return None


def _entity_key(ent) -> Any:
    k = getattr(ent, "per_scene_id", None)
    return k if k is not None else getattr(ent, "name", id(ent))


def _canon_quat(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-9)
    return -q if q[0] < 0 else q


def _quat_z_axis(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


@dataclass
class ObjectSlot:
    name: str
    category: str
    kind: str                      # 'actor' | 'articulation'
    handle: Any                    # mani_skill Actor or Articulation
    entities: List[Any]            # sapien entities belonging to the object
    half_extents: np.ndarray
    is_static: bool
    rest_z: float = 0.0


class ObjectCentricExtractor:
    def __init__(self, env, task: Optional[str] = None, resolver: Optional[CategoryResolver] = None,
                 flag_map: Optional[Dict[str, Dict[str, str]]] = None, max_objects: int = 16,
                 contact_impulse_eps: float = 1e-6, level: Optional[str] = None):
        self.env = env
        self.contact_impulse_eps = contact_impulse_eps
        self.base = env.unwrapped
        self.task = task
        self.success_event = success_event(task, level)
        self.resolver = resolver or CategoryResolver()
        if flag_map is None and FLAG_MAP_FILE.exists():
            flag_map = json.load(open(FLAG_MAP_FILE, encoding="utf-8")).get(task or "", {})
        # flag_map: {predicate_key: {"object": actor_name, "flag": "filled"|"cut"|...}}
        self.flag_map: Dict[str, Dict[str, str]] = flag_map or {}
        self.max_objects = max_objects
        self.slots: List[ObjectSlot] = []
        self.pred_names: List[str] = []
        self._prev_contact: Optional[np.ndarray] = None
        self._latched_events = np.zeros(len(EVENT_TYPES), dtype=np.uint8)
        self._latched_flags: Dict[Tuple[int, str], bool] = {}

    # ------------------------------------------------------------------ discovery
    @property
    def agents(self):
        return list(self.base.agent.agents)

    def _robot_names(self) -> set:
        return {a.robot.name for a in self.agents}

    def _static_names(self) -> set:
        names = set()
        ts = getattr(self.base, "table_scene", None)
        for attr in ("table", "ground", "wall"):
            obj = getattr(ts, attr, None) if ts is not None else None
            if obj is not None and hasattr(obj, "name"):
                names.add(obj.name)
        return names

    @staticmethod
    def _half_extents(handle) -> np.ndarray:
        try:
            mesh = handle.get_first_collision_mesh(to_world_frame=False)
            if mesh is not None:
                b = np.asarray(mesh.bounding_box.bounds)
                return ((b[1] - b[0]) / 2.0).astype(np.float32)
        except Exception:
            pass
        return np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        """Call after every env.reset(): scenes may be reconfigured (new actors)."""
        scene = self.base.scene
        skip = self._robot_names() | self._static_names()
        slots: List[ObjectSlot] = []
        for name, actor in scene.actors.items():
            if name in skip or EXCLUDE_NAME_RE.match(name):
                continue
            try:
                if len(actor._bodies) == 0 or len(actor._bodies[0].get_collision_shapes()) == 0:
                    continue          # visual-only marker
            except Exception:
                pass
            body_type = getattr(actor, "px_body_type", "dynamic")
            slots.append(ObjectSlot(
                name=name, category=self.resolver(name, self.task), kind="actor", handle=actor,
                entities=[actor._bodies[0].entity], half_extents=self._half_extents(actor),
                is_static=body_type != "dynamic"))
        for name, art in scene.articulations.items():
            if name in skip:
                continue
            ents = []
            for link in art.links:
                try:
                    ents.append(link._bodies[0].entity)
                except Exception:
                    pass
            slots.append(ObjectSlot(
                name=name, category=self.resolver(name, self.task), kind="articulation", handle=art,
                entities=ents, half_extents=self._half_extents(art), is_static=True))
        slots.sort(key=lambda s: (s.kind, s.name))
        if len(slots) > self.max_objects:
            slots = slots[: self.max_objects]
        self.slots = slots
        self._entity_to_obj = {}
        for i, s in enumerate(self.slots):
            for e in s.entities:
                self._entity_to_obj[_entity_key(e)] = i
        self._entity_to_arm = {}
        for arm, agent in enumerate(self.agents):
            links = [agent.finger1_link, agent.finger2_link]
            hand = agent.robot.links_map.get("panda_hand") if hasattr(agent.robot, "links_map") else None
            if hand is not None:
                links.append(hand)
            for l in links:
                self._entity_to_arm[_entity_key(l._bodies[0].entity)] = arm
        for s in self.slots:
            s.rest_z = float(self._pose(s)[0][2])
        self._prev_contact = None
        self._latched_events[:] = 0
        self._latched_flags.clear()
        self.pred_names = []

    # ------------------------------------------------------------------ features
    def _pose(self, s: ObjectSlot) -> Tuple[np.ndarray, np.ndarray]:
        pose = s.handle.root_pose if s.kind == "articulation" else s.handle.pose
        return _np(pose.p).reshape(-1, 3)[0], _np(pose.q).reshape(-1, 4)[0]

    def robot_tokens(self) -> np.ndarray:
        out = np.zeros((2, ROBOT_DIM), dtype=np.float32)
        for i, agent in enumerate(self.agents[:2]):
            q = _np(agent.robot.get_qpos()).reshape(-1)
            out[i, : min(9, q.size)] = q[:9]
            tcp = agent.tcp.pose
            out[i, 9:12] = _np(tcp.p).reshape(-1)[:3]
            out[i, 12:16] = _canon_quat(_np(tcp.q).reshape(-1)[:4])
        return out

    def contacts(self) -> np.ndarray:
        """[2, N] bool: arm fingers/hand touching object. One pass over the contact list."""
        n = len(self.slots)
        c = np.zeros((2, n), dtype=bool)
        px = self.base.scene.px
        try:
            contact_list = px.get_contacts()
        except Exception:
            return self._contacts_pairwise()
        for ct in contact_list:
            k0, k1 = _entity_key(ct.bodies[0].entity), _entity_key(ct.bodies[1].entity)
            arm, obj = self._entity_to_arm.get(k0), self._entity_to_obj.get(k1)
            if arm is None or obj is None:
                arm, obj = self._entity_to_arm.get(k1), self._entity_to_obj.get(k0)
            if arm is None or obj is None:
                continue
            imp = np.sum([p.impulse for p in ct.points], axis=0) if len(ct.points) else np.zeros(3)
            # PhysX also reports zero-impulse pairs inside the contact offset; ignore those.
            if np.linalg.norm(imp) > self.contact_impulse_eps:
                c[arm, obj] = True
        return c

    def _contacts_pairwise(self) -> np.ndarray:
        scene = self.base.scene
        c = np.zeros((2, len(self.slots)), dtype=bool)
        for arm, agent in enumerate(self.agents[:2]):
            for j, s in enumerate(self.slots):
                if s.kind != "actor":
                    continue
                for link in (agent.finger1_link, agent.finger2_link):
                    f = _np(scene.get_pairwise_contact_forces(link, s.handle)).reshape(-1, 3)[0]
                    if np.linalg.norm(f) > 1e-3:
                        c[arm, j] = True
        return c

    def _grasped(self) -> np.ndarray:
        g = np.zeros((2, len(self.slots)), dtype=bool)
        for arm, agent in enumerate(self.agents[:2]):
            for j, s in enumerate(self.slots):
                if s.kind != "actor" or s.is_static:
                    continue
                try:
                    g[arm, j] = bool(_np(agent.is_grasping(s.handle)).reshape(-1)[0])
                except Exception:
                    pass
        return g

    @staticmethod
    def _open_frac(art) -> float:
        try:
            q = _np(art.get_qpos()).reshape(-1)
            lim = _np(art.get_qlimits()).reshape(-1, 2)
            if q.size == 0:
                return 0.0
            lo, hi = lim[:, 0], lim[:, 1]
            span = np.where(np.isfinite(hi - lo) & ((hi - lo) > 1e-6), hi - lo, 1.0)
            frac = np.clip((q - np.where(np.isfinite(lo), lo, 0.0)) / span, 0.0, 1.0)
            return float(frac.mean())
        except Exception:
            return 0.0

    def predicates(self, info: Optional[Dict[str, Any]]) -> Dict[str, bool]:
        if info is None:
            try:
                info = self.base.evaluate()
            except Exception:
                info = {}
        out = {}
        for k, v in info.items():
            b = _scalar_bool(v)
            if b is not None:
                out[k] = b
        return out

    def extract(self, info: Optional[Dict[str, Any]] = None, contact: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        n = len(self.slots)
        obj = np.zeros((n, OBJ_DIM), dtype=np.float32)
        contact = self.contacts() if contact is None else contact
        grasped = self._grasped()
        for j, s in enumerate(self.slots):
            p, q = self._pose(s)
            q = _canon_quat(q)
            obj[j, 0:3], obj[j, 3:7], obj[j, 7:10] = p, q, s.half_extents
            f = obj[j]
            f[FLAG_INDEX["is_static"]] = float(s.is_static)
            f[FLAG_INDEX["is_articulated"]] = float(s.kind == "articulation")
            f[FLAG_INDEX["grasped_0"]], f[FLAG_INDEX["grasped_1"]] = grasped[0, j], grasped[1, j]
            f[FLAG_INDEX["contact_0"]], f[FLAG_INDEX["contact_1"]] = contact[0, j], contact[1, j]
            f[FLAG_INDEX["lifted"]] = float(p[2] > s.rest_z + LIFT_EPS)
            f[FLAG_INDEX["upright"]] = float(_quat_z_axis(q)[2] > UPRIGHT_COS)
            if s.kind == "articulation":
                of = self._open_frac(s.handle)
                f[FLAG_INDEX["open_frac"]], f[FLAG_INDEX["is_open"]] = of, float(of > 0.5)

        preds = self.predicates(info)
        if not self.pred_names:
            self.pred_names = sorted(preds.keys())
        pred_vec = np.array([preds.get(k, False) for k in self.pred_names], dtype=np.uint8)

        ev = np.zeros(len(EVENT_TYPES), dtype=np.uint8)
        if self.success_event and preds.get("success", False):
            self._latched_events[EVENT_INDEX[self.success_event]] = 1
        for k, b in preds.items():
            if not b:
                continue
            for e in events_for_key(k):
                ev[EVENT_INDEX[e]] = 1
                if e in LATCHED_EVENTS:
                    self._latched_events[EVENT_INDEX[e]] = 1
            fm = self.flag_map.get(k)
            if fm:
                idx = next((i for i, s in enumerate(self.slots) if s.name == fm["object"]), None)
                if idx is not None:
                    self._latched_flags[(idx, fm["flag"])] = True
        ev = np.maximum(ev, self._latched_events)
        for (idx, flag), on in self._latched_flags.items():
            if on and flag in FLAG_INDEX:
                obj[idx, FLAG_INDEX[flag]] = 1.0

        return dict(robot=self.robot_tokens(), obj=obj, events=ev, preds=pred_vec,
                    contact=contact.astype(np.uint8),
                    success=np.uint8(bool(preds.get("success", False))))

    def onsets(self, contact: np.ndarray) -> List[Tuple[int, int]]:
        """(arm, obj) pairs whose contact starts now. Updates internal history."""
        prev = self._prev_contact if self._prev_contact is not None else np.zeros_like(contact)
        new = np.argwhere(contact & ~prev)
        self._prev_contact = contact.copy()
        return [(int(a), int(o)) for a, o in new]

    def describe(self) -> Dict[str, Any]:
        return dict(obj_names=[s.name for s in self.slots],
                    obj_categories=[s.category for s in self.slots],
                    obj_kinds=[s.kind for s in self.slots],
                    pred_names=list(self.pred_names),
                    event_types=list(EVENT_TYPES))
