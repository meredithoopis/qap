"""Play primitives on top of the IG dual-arm motion planner (PandaArmMotionPlanningSolver).

Every primitive is a short waypoint program executed with screw-motion planning; it
returns True if every planning call succeeded. Failures leave the scene in whatever
state it reached, which is fine for play (all transitions are valid robot data).
The other arm always receives `other_gripper_state` (IG convention, avoids collisions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import CLOSED, OPEN, PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

FINGER_LENGTH = 0.025
REACH_RADIUS = 0.85


def _np(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


@dataclass
class Target:
    name: str
    kind: str          # 'actor' | 'articulation'
    handle: object
    dynamic: bool


class PlayArms:
    """Two planners sharing one scene. Create after env.reset()."""

    def __init__(self, env, rng: np.random.Generator):
        self.env = env
        self.base = env.unwrapped
        self.rng = rng
        self.planners: List[PandaArmMotionPlanningSolver] = [
            PandaArmMotionPlanningSolver(env, debug=False, vis=False,
                                         base_pose=self.base.agent.agents[i].robot.pose,
                                         visualize_target_grasp_pose=False, print_env_info=False,
                                         multi_robot_id=i)
            for i in (0, 1)
        ]
        self.home = [self.agent(i).tcp.pose.sp for i in (0, 1)]
        self.holding: List[Optional[Target]] = [None, None]

    # ------------------------------------------------------------- helpers
    def agent(self, i):
        return self.base.agent.agents[i]

    def base_xy(self, i) -> np.ndarray:
        return _np(self.agent(i).robot.pose.p).reshape(-1)[:2]

    def other_state(self, i):
        return self.planners[1 - i].gripper_state

    def reachable(self, i, p) -> bool:
        return float(np.linalg.norm(np.asarray(p)[:2] - self.base_xy(i))) < REACH_RADIUS

    def nearest_arm(self, p) -> int:
        d = [np.linalg.norm(np.asarray(p)[:2] - self.base_xy(i)) for i in (0, 1)]
        return int(np.argmin(d))

    def move(self, i, pose: sapien.Pose) -> bool:
        res = self.planners[i].move_to_pose_with_screw(pose, other_gripper_state=self.other_state(i))
        return not (isinstance(res, int) and res == -1)

    def close(self, i):
        self.planners[i].close_gripper(other_gripper_state=self.other_state(i))

    def open(self, i):
        self.planners[i].open_gripper(other_gripper_state=self.other_state(i))
        self.holding[i] = None

    def topdown(self, i, p, yaw: float) -> sapien.Pose:
        approaching = np.array([0.0, 0.0, -1.0])
        closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return self.agent(i).build_grasp_pose(approaching, closing, np.asarray(p, dtype=np.float64))

    def tcp(self, i) -> sapien.Pose:
        return self.agent(i).tcp.pose.sp

    @staticmethod
    def pos(t: Target) -> np.ndarray:
        pose = t.handle.root_pose if t.kind == "articulation" else t.handle.pose
        return _np(pose.p).reshape(-1)[:3]

    # ------------------------------------------------------------- primitives
    def reach_random(self, i, box: Sequence[np.ndarray]) -> bool:
        lo, hi = box
        for _ in range(5):
            p = self.rng.uniform(lo, hi)
            if self.reachable(i, p):
                return self.move(i, self.topdown(i, p, self.rng.uniform(-np.pi, np.pi)))
        return False

    def go_home(self, i) -> bool:
        return self.move(i, self.home[i])

    def pick(self, i, t: Target, lift: float = 0.12) -> bool:
        if not t.dynamic or t.kind != "actor":
            return False
        try:
            obb = get_actor_obb(t.handle)
        except Exception:
            return False
        closing_hint = _np(self.agent(i).tcp.pose.to_transformation_matrix())[0, :3, 1]
        info = compute_grasp_info_by_obb(obb, approaching=np.array([0, 0, -1.0]),
                                         target_closing=closing_hint, depth=FINGER_LENGTH)
        grasp = self.agent(i).build_grasp_pose(np.array([0, 0, -1.0]), info["closing"], info["center"])
        if not self.reachable(i, grasp.p):
            return False
        if not self.move(i, grasp * sapien.Pose([0, 0, -0.08])):
            return False
        if not self.move(i, grasp):
            return False
        self.close(i)
        self.holding[i] = t
        return self.move(i, sapien.Pose(grasp.p + np.array([0, 0, lift]), grasp.q))

    def place(self, i, xyz: np.ndarray, release: bool = True) -> bool:
        cur = self.tcp(i)
        above = sapien.Pose(np.array([xyz[0], xyz[1], max(xyz[2] + 0.15, cur.p[2])]), cur.q)
        if not self.reachable(i, xyz) or not self.move(i, above):
            return False
        ok = self.move(i, sapien.Pose(np.asarray(xyz, dtype=np.float64), cur.q))
        if release:
            self.open(i)
            self.move(i, above)
        return ok

    def tilt(self, i, angle: float) -> bool:
        """Pour-like: rotate the held object about the TCP x axis, hold, rotate back."""
        cur = self.tcp(i)
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        rot = sapien.Pose(q=[c, s, 0, 0])
        ok = self.move(i, cur * rot)
        ok = self.move(i, cur) and ok
        return ok

    def push(self, i, t: Target, dist: float = 0.08) -> bool:
        p = self.pos(t)
        d = self.rng.normal(size=2)
        d /= np.linalg.norm(d) + 1e-9
        start = np.array([p[0] - d[0] * 0.08, p[1] - d[1] * 0.08, p[2] + 0.02])
        end = start + np.array([d[0], d[1], 0.0]) * (0.08 + dist)
        if not (self.reachable(i, start) and self.reachable(i, end)):
            return False
        yaw = np.arctan2(d[1], d[0]) + np.pi / 2
        self.close(i)
        ok = self.move(i, self.topdown(i, start + [0, 0, 0.1], yaw)) and self.move(i, self.topdown(i, start, yaw))
        ok = ok and self.move(i, self.topdown(i, end, yaw))
        self.move(i, self.topdown(i, end + [0, 0, 0.1], yaw))
        self.open(i)
        return ok

    def press(self, i, t: Target, depth: float = 0.02) -> bool:
        p = self.pos(t)
        top = p[2] + 0.05
        try:
            top = float(t.handle.get_first_collision_mesh(to_world_frame=True).bounds[1, 2])
        except Exception:
            pass
        yaw = self.rng.uniform(-np.pi, np.pi)
        self.close(i)
        ok = self.move(i, self.topdown(i, [p[0], p[1], top + 0.08], yaw))
        ok = ok and self.move(i, self.topdown(i, [p[0], p[1], top - depth], yaw))
        self.move(i, self.topdown(i, [p[0], p[1], top + 0.08], yaw))
        self.open(i)
        return ok

    def articulate(self, i, t: Target) -> bool:
        """Grab a random moving link of an articulation and drag it a few cm."""
        if t.kind != "articulation":
            return False
        links = [l for l in t.handle.links[1:]] or list(t.handle.links)
        link = links[self.rng.integers(len(links))]
        p = _np(link.pose.p).reshape(-1)[:3]
        if not self.reachable(i, p):
            return False
        yaw = self.rng.uniform(-np.pi, np.pi)
        ok = self.move(i, self.topdown(i, p + [0, 0, 0.08], yaw)) and self.move(i, self.topdown(i, p, yaw))
        if not ok:
            return False
        self.close(i)
        delta = self.rng.normal(size=3) * np.array([0.05, 0.05, 0.06])
        delta[2] = abs(delta[2])
        ok = self.move(i, self.topdown(i, p + delta, yaw))
        self.open(i)
        self.move(i, self.topdown(i, p + delta + [0, 0, 0.08], yaw))
        return ok

    @property
    def gripper_closed(self) -> List[bool]:
        return [pl.gripper_state == CLOSED for pl in self.planners]


__all__ = ["PlayArms", "Target", "OPEN", "CLOSED"]
