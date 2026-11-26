"""Environment factory that reproduces the level switches of two_robot_run.py.

L0/L1/L2 share one registered class (``TwoRobot<Task>-v1``) and are selected with the
global flags in ``L0_L3_utils``; L3 is a separate class (``TwoRobot<Task>L3-v1``).
The flags are module globals, so set them *before* ``gym.make`` and keep one task per
process (the scripts in qap/collect spawn one process per job).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from qap.common.tasks import env_id_for


def set_level_flags(level: str, mirror_robot_pose: bool = True) -> None:
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils as U

    level = level.upper()
    if level not in ("L0", "L1", "L2", "L3"):
        raise ValueError(f"bad level {level}")
    U.set_l1_enabled(level == "L1")
    U.set_l2_enabled(level == "L2")
    # Standalone L3 envs are separate classes; the legacy base-env L3 flag stays off
    # (same as configure_dual_task_level("L3")).
    U.set_l3_enabled(False)
    U.set_lr_mirror_enabled(None)
    U.set_lr_mirror_robot_pose_enabled(mirror_robot_pose)


def make_env(task: str, level: str = "L0", *, obs_mode: str = "none",
             control_mode: str = "pd_joint_pos", sim_backend: str = "physx_cpu",
             render_mode: Optional[str] = None, shader: Optional[str] = None,
             mirror_robot_pose: bool = True, env_kwargs: Optional[Dict[str, Any]] = None,
             before_make: Optional[Callable[[], None]] = None):
    """Create one IG task environment at a given level.

    obs_mode='none' is enough for state collection (states are read from the simulator
    by qap.state.extractor). Use obs_mode='rgb' + shader='rt-fast' or 'default' when
    images are needed (DINOv2 visual-cosine baseline).
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers every task)

    set_level_flags(level, mirror_robot_pose)
    if before_make is not None:
        before_make()
    kwargs: Dict[str, Any] = dict(obs_mode=obs_mode, control_mode=control_mode,
                                  sim_backend=sim_backend)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if shader is not None:
        kwargs["sensor_configs"] = dict(shader_pack=shader)
        kwargs["human_render_camera_configs"] = dict(shader_pack=shader)
    # Optional: QAP_RENDER_BACKEND=none disables SAPIEN's renderer (Vulkan) so state-only
    # collection can run on CPU-only nodes. Only valid when no images are requested.
    rb = os.environ.get("QAP_RENDER_BACKEND")
    if rb and obs_mode == "none" and render_mode is None and shader is None:
        # pass the string "none": ManiSkill's parse_sim_and_render_backend slices the value
        # (render_backend[:4]) before its None check, so a Python None raises a TypeError
        kwargs["render_backend"] = "none" if rb.lower() == "none" else rb
    if env_kwargs:
        kwargs.update(env_kwargs)
    return gym.make(env_id_for(task, level), **kwargs)


def mp_solution(task: str, level: str) -> Callable:
    """Return the registered IG motion-planning solve() for (task, level)."""
    from mani_skill.examples.motionplanning.dual.two_robot_run import MP_SOLUTIONS

    key = env_id_for(task, level)
    if key not in MP_SOLUTIONS:
        raise KeyError(f"no motion-planning solution registered for {key}")
    return MP_SOLUTIONS[key]


def has_mp_solution(task: str, level: str) -> bool:
    try:
        mp_solution(task, level)
        return True
    except KeyError:
        return False
