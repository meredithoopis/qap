"""Instantiate IG-Sub graded scenes (G0-G4) from the released task envs.

    env, solve, spec = make_graded_env("PourKetchupFries", "G4")
    solve(env, seed=0)          # the IG solver of spec['solver'] level; G4 is expected to fail

Mechanisms (no task file is edited):
  * asset remap (G2): wraps the loader functions the task module calls
    (``create_actor`` in the task module namespace, ``actors.get_actor_builder``) so that
    the demonstrated asset is replaced before ``_load_scene`` runs. One task per process.
  * hide (G4): after every reset the named actors/articulations are parked far outside
    the workspace (they stay in the scene, so the object-token set is unchanged, but no
    arm can reach them).
"""

from __future__ import annotations

import inspect
import sys
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from qap.common.envs import make_env, mp_solution
from qap.common.io import load_json
from qap.common.paths import inventory_dir
from qap.common.tasks import env_id_for
from qap.scenes.graded_specs import grade_spec

PARK_XYZ = np.array([2.5, 0.0, 0.05])


def _task_module(task: str, level: str):
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.registration import REGISTERED_ENVS

    cls = REGISTERED_ENVS[env_id_for(task, level)].cls
    return sys.modules[cls.__module__]


def install_asset_remap(task: str, level: str, remap: Dict[str, object]) -> List[str]:
    """Patch loaders for this process. Returns a log of installed patches."""
    from mani_skill.utils.building import actors as actors_mod

    log: List[str] = []
    ycb = {k[4:]: str(v)[4:] for k, v in remap.items() if k.startswith("ycb:")}
    if ycb and not getattr(actors_mod.get_actor_builder, "_qap_patched", False):
        orig = actors_mod.get_actor_builder

        def get_actor_builder(scene, id, *a, **kw):  # noqa: A002 (mirror upstream signature)
            if isinstance(id, str) and id.startswith("ycb:") and id[4:] in ycb:
                id = "ycb:" + ycb[id[4:]]
            return orig(scene, id, *a, **kw)

        get_actor_builder._qap_patched = True
        actors_mod.get_actor_builder = get_actor_builder
        log.append(f"ycb remap {ycb}")

    rw = {k[len("robotwin:"):]: v for k, v in remap.items() if k.startswith("robotwin:")}
    if rw:
        mod = _task_module(task, level)
        if not hasattr(mod, "create_actor"):
            raise RuntimeError(f"{mod.__name__} has no create_actor to patch")
        orig_ca = mod.create_actor
        sig = inspect.signature(orig_ca)

        def create_actor(*a, **kw):
            bound = sig.bind_partial(*a, **kw)
            name = bound.arguments.get("modelname")
            if name in rw:
                r = rw[name]
                bound.arguments["modelname"] = r["modelname"]
                for key in ("model_id", "scale", "replace_scale"):
                    if key in r:
                        bound.arguments[key] = r[key]
            return orig_ca(*bound.args, **bound.kwargs)

        mod.create_actor = create_actor
        log.append(f"robotwin remap {rw} in {mod.__name__}")
    return log


def resolve_hide(task: str, hide) -> List[str]:
    if isinstance(hide, (list, tuple)):
        return list(hide)
    if hide != "@l3_only":
        return [str(hide)]
    path = inventory_dir() / "runtime_inventory.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run qap.week1.runtime_inventory first (needed to resolve @l3_only)")
    inv = load_json(path)[task]
    n0 = set(inv["L0"]["obj_names"])
    names = [n for n in inv["L3"]["obj_names"] if n not in n0]
    if not names:   # names coincide across levels: fall back to category difference
        c0 = set(inv["L0"]["obj_categories"])
        names = [n for n, c in zip(inv["L3"]["obj_names"], inv["L3"]["obj_categories"]) if c not in c0]
    if not names:
        raise RuntimeError(f"cannot resolve the L3 substitute of {task}; set G4 'hide' explicitly")
    return names


class HideActors(gym.Wrapper):
    """Park the named objects outside the workspace after every reset (G4)."""

    def __init__(self, env, names: List[str]):
        super().__init__(env)
        self.names = list(names)

    def _park(self) -> None:
        import sapien
        from mani_skill.utils.structs.pose import Pose

        scene = self.env.unwrapped.scene
        for k, n in enumerate(self.names):
            p = PARK_XYZ + np.array([0.0, 0.35 * k, 0.0])
            sp = sapien.Pose(p=p.tolist())
            if n.startswith("attr:"):     # the task attribute holding the actor (see graded_specs.G4_HIDE)
                obj = getattr(self.env.unwrapped, n[5:], None)
                if obj is None:
                    raise KeyError(f"G4 hide: env has no attribute {n[5:]}")
                if hasattr(obj, "set_root_pose"):
                    obj.set_root_pose(sp)
                else:
                    obj.set_pose(Pose.create(sp))
            elif n in scene.actors:
                scene.actors[n].set_pose(Pose.create(sp))
            elif n in scene.articulations:
                scene.articulations[n].set_root_pose(sp)
            else:
                raise KeyError(f"G4 hide: no actor/articulation named {n}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._park()
        return obs, info


def make_graded_env(task: str, grade: str, **make_kwargs) -> Tuple[gym.Env, object, Dict]:
    spec = grade_spec(task, grade)
    if spec is None:
        raise ValueError(f"{task} has no {grade} in IG-Sub (see qap/scenes/graded_specs.py)")
    level = str(spec["level"])
    patches: List[str] = []

    def before():
        if spec.get("remap"):
            patches.extend(install_asset_remap(task, level, spec["remap"]))

    env = make_env(task, level, before_make=before, **make_kwargs)
    if spec.get("hide"):
        env = HideActors(env, resolve_hide(task, spec["hide"]))
    spec = dict(spec, grade=grade.upper(), task=task, patches=patches)
    return env, mp_solution(task, str(spec["solver"])), spec
