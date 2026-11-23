"""gym.Wrapper that records object-centric states for every env.step().

The IG motion planners call ``env.step`` on whatever env they are given, so wrapping the
environment once captures planner episodes, play episodes and policy rollouts alike:

    env = StateRecorder(make_env(task, level), task=task)
    solve(env, seed=seed)                     # planner steps through the wrapper
    env.flush(meta=dict(source="mp", seed=seed), out_dir=..., save=success)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from qap.common.io import write_episode
from qap.state.categories import CategoryVocab
from qap.state.extractor import ObjectCentricExtractor, _np


def flatten_action(action) -> np.ndarray:
    if isinstance(action, dict):
        keys = sorted(action.keys(), key=lambda k: int(str(k).rsplit("-", 1)[-1]) if str(k)[-1].isdigit() else 0)
        return np.concatenate([_np(action[k]).reshape(-1) for k in keys]).astype(np.float32)
    return _np(action).reshape(-1).astype(np.float32)


class StateRecorder(gym.Wrapper):
    def __init__(self, env, task: Optional[str] = None, sim_state_every: int = 5,
                 extractor: Optional[ObjectCentricExtractor] = None, vocab: Optional[CategoryVocab] = None,
                 max_steps: Optional[int] = None, level: Optional[str] = None):
        super().__init__(env)
        self.task = task
        self.ex = extractor or ObjectCentricExtractor(env, task=task, level=level)
        self.vocab = vocab or CategoryVocab()
        self.sim_state_every = max(1, int(sim_state_every))
        self.max_steps = max_steps
        self._buf: Optional[Dict[str, List[Any]]] = None
        self.last_info: Dict[str, Any] = {}

    # --------------------------------------------------------------- buffer
    def _new_buffer(self) -> None:
        self._buf = dict(robot=[], obj=[], events=[], preds=[], contact=[], success=[],
                         action=[], onsets=[], sim_t=[], sim_state=[])

    def _push_state(self, info: Optional[Dict[str, Any]]) -> None:
        b = self._buf
        t = len(b["robot"])
        contact = self.ex.contacts()
        s = self.ex.extract(info, contact=contact.astype(bool))
        for k in ("robot", "obj", "events", "preds", "contact", "success"):
            b[k].append(s[k])
        on = self.ex.onsets(contact.astype(bool))
        for arm, obj in on:
            b["onsets"].append((t, arm, obj))
        if t % self.sim_state_every == 0 or on:
            try:
                b["sim_t"].append(t)
                b["sim_state"].append(_np(self.env.unwrapped.get_state()).reshape(-1).astype(np.float32))
            except Exception:
                b["sim_t"].pop()

    @property
    def num_steps(self) -> int:
        return 0 if self._buf is None else len(self._buf["action"])

    # --------------------------------------------------------------- gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ex.reset()
        self._new_buffer()
        self._push_state(None)
        self.last_info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._buf is not None and (self.max_steps is None or self.num_steps < self.max_steps):
            self._buf["action"].append(flatten_action(action))
            self._push_state(info)
        self.last_info = info
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- output
    def episode_arrays(self) -> Dict[str, np.ndarray]:
        b = self._buf
        if b is None or not b["robot"]:
            return {}
        n_obj = b["obj"][0].shape[0]
        sim = b["sim_state"]
        same = sim and all(x.shape == sim[0].shape for x in sim)
        arrays = dict(
            robot=np.stack(b["robot"]).astype(np.float32),
            obj=np.stack(b["obj"]).astype(np.float32),
            events=np.stack(b["events"]).astype(np.uint8),
            preds=np.stack(b["preds"]).astype(np.uint8) if b["preds"][0].size else np.zeros((len(b["robot"]), 0), np.uint8),
            contact=np.stack(b["contact"]).astype(np.uint8),
            success=np.asarray(b["success"], dtype=np.uint8),
            action=(np.stack(b["action"]).astype(np.float32) if b["action"] else np.zeros((0, 16), np.float32)),
            onsets=(np.asarray(b["onsets"], dtype=np.int32) if b["onsets"] else np.zeros((0, 3), np.int32)),
            sim_t=np.asarray(b["sim_t"] if same else [], dtype=np.int32),
            sim_state=(np.stack(sim).astype(np.float32) if same else np.zeros((0, 0), np.float32)),
        )
        arrays["obj_cat"] = np.array([self.vocab.id(c) for c in self.ex.describe()["obj_categories"]],
                                     dtype=np.int32).reshape(n_obj)
        return arrays

    def flush(self, out_dir: Path, meta: Dict[str, Any], save: bool = True,
              min_transitions: int = 1) -> Optional[Path]:
        arrays = self.episode_arrays()
        path = None
        if save and arrays and arrays["action"].shape[0] >= min_transitions:
            m = dict(meta)
            m.update(self.ex.describe())
            m.setdefault("task", self.task)
            m["final_success"] = bool(arrays["success"][-1])
            m["ever_success"] = bool(arrays["success"].max())
            path = write_episode(Path(out_dir), arrays, m)
        self._buf = None
        return path
