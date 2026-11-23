"""Start a planner solution from an arbitrary saved simulator state.

IG solutions begin with ``env.reset(seed=seed)``. This wrapper performs that reset (so the
scene is configured exactly as for that seed) and then restores a saved flat simulator
state (``env.unwrapped.get_state()``) recorded earlier in an episode of the same scene and
seed. The solution then plans from the restored object/robot configuration.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch


class StartStateOverride(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pending: Optional[np.ndarray] = None

    def set_next_start(self, flat_state: Optional[np.ndarray]) -> None:
        self.pending = None if flat_state is None else np.asarray(flat_state, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.pending is not None:
            base = self.env.unwrapped
            st = torch.as_tensor(self.pending, device=base.device).reshape(1, -1)
            base.set_state(st)
            # Let PD targets follow the restored joints so the arm does not snap back.
            try:
                for agent in base.agent.agents:
                    agent.controller.reset()
            except Exception:
                pass
            obs = base.get_obs()
            self.pending = None
        return obs, info


def capture_rgb(env, camera: str = "zed2i") -> Optional[np.ndarray]:
    """[H, W, 3] uint8 from the env's sensor (obs_mode must include rgb)."""
    try:
        obs = env.unwrapped.get_obs()
        sd = obs["sensor_data"]
        cam = camera if camera in sd else next(iter(sd))
        img = sd[cam]["rgb"]
        img = img[0] if img.ndim == 4 else img
        return img.detach().cpu().numpy().astype(np.uint8) if hasattr(img, "detach") else np.asarray(img, np.uint8)
    except Exception:
        return None
