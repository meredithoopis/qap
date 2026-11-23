"""In-memory store of recorded robot episodes + the samplers used by every objective.

All episodes (MP corpus + play [+ graded]) are concatenated into flat, padded arrays:
    robot  [S, 2, R]   obj [S, Nmax, D]   cat [S, Nmax]   mask [S, Nmax]   events [S, E]
S = total number of states. Transitions use a frame skip k: (s_t, s_{t+k}) counts as one
step of cost 1, so distances are in units of k control steps (k=5 by default, ~0.25 s).

Samplers (all return dicts of torch tensors on `device`):
    transitions(B)          (s, s', a) with s' = s_{t+k}
    random_states(B)        uniform over all states (QRL 'random pairs')
    future(B, gamma)        (s_t, s_{t+j}) with j ~ Geom(1-gamma) in units of k, truncated
    hindsight(B, ...)       AWR batch: (s, a, s', g) with g from the same episode future or random
    same_episode_pairs(B)   (s_t, s_u, u>t) within planner episodes (distillation targets)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from qap.common.io import iter_episode_files
from qap.state.categories import CategoryVocab
from qap.state.schema import EVENT_TYPES, OBJ_DIM, ROBOT_DIM

State = Dict[str, torch.Tensor]


@dataclass
class StoreConfig:
    max_objects: int = 16
    frame_skip: int = 5
    max_episodes: Optional[int] = None
    obj_dtype: str = "float32"


class EpisodeStore:
    def __init__(self, dirs: Iterable[Path], cfg: StoreConfig = StoreConfig(),
                 keep: Optional[Callable[[Dict], bool]] = None, vocab: Optional[CategoryVocab] = None,
                 verbose: bool = True):
        self.cfg = cfg
        self.vocab = vocab or CategoryVocab()
        robots, objs, cats, masks, events, actions, succ = [], [], [], [], [], [], []
        ep_start, ep_len, metas = [], [], []
        n_states = 0
        files = list(iter_episode_files(*dirs))
        for i, f in enumerate(files):
            if cfg.max_episodes and len(metas) >= cfg.max_episodes:
                break
            side = f.with_suffix(".json")
            meta = json.load(open(side, encoding="utf-8")) if side.exists() else {}
            if keep is not None and not keep(meta):
                continue
            with np.load(f) as z:
                T1 = z["robot"].shape[0]
                if T1 < cfg.frame_skip + 1:
                    continue
                n = min(z["obj"].shape[1], cfg.max_objects)
                o = np.zeros((T1, cfg.max_objects, OBJ_DIM), dtype=cfg.obj_dtype)
                o[:, :n] = z["obj"][:, :n]
                c = np.zeros((cfg.max_objects,), dtype=np.int64)
                names = meta.get("obj_categories")
                if names:
                    c[:n] = [self.vocab.id(x) for x in names[:n]]
                else:
                    c[:n] = z["obj_cat"][:n]
                m = np.zeros((cfg.max_objects,), dtype=bool)
                m[:n] = True
                a = z["action"]
                a = np.concatenate([a, np.zeros((1, a.shape[1]), a.dtype)], 0) if a.shape[0] == T1 - 1 else a[:T1]
                robots.append(z["robot"].astype(np.float32))
                objs.append(o)
                cats.append(np.repeat(c[None], T1, 0))
                masks.append(np.repeat(m[None], T1, 0))
                events.append(z["events"].astype(np.float32))
                actions.append(a.astype(np.float32))
                succ.append(z["success"].astype(bool))
            ep_start.append(n_states)
            ep_len.append(T1)
            metas.append(meta)
            n_states += T1
            if verbose and (len(metas) % 1000 == 0):
                print(f"  loaded {len(metas)} episodes, {n_states} states")
        if not metas:
            raise RuntimeError(f"no episodes found in {list(dirs)}")
        self.robot = torch.from_numpy(np.concatenate(robots))
        self.obj = torch.from_numpy(np.concatenate(objs))
        self.cat = torch.from_numpy(np.concatenate(cats))
        self.mask = torch.from_numpy(np.concatenate(masks))
        self.events = torch.from_numpy(np.concatenate(events))
        self.action = torch.from_numpy(np.concatenate(actions))
        self.success = torch.from_numpy(np.concatenate(succ))
        self.ep_start = torch.tensor(ep_start, dtype=torch.long)
        self.ep_len = torch.tensor(ep_len, dtype=torch.long)
        self.metas = metas
        ep_of = torch.repeat_interleave(torch.arange(len(ep_len)), self.ep_len)
        self.ep_of = ep_of
        t_in = torch.arange(n_states) - self.ep_start[ep_of]
        self.t_in = t_in
        k = cfg.frame_skip
        self.valid_t = torch.nonzero(t_in + k < self.ep_len[ep_of]).squeeze(1)   # states with a k-step successor
        self.is_planner = torch.tensor([m.get("source") in ("mp", "graded") for m in metas])
        if verbose:
            print(f"EpisodeStore: {len(metas)} episodes, {n_states} states, {len(self.valid_t)} k-transitions (k={k})")

    # ------------------------------------------------------------------ basic access
    @property
    def num_states(self) -> int:
        return self.robot.shape[0]

    def gather(self, idx: torch.Tensor, device) -> State:
        return dict(robot=self.robot[idx].to(device, non_blocking=True),
                    obj=self.obj[idx].float().to(device, non_blocking=True),
                    cat=self.cat[idx].to(device, non_blocking=True),
                    mask=self.mask[idx].to(device, non_blocking=True),
                    events=self.events[idx].to(device, non_blocking=True))

    # ------------------------------------------------------------------ samplers
    def transitions(self, B: int, device) -> Dict[str, State]:
        i = self.valid_t[torch.randint(len(self.valid_t), (B,))]
        j = i + self.cfg.frame_skip
        return dict(s=self.gather(i, device), s_next=self.gather(j, device),
                    action=self.action[i].to(device), idx=i)

    def random_states(self, B: int, device) -> State:
        return self.gather(torch.randint(self.num_states, (B,)), device)

    def _future_index(self, i: torch.Tensor, gamma: float) -> torch.Tensor:
        k = self.cfg.frame_skip
        ep = self.ep_of[i]
        last = self.ep_start[ep] + self.ep_len[ep] - 1
        geo = torch.distributions.Geometric(probs=torch.full((len(i),), 1.0 - gamma)).sample().long() + 1
        return torch.minimum(i + geo * k, last)

    def future(self, B: int, gamma: float, device) -> Dict[str, object]:
        i = self.valid_t[torch.randint(len(self.valid_t), (B,))]
        g = self._future_index(i, gamma)
        return dict(s=self.gather(i, device), g=self.gather(g, device),
                    s_next=self.gather(i + self.cfg.frame_skip, device),
                    steps=((g - i).float() / self.cfg.frame_skip).to(device), idx=i, gidx=g)

    def hindsight(self, B: int, gamma: float, p_random: float, device) -> Dict[str, object]:
        """AWR batch: action a_t (1 control step) at s_t; value is evaluated at s_{t+k}."""
        out = self.future(B, gamma, device)
        n_rand = int(p_random * B)
        if n_rand:
            r = torch.randint(self.num_states, (n_rand,))
            rg = self.gather(r, device)
            for key in out["g"]:
                out["g"][key][:n_rand] = rg[key]
        out["action"] = self.action[out["idx"]].to(device)
        return out

    def same_episode_pairs(self, B: int, device, planner_only: bool = True) -> Dict[str, object]:
        pool = self.valid_t
        if planner_only and bool(self.is_planner.any()):
            pool = pool[self.is_planner[self.ep_of[pool]]]
        i = pool[torch.randint(len(pool), (B,))]
        ep = self.ep_of[i]
        last = self.ep_start[ep] + self.ep_len[ep] - 1
        u = i + (torch.rand(B) * (last - i + 1).float()).long()
        return dict(s=self.gather(i, device), g=self.gather(u, device),
                    steps=((u - i).float() / self.cfg.frame_skip).to(device),
                    ev_s=self.events[i].to(device), ev_g=self.events[u].to(device))


def stack_states(states: List[State]) -> State:
    return {k: torch.cat([s[k] for s in states], 0) for k in states[0]}


def state_from_arrays(robot: np.ndarray, obj: np.ndarray, cat: np.ndarray, events: np.ndarray,
                      max_objects: int = 16, device="cpu") -> State:
    """Single state (or a batch with leading dim) from extractor outputs -> model input."""
    if robot.ndim == 2:
        robot, obj, events = robot[None], obj[None], events[None]
    B, n = obj.shape[0], min(obj.shape[1], max_objects)
    o = np.zeros((B, max_objects, OBJ_DIM), np.float32)
    o[:, :n] = obj[:, :n]
    c = np.zeros((B, max_objects), np.int64)
    c[:, :n] = np.asarray(cat).reshape(-1)[:n]
    m = np.zeros((B, max_objects), bool)
    m[:, :n] = True
    return dict(robot=torch.as_tensor(robot, dtype=torch.float32, device=device),
                obj=torch.as_tensor(o, device=device), cat=torch.as_tensor(c, device=device),
                mask=torch.as_tensor(m, device=device),
                events=torch.as_tensor(events, dtype=torch.float32, device=device))


assert ROBOT_DIM == 16 and len(EVENT_TYPES) > 0
