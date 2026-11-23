"""f_theta: a transformer over object tokens (proposal Sec. 4.2 / 6.5).

Tokens: [CLS] + 2 robot tokens + N object tokens (+ category embedding) + 1 event token.
Four pre-LN transformer layers; the CLS output is projected to the head's input size
(IQE: 64 components x 32 = 2048).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from qap.state.schema import EVENT_TYPES, OBJ_DIM, ROBOT_DIM


@dataclass
class EncoderConfig:
    num_categories: int = 512
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.0
    out_dim: int = 2048
    use_events: bool = True


class ObjectTokenEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.robot_in = nn.Sequential(nn.Linear(ROBOT_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.obj_in = nn.Sequential(nn.Linear(OBJ_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.event_in = nn.Sequential(nn.Linear(len(EVENT_TYPES), d), nn.GELU(), nn.Linear(d, d))
        self.cat_emb = nn.Embedding(cfg.num_categories, d)
        self.arm_emb = nn.Parameter(torch.randn(2, d) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)   # cls, robot, object, event
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, cfg.out_dim))

    def forward(self, s) -> torch.Tensor:
        robot, obj, cat, mask, ev = s["robot"], s["obj"], s["cat"], s["mask"], s["events"]
        B = robot.shape[0]
        cat = cat.clamp(0, self.cfg.num_categories - 1)
        r = self.robot_in(robot) + self.arm_emb[None] + self.type_emb[1]
        o = self.obj_in(obj) + self.cat_emb(cat) + self.type_emb[2]
        toks = [self.cls.expand(B, -1, -1) + self.type_emb[0], r, o]
        pad = [torch.zeros(B, 1 + 2, dtype=torch.bool, device=robot.device), ~mask]
        if self.cfg.use_events:
            toks.append((self.event_in(ev) + self.type_emb[3])[:, None])
            pad.append(torch.zeros(B, 1, dtype=torch.bool, device=robot.device))
        x = torch.cat(toks, 1)
        h = self.tf(x, src_key_padding_mask=torch.cat(pad, 1))
        return self.out(h[:, 0])
