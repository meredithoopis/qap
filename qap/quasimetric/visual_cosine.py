"""Visual-cosine baseline distance: 1 - cos(DINOv2(img_s), DINOv2(img_g)).

Same backbone loader as the IG baselines (examples/baselines/encoders/obs_encoder.py:
torch.hub 'facebookresearch/dinov2:main'). Symmetric and appearance-based by design.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoFeatures:
    def __init__(self, model_name: str = "dinov2_vitl14", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2:main", model_name).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, imgs: np.ndarray, batch: int = 64) -> torch.Tensor:
        """imgs [B, H, W, 3] uint8 -> [B, C] L2-normalised CLS features."""
        out = []
        for i in range(0, len(imgs), batch):
            x = torch.as_tensor(imgs[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = ((x - _MEAN) / _STD).to(self.device)
            f = self.model(x)
            out.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(out, 0)


def visual_cosine_distance(feat_s: torch.Tensor, feat_g: torch.Tensor) -> torch.Tensor:
    return 1.0 - (feat_s * feat_g).sum(-1)


_DINO: Optional[DinoFeatures] = None


def get_dino(device: str = "cuda") -> DinoFeatures:
    global _DINO
    if _DINO is None:
        _DINO = DinoFeatures(device=device)
    return _DINO
