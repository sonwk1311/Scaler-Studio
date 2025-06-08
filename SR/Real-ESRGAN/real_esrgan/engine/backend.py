from pathlib import Path

import torch
from torch import nn, Tensor

from real_esrgan.utils.checkpoint import load_checkpoint

__all__ = [
    "SuperResolutionBackend",
]


class SuperResolutionBackend(nn.Module):
    def __init__(self, weights_path: str | Path, device: torch.device = None):
        super().__init__()
        assert isinstance(weights_path, str) and Path(weights_path).suffix == ".pkl", f"{Path(weights_path).suffix} format is not supported."
        model = load_checkpoint(weights_path, map_location=device)
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
