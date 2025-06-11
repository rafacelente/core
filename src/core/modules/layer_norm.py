from abc import abstractmethod
from enum import Enum
from typing import cast

import torch
import torch.nn as nn


class LayerNormType(str, Enum):
    DEFAULT = "default"
    RMS = "rms"
    RMS_FAST = "rms_fast"


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def build(type: LayerNormType, hidden_size: int, eps: float = 1e-5, **kwargs) -> "LayerNorm":
        if type == LayerNormType.DEFAULT:
            return cast(LayerNorm, nn.LayerNorm(hidden_size, eps=eps, **kwargs))
        elif type == LayerNormType.RMS:
            return cast(LayerNorm, nn.RMSNorm(hidden_size, eps=eps, **kwargs))
        elif type == LayerNormType.RMS_FAST:
            raise NotImplementedError("RMSFast is not implemented yet.")
        else:
            raise ValueError(f"Invalid layer norm type: {type}")