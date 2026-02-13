from abc import abstractmethod
from enum import Enum
from typing import cast

import torch
import torch.nn as nn
from quack.rmsnorm import QuackRMSNorm
from pydantic import BaseModel, ConfigDict
from core.utils import is_sm100_or_higher, DType


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
            if is_sm100_or_higher():
                return cast(LayerNorm, QuackRMSNorm(hidden_size, eps=eps, **kwargs))
            else:
                device_capability = torch.cuda.get_device_capability()
                raise ValueError(f"RMSFast is only supported on SM100 or higher. Current device capability: {device_capability}. Use LayerNormType.RMS instead.")
        else:
            raise ValueError(f"Invalid layer norm type: {type}")

class LayerNormConfig(BaseModel):
    """
    Configuration for the layer norm.
    """

    dtype: DType = DType.FLOAT32
    layer_norm_type: LayerNormType = LayerNormType.DEFAULT
    eps: float = 1e-5

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def build(self, hidden_size: int) -> "LayerNorm":
        return LayerNorm.build(
            type=self.layer_norm_type,
            hidden_size=hidden_size,
            dtype=self.dtype.to_torch_dtype(),
            eps=self.eps,
        )