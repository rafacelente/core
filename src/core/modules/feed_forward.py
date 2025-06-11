from abc import abstractmethod
from enum import Enum
from typing import Optional, cast

import torch
import torch.nn as nn

from core.utils import BufferCache


class ActivationType(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


class FeedForwardType(str, Enum):
    MLP = "mlp"
    GLU = "glu"


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def build(type: ActivationType) -> "Activation":
        if type == ActivationType.RELU:
            return cast(Activation, nn.ReLU())
        elif type == ActivationType.GELU:
            return cast(Activation, nn.GELU())
        elif type == ActivationType.SILU:
            return cast(Activation, nn.SiLU())
        else:
            raise ValueError(f"Invalid activation type: {type}")


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        activation_type: ActivationType,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.w1 = nn.Linear(in_features=d_model, out_features=hidden_size, dtype=dtype, bias=False)
        self.activation = Activation.build(activation_type)
        self.w2 = nn.Linear(in_features=hidden_size, out_features=d_model, dtype=dtype, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x)))


class GLU(nn.Module):
    def __init__(
        self, d_model: int, hidden_size: int, activation_type: ActivationType, dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(in_features=d_model, out_features=hidden_size, dtype=dtype, bias=False)
        self.w2 = nn.Linear(in_features=hidden_size, out_features=d_model, dtype=dtype, bias=False)
        self.w3 = nn.Linear(in_features=d_model, out_features=hidden_size, dtype=dtype, bias=False)
        self.activation = Activation.build(activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def build(
        type: FeedForwardType,
        d_model: int,
        hidden_size: int,
        activation_type: ActivationType,
        dtype: torch.dtype = torch.float32,
        # TODO: cache for MoE when implemented
        cache: Optional[BufferCache] = None,
    ) -> "FeedForward":
        if type == FeedForwardType.MLP:
            return cast(FeedForward, MLP(d_model, hidden_size, activation_type, dtype))
        elif type == FeedForwardType.GLU:
            return cast(FeedForward, GLU(d_model, hidden_size, activation_type, dtype))
        else:
            raise ValueError(f"Invalid feed forward type: {type}")