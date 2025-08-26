from abc import abstractmethod
from enum import Enum
from typing import Optional, cast

import math

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
    NORMALIZED_MLP = "normalized_mlp"
    NORMALIZED_GLU = "normalized_glu"


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

class NormalizedMLP(MLP):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        activation_type: ActivationType,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(d_model, hidden_size, activation_type, dtype)
        self.sw1_init_value = 1.0
        self.sw1_init_scaling = 1.0 / math.sqrt(d_model)
        self.sw1 = nn.Parameter(self.sw1_init_scaling * torch.ones(hidden_size))
        self.sqrt_d_model = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sw1 = self.sw1 * ((self.sw1_init_value / self.sw1_init_scaling) * self.sqrt_d_model)
        return self.w2(self.activation(sw1 * self.w1(x)))

    def justnorm(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / x.norm(p=2, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)

    def _normalize_matrix(self, m: torch.Tensor, dim: int = -1) -> torch.Tensor:
        m.copy_(self.justnorm(m, dim=dim) * self.sqrt_d_model)

    @torch.no_grad()
    def normalize_matrices(self) -> None:
        self._normalize_matrix(self.w1.weight)
        self._normalize_matrix(self.w2.weight, dim=0)


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

class NormalizedGLU(GLU):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        activation_type: ActivationType,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(d_model, hidden_size, activation_type, dtype)
        self.sw_init_value = 1.0
        self.sw_init_scaling = 1.0 / math.sqrt(d_model)
        self.sw1 = nn.Parameter(self.sw_init_scaling * torch.ones(hidden_size))
        self.sw3 = nn.Parameter(self.sw_init_scaling * torch.ones(hidden_size))
        self.sqrt_d_model = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sw1 = self.sw1 * ((self.sw_init_value / self.sw_init_scaling) * self.sqrt_d_model)
        sw3 = self.sw3 * ((self.sw_init_value / self.sw_init_scaling))
        return self.w2(self.activation(sw1 * self.w1(x)) * sw3 * self.w3(x))

    def justnorm(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / x.norm(p=2, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)

    def _normalize_matrix(self, m: torch.Tensor, dim: int = -1) -> torch.Tensor:
        m.copy_(self.justnorm(m, dim=dim))
    
    @torch.no_grad()
    def normalize_matrices(self) -> None:
        self._normalize_matrix(self.w1.weight)
        self._normalize_matrix(self.w2.weight, dim=0)
        self._normalize_matrix(self.w3.weight)


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
        elif type == FeedForwardType.NORMALIZED_MLP:
            return cast(FeedForward, NormalizedMLP(d_model, hidden_size, activation_type, dtype))
        elif type == FeedForwardType.NORMALIZED_GLU:
            return cast(FeedForward, NormalizedGLU(d_model, hidden_size, activation_type, dtype))
        else:
            raise ValueError(f"Invalid feed forward type: {type}")