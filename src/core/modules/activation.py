from abc import abstractmethod
from enum import Enum
from typing import cast

import torch
import torch.nn as nn

from core.utils import DType

class CosNet(nn.Module):
    def __init__(
        self,
        r: int,
        omega_min: float = 0.8,
        omega_max: float = 1.2,
        phi_sigma: float = 0.1,
        omega_dtype: DType = DType.FLOAT32,
        phi_dtype: DType = DType.FLOAT32,
    ):
        super().__init__()
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.phi_sigma = phi_sigma
        self.omega_1 = nn.Parameter(torch.empty(r, dtype=omega_dtype.to_torch_dtype()))
        self.omega_2 = nn.Parameter(torch.empty(r, dtype=omega_dtype.to_torch_dtype()))
        self.phi_1 = nn.Parameter(torch.empty(r, dtype=phi_dtype.to_torch_dtype()))
        self.phi_2 = nn.Parameter(torch.empty(r, dtype=phi_dtype.to_torch_dtype()))
        self.M = nn.Linear(r, r, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.omega_2 * (self.M(torch.cos(self.omega_1 * x + self.phi_1))) + self.phi_2)

    def reset_parameters(self):
        with torch.no_grad():
            self.omega_1.uniform_(self.omega_min, self.omega_max)
            self.omega_2.uniform_(self.omega_min, self.omega_max)
            self.phi_1.normal_(mean=0.0, std=self.phi_sigma)
            self.phi_2.normal_(mean=0.0, std=self.phi_sigma)
        nn.init.xavier_normal_(self.M.weight)


class ActivationType(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    COSNET = "cosnet"


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def build(type: ActivationType, **kwargs) -> "Activation":
        if type == ActivationType.RELU:
            return cast(Activation, nn.ReLU())
        elif type == ActivationType.GELU:
            return cast(Activation, nn.GELU())
        elif type == ActivationType.SILU:
            return cast(Activation, nn.SiLU())
        elif type == ActivationType.SIGMOID:
            return cast(Activation, nn.Sigmoid())
        elif type == ActivationType.COSNET:
            return cast(Activation, CosNet(**kwargs))
        else:
            raise ValueError(f"Invalid activation type: {type}")
