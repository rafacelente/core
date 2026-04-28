from typing import List, Literal

import torch
import torch.nn as nn

from core.modules.activation import Activation, ActivationType

from pydantic import BaseModel, ConfigDict


class NobleConfig(BaseModel):
    r: int = 64
    activation_type: ActivationType = ActivationType.COSNET
    apply_to: List[Literal["all", "att", "ff", "lm_head"]] = ["all"]

    model_config = ConfigDict(extra="forbid")


class Noble(nn.Module):
    def __init__(
        self,
        r: int,
        d_in: int,
        d_out: int,
        activation: Activation,
    ):
        super().__init__()
        self.w_up = nn.Linear(r, d_out)
        self.w_down = nn.Linear(d_in, r)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_up(self.activation(self.w_down(x)))


class LinearNoble(nn.Module):
    def __init__(
        self,
        r: int,
        d_in: int,
        d_out: int,
        activation: Activation,
        **kwargs,
    ):
        super().__init__()
        self.noble = Noble(r, d_in, d_out, activation)
        self.linear = nn.Linear(d_in, d_out, **kwargs)

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.noble(x) + self.linear(x)
