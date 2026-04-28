from typing import Optional
from enum import Enum

import torch.nn as nn
from pydantic import BaseModel, ConfigDict

from core.modules.activation import Activation
from core.modules.noble import NobleConfig, LinearNoble


class LinearType(str, Enum):
    DEFAULT = "default"
    NOBLE = "noble"


class LinearConfig(BaseModel):
    type: LinearType = LinearType.DEFAULT
    noble: Optional[NobleConfig] = None

    model_config = ConfigDict(extra="forbid")

    def build(self, in_features: int, out_features: int, **kwargs) -> nn.Module:
        if self.type == LinearType.DEFAULT:
            return nn.Linear(in_features, out_features, **kwargs)
        elif self.type == LinearType.NOBLE:
            activation = Activation.build(self.noble.activation_type, r=self.noble.r)
            return LinearNoble(
                r=self.noble.r,
                d_in=in_features,
                d_out=out_features,
                activation=activation,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown linear type: {self.type}")
