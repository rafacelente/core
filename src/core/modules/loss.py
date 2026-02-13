from enum import Enum
from abc import abstractmethod
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from core.kernels.cute.cross_entropy import _FusedCrossEntropyFunction

class LossType(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    FUSED_CROSS_ENTROPY = "fused_cross_entropy"


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def build(
        type: LossType,
        ignore_index: int = -100,
    ) -> "Loss":
        if type == LossType.CROSS_ENTROPY:
            return cast(Loss, CrossEntropyLoss(ignore_index=ignore_index))
        elif type == LossType.FUSED_CROSS_ENTROPY:
            return cast(Loss, FusedCrossEntropyLoss(ignore_index=ignore_index))
        else:
            raise ValueError(f"Invalid loss type: {type}")


class CrossEntropyLoss(Loss):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)


class FusedCrossEntropyLoss(Loss):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = _FusedCrossEntropyFunction.apply(logits, labels, self.ignore_index)
        valid_count = (labels != self.ignore_index).sum()
        return loss.sum() / valid_count.float()


class LossConfig(BaseModel):
    """
    Configuration for the loss.
    """

    type: LossType = LossType.CROSS_ENTROPY
    ignore_index: int = -100

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def build(self) -> "Loss":
        return Loss.build(type=self.type, ignore_index=self.ignore_index)