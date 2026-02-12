from dataclasses import dataclass
from typing import List, Optional

import torch

@dataclass
class CoreOutput:
    logits: torch.Tensor
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    loss: Optional[torch.Tensor] = None