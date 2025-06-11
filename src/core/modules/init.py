from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from core.modules.attention import Attention
from core.modules.feed_forward import FeedForward, FeedForwardType


class InitMethod(str, Enum):
    NORMAL = "normal"
    LLAMA = "llama"

    def _init_linear(self, module: nn.Linear, std: float = 0.02, generator: Optional[torch.Generator] = None) -> None:
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def init_embeddings(
        self, module: nn.Embedding, std: float = 0.02, generator: Optional[torch.Generator] = None
    ) -> None:
        if self in [InitMethod.LLAMA]:
            nn.init.normal_(module.weight, generator=generator)
        else:
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-3 * 0.02, b=3 * 0.02, generator=generator)

    def init_final_w_out(
        self, module: nn.Linear, d_model: int, std: float = 0.02, generator: Optional[torch.Generator] = None
    ) -> None:
        std = 0.02
        if self in [InitMethod.LLAMA]:
            std = d_model**-0.5
        self._init_linear(module, std, generator)

    def init_attention(self, module: Attention, num_blocks: int, generator: Optional[torch.Generator] = None) -> None:
        std = 0.02
        for w in [module.w_q, module.w_k, module.w_v]:
            self._init_linear(w, std, generator)

        if self == InitMethod.LLAMA:
            std = std / (2 * num_blocks) ** 0.5

        self._init_linear(module.w_o, std, generator)

    def _init_feed_forward_glu(
        self, module: FeedForward, num_blocks: int, generator: Optional[torch.Generator] = None
    ) -> None:
        std = 0.02
        self._init_linear(module.w1, std, generator)
        if self == InitMethod.LLAMA:
            std = std / (2 * num_blocks) ** 0.5
        self._init_linear(module.w3, std, generator)
        self._init_linear(module.w2, std, generator)

    def _init_feed_forward_mlp(
        self, module: FeedForward, num_blocks: int, generator: Optional[torch.Generator] = None
    ) -> None:
        std = 0.02
        self._init_linear(module.w1, std, generator)
        if self == InitMethod.LLAMA:
            std = std / (2 * num_blocks) ** 0.5
        self._init_linear(module.w2, std, generator)

    def init_feed_forward(
        self, module: FeedForward, d_model: int, num_blocks: int, generator: Optional[torch.Generator] = None
    ) -> None:
        if module.type == FeedForwardType.GLU:
            self._init_feed_forward_glu(module, num_blocks, generator)
        elif module.type == FeedForwardType.MLP:
            self._init_feed_forward_mlp(module, num_blocks, generator)