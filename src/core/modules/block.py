from typing import Optional
import math

import torch
import torch.nn as nn

from core.config import AttentionConfig, FeedForwardConfig, LayerNormConfig
from core.utils import BufferCache


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        block_idx: int,
        attention_config: AttentionConfig,
        feed_forward_config: FeedForwardConfig,
        layer_norm_config: LayerNormConfig,
        dropout: float = 0.0,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention_config.build(d_model, cache)
        self.feed_forward = feed_forward_config.build(d_model, cache)
        self.att_norm = layer_norm_config.build(d_model)
        self.ff_norm = layer_norm_config.build(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.att_norm(x)))
        return h + self.dropout(self.feed_forward(self.ff_norm(h)))

class NormalizedBlock(Block):
    def __init__(
        self,
        d_model: int,
        block_idx: int,
        attention_config: AttentionConfig,
        feed_forward_config: FeedForwardConfig,
        layer_norm_config: LayerNormConfig,
        dropout: float = 0.0,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(d_model, block_idx, attention_config, feed_forward_config, layer_norm_config, dropout, cache)
        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.attn_alpha = nn.Parameter(self.attn_alpha_init_scaling * torch.ones(d_model))

        self.ff_alpha_init_value = 0.05
        self.ff_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.ff_alpha = nn.Parameter(self.ff_alpha_init_scaling * torch.ones(d_model))

    def justnorm(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / x.norm(p=2, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.justnorm(
            torch.lerp(
                x,
                self.attention(x),
                (self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)).abs(),
            )
        )
        return self.justnorm(
            torch.lerp(
                h,
                self.feed_forward(h),
                (self.ff_alpha * (self.ff_alpha_init_value / self.ff_alpha_init_scaling)).abs(),
            )
        )