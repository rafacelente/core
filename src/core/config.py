from enum import Enum
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict

from core.modules.attention import AttentionType, Attention
from core.modules.feed_forward import ActivationType, FeedForward, FeedForwardType
from core.modules.layer_norm import LayerNorm, LayerNormType
from core.modules.rope import RoPEType
from core.utils import BufferCache


class DType(str, Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    def to_torch_dtype(self) -> torch.dtype:
        if self == DType.FLOAT32 or self == DType.FP32:
            return torch.float32
        elif self == DType.BFLOAT16 or self == DType.BF16:
            return torch.bfloat16
        elif self == DType.FLOAT16 or self == DType.FP16:
            return torch.float16
        else:
            raise ValueError(f"Invalid dtype: {self}")


class FeedForwardConfig(BaseModel):
    """
    Configuration for the feed forward network.

    Default values are based on the GPT-2 MLP.

    Args:
        hidden_size: The hidden size of the feed forward network.
        ff_ratio: The ratio of the hidden size to the feed forward network.
        ff_hidden_size: The hidden size of the feed forward network.
        activation_type: The activation function of the feed forward network.
        feed_forward_type: The type of the feed forward network.
    """

    ff_ratio: float = 4.0
    ff_hidden_size: Optional[int] = None
    activation_type: ActivationType = ActivationType.GELU
    feed_forward_type: FeedForwardType = FeedForwardType.MLP
    dropout: float = 0.0
    dtype: DType = DType.FLOAT32

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def build(self, d_model: int, cache: Optional[BufferCache] = None) -> "FeedForward":
        # TODO: cache for MoE when implemented
        if self.ff_hidden_size is None:
            ff_hidden_size = int(d_model * self.ff_ratio)
        else:
            ff_hidden_size = self.ff_hidden_size
        return FeedForward.build(
            type=self.feed_forward_type,
            d_model=d_model,
            hidden_size=ff_hidden_size,
            activation_type=self.activation_type,
            dtype=self.dtype.to_torch_dtype(),
            cache=cache,
        )


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


class AttentionConfig(BaseModel):
    """
    Configuration for the attention.
    """
    type: AttentionType = AttentionType.DEFAULT
    n_heads: int
    n_kv_heads: Optional[int] = None
    use_rope: bool = True
    rope_type: RoPEType = RoPEType.DEFAULT
    clip_qkv: Optional[float] = None
    qk_norm_type: Optional[LayerNormType] = None
    dropout: float = 0.0
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @property
    def effective_kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    def build(self, d_model: int, cache: Optional[BufferCache] = None) -> "Attention":
        if d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {self.n_heads}")
        if d_model % self.effective_kv_heads != 0:
            raise ValueError(
                f"d_model must be divisible by effective_kv_heads, got {d_model} and {self.effective_kv_heads}"
            )
        return Attention.build(
            type=self.type,
            d_model=d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            use_rope=self.use_rope,
            rope_type=self.rope_type,
            clip_qkv=self.clip_qkv,
            qk_norm_type=self.qk_norm_type,
            dropout=self.dropout,
            cache=cache,
        )