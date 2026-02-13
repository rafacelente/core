from typing import Optional, cast
import math

from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from pydantic import BaseModel, ConfigDict, Field

from core.modules.layer_norm import LayerNorm, LayerNormType, LayerNormConfig
from core.modules.rope import RoPEType, RoPEConfig
from core.utils import BufferCache
from core.modules.feed_forward import Activation, ActivationType

class AttentionType(str, Enum):
    DEFAULT = "default"
    NORMALIZED = "normalized"

class DefaultAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rope_config: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        qk_norm_config: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        cache: Optional[BufferCache] = None,
        use_post_sdpa_gate: bool = False,
        gate_activation_type: ActivationType = ActivationType.SIGMOID,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or d_model // n_heads
        self.use_rope = rope_config is not None
        self.rope_config = rope_config
        self.clip_qkv = clip_qkv
        self.qk_norm_config = qk_norm_config
        self.dropout_p = dropout
        self.use_post_sdpa_gate = use_post_sdpa_gate
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        query_size = self.n_heads * self.head_dim
        key_size = self.n_kv_heads * self.head_dim
        value_size = self.n_kv_heads * self.head_dim
        self.w_q = nn.Linear(d_model, query_size, bias=False)
        self.w_k = nn.Linear(d_model, key_size, bias=False)
        self.w_v = nn.Linear(d_model, value_size, bias=False)
        self.w_o = nn.Linear(query_size, d_model, bias=False)

        if self.use_rope:
            self.rope = self.rope_config.build(head_size=self.head_dim, cache=cache)
        self._setup_qk_norm()

        if self.use_post_sdpa_gate:
            self.post_sdpa_gate = nn.Linear(d_model, d_model, bias=False)
            self.gate_activation = Activation.build(gate_activation_type)

    def _setup_qk_norm(self) -> None:
        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        if self.qk_norm_config is not None:
            self.q_norm = self.qk_norm_config.build(hidden_size=self.head_dim)
            self.k_norm = self.qk_norm_config.build(hidden_size=self.head_dim)

    def sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: B, T, n_heads, head_dim -> B, n_heads, T, head_dim
        # k: B, T, n_kv_heads, head_dim -> B, n_kv_heads, T, head_dim
        # v: B, T, n_kv_heads, head_dim -> B, n_kv_heads, T, head_dim
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(
            backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH],
        ):
            att = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=True, enable_gqa=True
            )
        return att.transpose(1, 2).contiguous()

    def forward(
        self, x: torch.Tensor, pos_sin: Optional[torch.Tensor] = None, pos_cos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(bs, seq_len, -1, self.head_dim)
        k = k.view(bs, seq_len, -1, self.head_dim)
        v = v.view(bs, seq_len, -1, self.head_dim)

        if self.clip_qkv is not None:
            q = torch.clamp(q, -self.clip_qkv, self.clip_qkv)
            k = torch.clamp(k, -self.clip_qkv, self.clip_qkv)
            v = torch.clamp(v, -self.clip_qkv, self.clip_qkv)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if self.use_rope:
            q, k = self.rope(q, k, pos_sin, pos_cos, interleaved=False)
        
        att = self.sdpa(q, k, v)
        att = att.view(bs, seq_len, -1)

        if self.use_post_sdpa_gate:
            att = att * self.gate_activation(self.post_sdpa_gate(att))

        return self.w_o(att)


class NormalizedAttention(DefaultAttention):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rope_config: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        qk_norm_config: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        cache: Optional[BufferCache] = None,
        use_post_sdpa_gate: bool = False,
        gate_activation_type: ActivationType = ActivationType.SIGMOID,
    ):
        super().__init__(d_model, n_heads, n_kv_heads, head_dim, rope_config, clip_qkv, qk_norm_config, dropout, cache, use_post_sdpa_gate, gate_activation_type)
        self.sq_init_value = 1.0
        self.sk_init_value = 1.0
        self.sq_init_scaling = 1.0 / math.sqrt(d_model)
        self.sk_init_scaling = 1.0 / math.sqrt(d_model)
        self.sq = nn.Parameter(self.sq_init_scaling * torch.ones(self.n_heads * self.head_dim))
        self.sk = nn.Parameter(self.sk_init_scaling * torch.ones(self.n_kv_heads * self.head_dim))

        self.sqrt_head_dim = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, pos_sin: Optional[torch.Tensor] = None, pos_cos: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        sq = (self.sq * (self.sq_init_value / self.sq_init_scaling)).view(1, 1, -1)
        q = sq * q

        sk = (self.sk * (self.sk_init_value / self.sk_init_scaling)).view(1, 1, -1)
        k = sk * k

        q = q.view(bs, seq_len, -1, self.head_dim) # B, T, n_heads, head_dim
        k = k.view(bs, seq_len, -1, self.head_dim) # B, T, n_kv_heads, head_dim
        v = v.view(bs, seq_len, -1, self.head_dim) # B, T, n_kv_heads, head_dim

        if self.use_rope:
            q, k = self.rope(q, k, pos_sin, pos_cos)

        att = self.sdpa(q, k, v)
        att = att.view(bs, seq_len, -1)
        return self.w_o(att)

    def justnorm(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / x.norm(p=2, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)

    def _normalize_matrix(self, m: torch.Tensor, dim: int = -1) -> torch.Tensor:
        m.copy_(self.justnorm(m, dim=dim))

    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.w_q.weight)
        self._normalize_matrix(self.w_k.weight)
        self._normalize_matrix(self.w_v.weight)
        self._normalize_matrix(self.w_o.weight, dim=0)


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        rope_config: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        qk_norm_config: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        cache: Optional[BufferCache] = None,
        use_post_sdpa_gate: bool = False,
        gate_activation_type: ActivationType = ActivationType.SIGMOID,
    ):
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def build(
        type: AttentionType,
        **kwargs,
    ) -> "Attention":
        if type == AttentionType.DEFAULT:
            return cast(Attention, DefaultAttention(**kwargs))
        elif type == AttentionType.NORMALIZED:
            return cast(Attention, NormalizedAttention(**kwargs))
        else:
            raise ValueError(f"Invalid attention type: {type}")

class AttentionConfig(BaseModel):
    """
    Configuration for the attention.
    """

    type: AttentionType = AttentionType.DEFAULT
    n_heads: int
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rope: Optional[RoPEConfig] = Field(default_factory=lambda: RoPEConfig())
    clip_qkv: Optional[float] = None
    qk_norm: Optional[LayerNormConfig] = None
    dropout: float = 0.0
    use_post_sdpa_gate: bool = False
    gate_activation_type: Optional[ActivationType] = ActivationType.SIGMOID

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @property
    def effective_kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    def build(self, d_model: int, cache: Optional[BufferCache] = None) -> "Attention":
        if d_model % self.n_heads != 0 and self.head_dim is None:
            raise ValueError(f"d_model must be divisible by n_heads if head_dim is not provided, got d_model: {d_model}, n_heads: {self.n_heads}")
        if d_model % self.effective_kv_heads != 0 and self.head_dim is None:
            raise ValueError(f"d_model must be divisible by effective_kv_heads if head_dim is not provided, got d_model: {d_model}, effective_kv_heads: {self.effective_kv_heads}")

        return Attention.build(
            type=self.type,
            d_model=d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            rope_config=self.rope,
            clip_qkv=self.clip_qkv,
            qk_norm_config=self.qk_norm,
            dropout=self.dropout,
            cache=cache,
            use_post_sdpa_gate=self.use_post_sdpa_gate,
            gate_activation_type=self.gate_activation_type,
        )