from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from core.modules.layer_norm import LayerNorm, LayerNormType
from core.modules.rope import RoPE, RoPEType
from core.utils import BufferCache


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        use_rope: bool = True,
        rope_type: RoPEType = RoPEType.DEFAULT,
        clip_qkv: Optional[float] = None,
        qk_norm_type: Optional[LayerNormType] = None,
        dropout: float = 0.0,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.use_rope = use_rope
        self.rope_type = rope_type
        self.clip_qkv = clip_qkv
        self.qk_norm_type = qk_norm_type
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        if self.use_rope:
            self.rope = RoPE.build(type=self.rope_type, head_size=self.head_dim, cache=cache)
        self._setup_qk_norm()

    def _setup_qk_norm(self) -> None:
        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        if self.qk_norm_type is not None:
            self.q_norm = LayerNorm.build(type=self.qk_norm_type, hidden_size=self.head_dim)
            self.k_norm = LayerNorm.build(type=self.qk_norm_type, hidden_size=self.head_dim)

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
        if self.clip_qkv is not None:
            q = torch.clamp(q, -self.clip_qkv, self.clip_qkv)
            k = torch.clamp(k, -self.clip_qkv, self.clip_qkv)
            v = torch.clamp(v, -self.clip_qkv, self.clip_qkv)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q.view(bs, seq_len, -1, self.head_dim)
        k = k.view(bs, seq_len, -1, self.head_dim)
        v = v.view(bs, seq_len, -1, self.head_dim)
        if self.use_rope:
            q, k = self.rope(q, k, pos_sin, pos_cos)
        att = self.sdpa(q, k, v)
        att = att.view(bs, seq_len, -1)
        return self.w_o(att)