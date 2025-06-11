from abc import abstractmethod
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel

from core.utils import BufferCache


class RoPEType(str, Enum):
    DEFAULT = "default"
    FUSED = "fused"


class RoPEConfig(BaseModel):
    """
    Configuration for the RoPE module.
    """

    type: RoPEType = RoPEType.DEFAULT
    theta: int = 10_000


class RoPE(nn.Module):
    def __init__(
        self,
        head_size: int,
        theta: int = 10_000,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.dim = head_size
        self.theta = theta
        self._cache = cache

    @abstractmethod
    def warmup_cache(self, max_seq_len: int, device: torch.device) -> None:
        raise NotImplementedError

    @staticmethod
    def build(type: RoPEType, head_size: int, **kwargs) -> "RoPE":
        if type == RoPEType.DEFAULT:
            return DefaultRoPE(head_size, **kwargs)
        elif type == RoPEType.FUSED:
            raise NotImplementedError("Fused RoPE is not implemented yet")
        else:
            raise ValueError(f"Invalid RoPE type: {type}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class DefaultRoPE(RoPE):
    def __init__(
        self,
        head_size: int,
        theta: int = 10_000,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(head_size, theta, cache)
        self._cache = cache or BufferCache()

    def _get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_sin = self._cache.get(f"pos_sin")
        pos_cos = self._cache.get(f"pos_cos")
        if (
            (pos_sin is not None)
            and (pos_cos is not None)
            and (pos_sin.shape[-2] >= seq_len)
            and (pos_cos.shape[-2] >= seq_len)
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self._cache[f"pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self._cache[f"pos_cos"] = pos_cos
            return pos_sin[:seq_len, :], pos_cos[:seq_len, :]
        with torch.autocast(device_type=device.type, enabled=False):
            inv_freq = 1.0 / (
                self.theta ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
            )
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            pos_sin = emb.sin()
            pos_cos = emb.cos()
        self._cache[f"pos_sin"] = pos_sin
        self._cache[f"pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def warmup_cache(self, max_seq_len: int, device: torch.device) -> None:
        self._get_rotary_embedding(max_seq_len, device)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, n_heads, T, hs = x.size()
        x = x.view(B, n_heads, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_embedding(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len = q.size(1)
        k_len = k.size(1)
        if pos_sin is None or pos_cos is None:
            pos_sin, pos_cos = self._get_rotary_embedding(k_len, q.device)
        q = self._apply_rotary_embedding(
            pos_sin[None, k_len - q_len : k_len, None, :], pos_cos[None, k_len - q_len : k_len, None, :], q
        )
        k = self._apply_rotary_embedding(pos_sin[None, :, None, :], pos_cos[None, :, None, :], k)
        return q, k