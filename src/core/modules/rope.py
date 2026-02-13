from abc import abstractmethod
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from einops import repeat, rearrange


from core.utils import BufferCache
from core.kernels.triton.rotary import apply_rotary_emb as apply_rotary_emb_triton


class RoPEType(str, Enum):
    DEFAULT = "default"
    FUSED = "fused"


class RoPEConfig(BaseModel):
    """
    Configuration for the RoPE module.
    """

    type: RoPEType = RoPEType.DEFAULT
    theta: int = 10_000

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def build(self, head_size: int, **kwargs) -> "RoPE":
        return RoPE(head_size, theta=self.theta, use_fused=self.type == RoPEType.FUSED, **kwargs)


class RoPE(nn.Module):
    def __init__(
        self,
        head_size: int,
        theta: int = 10_000,
        cache: Optional[BufferCache] = None,
        use_fused: bool = True,
    ):
        super().__init__()
        self._cache = cache or BufferCache()
        self.dim = head_size
        self.theta = theta
        self.use_fused = use_fused

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
            freqs = torch.outer(t, inv_freq)
            pos_sin = freqs.sin()
            pos_cos = freqs.cos()
        self._cache[f"pos_sin"] = pos_sin
        self._cache[f"pos_cos"] = pos_cos
        return pos_cos, pos_sin

    def warmup_cache(self, max_seq_len: int, device: torch.device) -> None:
        self._get_rotary_embedding(max_seq_len, device)

    def rotate_half(self, x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    def _apply_rotary_embedding(self, x, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
        if self.use_fused:
            return apply_rotary_emb_triton(x, cos, sin, interleaved)
        else:
            ro_dim = cos.shape[-1] * 2
            assert ro_dim <= x.shape[-1], "RoPE dimension must be less than or equal to the input dimension"
            cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
            sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
            return torch.cat(
                [x[..., :ro_dim] * cos + self.rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
                dim=-1,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        interleaved: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len = q.size(1)
        k_len = k.size(1)
        if pos_sin is None or pos_cos is None:
            pos_cos, pos_sin = self._get_rotary_embedding(k_len, q.device)
        q = self._apply_rotary_embedding(
            q, pos_cos, pos_sin, interleaved
        )
        k = self._apply_rotary_embedding(
            k, pos_cos, pos_sin, interleaved
        )
        return q, k