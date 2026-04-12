from typing import Optional
import logging

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict

try:
    from fla.layers.gla import GatedLinearAttention as FLAGatedLinearAttention
    from fla.layers.gated_deltanet import GatedDeltaNet as FLAGatedDeltaNet
    from fla.layers.gated_deltanet import GatedDeltaNet2simp as FLAGatedDeltaNet2Simp
except ImportError:
    logging.warning(
        "flash-linear-attention not found. "
        "Install with: pip install -e third_party/flash-linear-attention"
    )
    FLAGatedLinearAttention = None
    FLAGatedDeltaNet = None
    FLAGatedDeltaNet2Simp = None


class FLAConfig(BaseModel):
    """Configuration for FLA-based linear attention variants (GLA, Gated Delta Net)."""

    mode: str = "chunk"

    expand_k: float = 0.5
    expand_v: float = 1.0

    use_short_conv: bool = False
    conv_size: int = 4
    conv_bias: bool = False

    # GLA-specific
    use_output_gate: bool = True
    gate_fn: str = "swish"
    gate_logit_normalizer: int = 16
    gate_low_rank_dim: int = 16
    fuse_norm: bool = True

    # GDN-specific
    use_gate: bool = True
    allow_neg_eigval: bool = False
    num_v_heads: Optional[int] = None

    norm_eps: float = 1e-5

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GLAAttention(nn.Module):
    """
    Wrapper around FLA's GatedLinearAttention that conforms to core's
    Attention interface: forward(x) -> Tensor.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        fla_config: Optional[FLAConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        assert FLAGatedLinearAttention is not None, (
            "flash-linear-attention is required for GLA. "
            "Install with: pip install -e third_party/flash-linear-attention"
        )
        cfg = fla_config or FLAConfig()
        self.d_model = d_model
        self.use_rope = False

        self.gla = FLAGatedLinearAttention(
            mode=cfg.mode,
            hidden_size=d_model,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            expand_k=cfg.expand_k,
            expand_v=cfg.expand_v,
            use_short_conv=cfg.use_short_conv,
            conv_size=cfg.conv_size,
            conv_bias=cfg.conv_bias,
            use_output_gate=cfg.use_output_gate,
            gate_fn=cfg.gate_fn,
            gate_logit_normalizer=cfg.gate_logit_normalizer,
            gate_low_rank_dim=cfg.gate_low_rank_dim,
            fuse_norm=cfg.fuse_norm,
            norm_eps=cfg.norm_eps,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        o, _, _ = self.gla(x)
        return o


class GatedDeltaNetAttention(nn.Module):
    """
    Wrapper around FLA's GatedDeltaNet that conforms to core's
    Attention interface: forward(x) -> Tensor.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        fla_config: Optional[FLAConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        assert FLAGatedDeltaNet is not None, (
            "flash-linear-attention is required for GatedDeltaNet. "
            "Install with: pip install -e third_party/flash-linear-attention"
        )
        cfg = fla_config or FLAConfig(expand_v=2.0, use_short_conv=True)
        self.d_model = d_model
        self.use_rope = False

        self.gdn = FLAGatedDeltaNet(
            mode=cfg.mode,
            hidden_size=d_model,
            num_heads=n_heads,
            num_v_heads=cfg.num_v_heads,
            head_dim=head_dim or d_model // n_heads,
            expand_v=cfg.expand_v,
            use_gate=cfg.use_gate,
            use_short_conv=cfg.use_short_conv,
            conv_size=cfg.conv_size,
            conv_bias=cfg.conv_bias,
            allow_neg_eigval=cfg.allow_neg_eigval,
            norm_eps=cfg.norm_eps,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        o, _, _ = self.gdn(x)
        return o



class GatedDeltaNet2SimpAttention(nn.Module):
    """
    Wrapper around FLA's GatedDeltaNet2simp that conforms to core's
    Attention interface: forward(x) -> Tensor.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        fla_config: Optional[FLAConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        assert FLAGatedDeltaNet2Simp is not None, (
            "flash-linear-attention is required for GatedDeltaNet2Simp. "
            "Install with: pip install -e third_party/flash-linear-attention"
        )
        cfg = fla_config or FLAConfig(expand_v=2.0, use_short_conv=True)
        self.d_model = d_model
        self.use_rope = False

        self.gdn = FLAGatedDeltaNet2Simp(
            mode=cfg.mode,
            hidden_size=d_model,
            num_heads=n_heads,
            num_v_heads=cfg.num_v_heads,
            head_dim=head_dim or d_model // n_heads,
            expand_v=cfg.expand_v,
            use_gate=cfg.use_gate,
            use_short_conv=cfg.use_short_conv,
            conv_size=cfg.conv_size,
            conv_bias=cfg.conv_bias,
            allow_neg_eigval=cfg.allow_neg_eigval,
            norm_eps=cfg.norm_eps,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        o, _, _ = self.gdn(x)
        return o
