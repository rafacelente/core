from typing import Optional

from core.models.model_config import CoreConfig, CoreType
from core.modules.attention import AttentionConfig
from core.modules.feed_forward import FeedForwardConfig
from core.modules.layer_norm import LayerNormConfig
from core.modules.linear_attention import FLAConfig
from core.models.model_recipes.model_recipe import ModelRecipe
from core.modules.loss import LossConfig
from core.modules.rope import RoPEConfig


# ---------------------------------------------------------------------------
# GPT family
# ---------------------------------------------------------------------------

class GPTRecipe(ModelRecipe):
    """
    Base recipe for GPT-style models.

    GPT models use:
    - Standard layer normalization
    - MLP feed-forward with GELU activation
    - Rotary position embeddings (RoPE)
    - ff_ratio-based hidden size scaling
    """

    n_layers: int
    d_model: int
    n_heads: int
    ff_ratio: int = 4
    n_kv_heads: Optional[int] = None

    def build_config(
        self,
        vocab_size: int,
        max_sequence_length: int = 2048,
        dropout: float = 0.0,
        transformer_type: str = "base",
        use_post_sdpa_gate: bool = False,
        gate_activation_type: str = "sigmoid",
        pad_token_id: int = 50256,
        rope_theta: int = 10_000,
    ) -> CoreConfig:
        return CoreConfig(
            transformer_type=CoreType(transformer_type),
            n_layers=self.n_layers,
            d_model=self.d_model,
            attention=AttentionConfig(
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                dropout=dropout,
                rope=RoPEConfig(type="default", theta=rope_theta),
                qk_norm=None,
                clip_qkv=None,
                use_post_sdpa_gate=use_post_sdpa_gate,
                gate_activation_type=gate_activation_type,
            ),
            output_norm=LayerNormConfig(layer_norm_type="default", eps=1e-5),
            feed_forward=FeedForwardConfig(
                feed_forward_type="mlp",
                ff_ratio=self.ff_ratio,
                activation_type="gelu",
            ),
            layer_norm=LayerNormConfig(layer_norm_type="default", eps=1e-5),
            loss=LossConfig(type="cross_entropy", ignore_index=pad_token_id),
            vocab_size=vocab_size,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
        )


@ModelRecipe.register("gpt-tiny")
class GPTTinyRecipe(GPTRecipe):
    n_layers = 10
    d_model = 576
    n_heads = 12


@ModelRecipe.register("gpt-small")
class GPTSmallRecipe(GPTRecipe):
    n_layers = 12
    d_model = 768
    n_heads = 12


@ModelRecipe.register("gpt-medium")
class GPTMediumRecipe(GPTRecipe):
    n_layers = 24
    d_model = 1024
    n_heads = 16


@ModelRecipe.register("gpt-large")
class GPTLargeRecipe(GPTRecipe):
    n_layers = 24
    d_model = 1536
    n_heads = 16


@ModelRecipe.register("gpt-xl")
class GPTXLRecipe(GPTRecipe):
    n_layers = 48
    d_model = 1600
    n_heads = 25


@ModelRecipe.register("gpt-2.7b")
class GPT27BRecipe(GPTRecipe):
    n_layers = 32
    d_model = 2560
    n_heads = 32


# ---------------------------------------------------------------------------
# LLaMA family
# ---------------------------------------------------------------------------

class LLaMARecipe(ModelRecipe):
    """
    Base recipe for LLaMA-style models.

    LLaMA models use:
    - RMS layer normalization
    - GLU feed-forward with SiLU activation
    - Rotary position embeddings (RoPE)
    - Explicit ff_hidden_size (not ratio-based)
    """

    n_layers: int
    d_model: int
    n_heads: int
    ff_hidden_size: int
    n_kv_heads: Optional[int] = None

    def build_config(
        self,
        vocab_size: int,
        max_sequence_length: int = 2048,
        dropout: float = 0.0,
        transformer_type: str = "base",
        use_post_sdpa_gate: bool = False,
        gate_activation_type: str = "sigmoid",
        pad_token_id: int = 50256,
        rope_theta: int = 10_000,
    ) -> CoreConfig:
        return CoreConfig(
            transformer_type=CoreType(transformer_type),
            n_layers=self.n_layers,
            d_model=self.d_model,
            attention=AttentionConfig(
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                dropout=dropout,
                rope=RoPEConfig(type="default", theta=rope_theta),
                qk_norm=None,
                clip_qkv=None,
                use_post_sdpa_gate=use_post_sdpa_gate,
                gate_activation_type=gate_activation_type,
            ),
            output_norm=LayerNormConfig(layer_norm_type="rms", eps=1e-5),
            feed_forward=FeedForwardConfig(
                feed_forward_type="glu",
                ff_hidden_size=self.ff_hidden_size,
                activation_type="silu",
            ),
            layer_norm=LayerNormConfig(layer_norm_type="rms", eps=1e-5),
            loss=LossConfig(type="cross_entropy", ignore_index=pad_token_id),
            vocab_size=vocab_size,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
        )


@ModelRecipe.register("llama-small")
class LLaMASmallRecipe(LLaMARecipe):
    n_layers = 12
    d_model = 768
    n_heads = 12
    ff_hidden_size = 2304


@ModelRecipe.register("llama-medium")
class LLaMAMediumRecipe(LLaMARecipe):
    n_layers = 24
    d_model = 1024
    n_heads = 16
    ff_hidden_size = 2816


@ModelRecipe.register("llama-large")
class LLaMALargeRecipe(LLaMARecipe):
    n_layers = 36
    d_model = 1280
    n_heads = 20
    ff_hidden_size = 3072

class GLARecipe(ModelRecipe):
    """
    Base recipe for GLA-style models.

    Uses the same feed-forward and normalization as LLaMA
    (RMS norm, GLU + SiLU) but replaces softmax attention with
    Gated Linear Attention from the FLA library.
    """

    n_layers: int
    d_model: int
    n_heads: int
    ff_hidden_size: int
    n_kv_heads: Optional[int] = None
    expand_k: float = 0.5
    expand_v: float = 1.0

    def build_config(
        self,
        vocab_size: int,
        max_sequence_length: int = 2048,
        dropout: float = 0.0,
        transformer_type: str = "base",
        use_post_sdpa_gate: bool = False,
        gate_activation_type: str = "sigmoid",
        pad_token_id: int = 50256,
    ) -> CoreConfig:
        return CoreConfig(
            transformer_type=CoreType(transformer_type),
            n_layers=self.n_layers,
            d_model=self.d_model,
            attention=AttentionConfig(
                type="gla",
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                rope=None,
                fla=FLAConfig(
                    expand_k=self.expand_k,
                    expand_v=self.expand_v,
                ),
            ),
            output_norm=LayerNormConfig(layer_norm_type="rms", eps=1e-5),
            feed_forward=FeedForwardConfig(
                feed_forward_type="glu",
                ff_hidden_size=self.ff_hidden_size,
                activation_type="silu",
            ),
            layer_norm=LayerNormConfig(layer_norm_type="rms", eps=1e-5),
            loss=LossConfig(type="cross_entropy", ignore_index=pad_token_id),
            vocab_size=vocab_size,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
        )


@ModelRecipe.register("gla-small")
class GLASmallRecipe(GLARecipe):
    n_layers = 12
    d_model = 768
    n_heads = 12
    ff_hidden_size = 2304


@ModelRecipe.register("gla-medium")
class GLAMediumRecipe(GLARecipe):
    n_layers = 24
    d_model = 1024
    n_heads = 16
    ff_hidden_size = 2816


@ModelRecipe.register("gla-large")
class GLALargeRecipe(GLARecipe):
    n_layers = 36
    d_model = 1280
    n_heads = 20
    ff_hidden_size = 3072

class GDNRecipe(ModelRecipe):
    """
    Base recipe for Gated Delta Net models.

    Uses the same feed-forward and normalization as LLaMA
    (RMS norm, GLU + SiLU) but replaces softmax attention with
    Gated Delta Net from the FLA library.

    Per the GDN paper, key_dim = num_heads * head_dim should be
    ~0.75 * d_model, with expand_v=2 and use_short_conv=True.
    Total attention params per layer ≈ 6 * d_model².
    """

    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int
    ff_hidden_size: int
    expand_v: float = 2.0

    def build_config(
        self,
        vocab_size: int,
        max_sequence_length: int = 2048,
        dropout: float = 0.0,
        transformer_type: str = "base",
        use_post_sdpa_gate: bool = False,
        gate_activation_type: str = "sigmoid",
        pad_token_id: int = 50256,
    ) -> CoreConfig:
        return CoreConfig(
            transformer_type=CoreType(transformer_type),
            n_layers=self.n_layers,
            d_model=self.d_model,
            attention=AttentionConfig(
                type="gated_delta_net",
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                rope=None,
                fla=FLAConfig(
                    expand_v=self.expand_v,
                    use_short_conv=True,
                ),
            ),
            output_norm=LayerNormConfig(layer_norm_type="rms", eps=1e-5),
            feed_forward=FeedForwardConfig(
                feed_forward_type="glu",
                ff_hidden_size=self.ff_hidden_size,
                activation_type="silu",
            ),
            layer_norm=LayerNormConfig(layer_norm_type="rms", eps=1e-5),
            loss=LossConfig(type="cross_entropy", ignore_index=pad_token_id),
            vocab_size=vocab_size,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
        )


@ModelRecipe.register("gdn-small")
class GDNSmallRecipe(GDNRecipe):
    n_layers = 12
    d_model = 768
    n_heads = 6
    head_dim = 96
    ff_hidden_size = 2304


@ModelRecipe.register("gdn-medium")
class GDNMediumRecipe(GDNRecipe):
    n_layers = 24
    d_model = 1024
    n_heads = 6
    head_dim = 128
    ff_hidden_size = 2816


@ModelRecipe.register("gdn-large")
class GDNLargeRecipe(GDNRecipe):
    n_layers = 36
    d_model = 1280
    n_heads = 10
    head_dim = 96
    ff_hidden_size = 3072


# ---------------------------------------------------------------------------
# 2Simp-GDN (GatedDeltaNet2Simp) family
# ---------------------------------------------------------------------------

class GDN2SimpRecipe(ModelRecipe):
    """
    Base recipe for GatedDeltaNet2Simp models.

    Same feed-forward and normalization as LLaMA (RMS norm, GLU + SiLU)
    with the GatedDeltaNet2Simp attention variant.

    Follows the same dimension conventions as GDN:
    key_dim = num_heads * head_dim ≈ 0.75 * d_model.
    """

    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int
    ff_hidden_size: int
    expand_v: float = 2.0

    def build_config(
        self,
        vocab_size: int,
        max_sequence_length: int = 2048,
        dropout: float = 0.0,
        transformer_type: str = "base",
        use_post_sdpa_gate: bool = False,
        gate_activation_type: str = "sigmoid",
        pad_token_id: int = 50256,
    ) -> CoreConfig:
        return CoreConfig(
            transformer_type=CoreType(transformer_type),
            n_layers=self.n_layers,
            d_model=self.d_model,
            attention=AttentionConfig(
                type="gated_delta_net_2simp",
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                rope=None,
                fla=FLAConfig(
                    expand_v=self.expand_v,
                    use_short_conv=True,
                ),
            ),
            output_norm=LayerNormConfig(layer_norm_type="rms", eps=1e-5),
            feed_forward=FeedForwardConfig(
                feed_forward_type="glu",
                ff_hidden_size=self.ff_hidden_size,
                activation_type="silu",
            ),
            layer_norm=LayerNormConfig(layer_norm_type="rms", eps=1e-5),
            loss=LossConfig(type="cross_entropy", ignore_index=pad_token_id),
            vocab_size=vocab_size,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
        )


@ModelRecipe.register("2simp-gdn-small")
class GDN2SimpSmallRecipe(GDN2SimpRecipe):
    n_layers = 12
    d_model = 768
    n_heads = 6
    head_dim = 96
    ff_hidden_size = 2304


@ModelRecipe.register("2simp-gdn-medium")
class GDN2SimpMediumRecipe(GDN2SimpRecipe):
    n_layers = 24
    d_model = 1024
    n_heads = 6
    head_dim = 128
    ff_hidden_size = 2816


@ModelRecipe.register("2simp-gdn-large")
class GDN2SimpLargeRecipe(GDN2SimpRecipe):
    n_layers = 36
    d_model = 1280
    n_heads = 10
    head_dim = 96
    ff_hidden_size = 3072
