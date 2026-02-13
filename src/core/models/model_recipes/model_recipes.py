from typing import Optional

from core.models.model_config import CoreConfig, CoreType
from core.modules.attention import AttentionConfig
from core.modules.feed_forward import FeedForwardConfig
from core.modules.layer_norm import LayerNormConfig
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
