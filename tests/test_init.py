"""Tests for weight initialization (core.modules.init).

Covers:
- init_feed_forward dispatch for all four feed-forward types (MLP, GLU,
  NormalizedMLP, NormalizedGLU) across all three InitMethod variants.
- Verifies that MLP now exposes d_model (required by the NORMALIZED path).
- Verifies that init_attention works for NORMAL, LLAMA, and NORMALIZED.
"""

import pytest
import torch
import torch.nn as nn

from core.modules.init import InitMethod
from core.modules.attention import AttentionConfig, AttentionType
from core.modules.feed_forward import (
    FeedForward,
    FeedForwardConfig,
    FeedForwardType,
    MLP,
    GLU,
    NormalizedMLP,
    NormalizedGLU,
    ActivationType,
)
from core.utils import BufferCache

D_MODEL = 32
HIDDEN_SIZE = 64
N_HEADS = 4
N_LAYERS = 2


def _build_ff(ff_type: FeedForwardType) -> FeedForward:
    return FeedForward.build(
        type=ff_type,
        d_model=D_MODEL,
        hidden_size=HIDDEN_SIZE,
        activation_type=ActivationType.GELU,
    )


# ---------------------------------------------------------------------------
# init_feed_forward: dispatch and correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ff_type", list(FeedForwardType))
@pytest.mark.parametrize("init_method", list(InitMethod))
def test_init_feed_forward_runs_without_error(ff_type: FeedForwardType, init_method: InitMethod):
    """init_feed_forward must not raise for any (ff_type, init_method) pair."""
    module = _build_ff(ff_type)
    gen = torch.Generator().manual_seed(0)
    init_method.init_feed_forward(module, D_MODEL, N_LAYERS, generator=gen)


@pytest.mark.parametrize("ff_type", list(FeedForwardType))
def test_init_feed_forward_modifies_weights(ff_type: FeedForwardType):
    """init_feed_forward must actually change the parameter values."""
    module = _build_ff(ff_type)

    before = {name: p.clone() for name, p in module.named_parameters() if "weight" in name}

    gen = torch.Generator().manual_seed(0)
    InitMethod.NORMAL.init_feed_forward(module, D_MODEL, N_LAYERS, generator=gen)

    changed_any = False
    for name, p in module.named_parameters():
        if "weight" in name and name in before:
            if not torch.equal(p, before[name]):
                changed_any = True
                break

    assert changed_any, f"init_feed_forward did not modify any weight for {ff_type}"


def test_mlp_has_d_model_attribute():
    """MLP must store d_model so that InitMethod.NORMALIZED can use it."""
    mlp = MLP(d_model=D_MODEL, hidden_size=HIDDEN_SIZE, activation_type=ActivationType.GELU)
    assert hasattr(mlp, "d_model")
    assert mlp.d_model == D_MODEL


def test_normalized_mlp_inherits_d_model():
    """NormalizedMLP (inherits MLP) must also expose d_model."""
    nmlp = NormalizedMLP(d_model=D_MODEL, hidden_size=HIDDEN_SIZE, activation_type=ActivationType.GELU)
    assert hasattr(nmlp, "d_model")
    assert nmlp.d_model == D_MODEL


def test_glu_has_d_model_attribute():
    glu = GLU(d_model=D_MODEL, hidden_size=HIDDEN_SIZE, activation_type=ActivationType.GELU)
    assert glu.d_model == D_MODEL


def test_isinstance_dispatch_covers_normalized_variants():
    """NormalizedMLP isinstance MLP and NormalizedGLU isinstance GLU."""
    nmlp = _build_ff(FeedForwardType.NORMALIZED_MLP)
    nglu = _build_ff(FeedForwardType.NORMALIZED_GLU)

    assert isinstance(nmlp, MLP)
    assert isinstance(nglu, GLU)


# ---------------------------------------------------------------------------
# init_feed_forward: NORMALIZED std scaling
# ---------------------------------------------------------------------------

def test_normalized_init_uses_d_model_scaling_mlp():
    """Under InitMethod.NORMALIZED the initial std should be d_model**-0.5."""
    module = _build_ff(FeedForwardType.MLP)

    gen = torch.Generator().manual_seed(42)
    InitMethod.NORMALIZED.init_feed_forward(module, D_MODEL, N_LAYERS, generator=gen)

    expected_std = D_MODEL ** -0.5
    w1_std = module.w1.weight.std().item()
    assert abs(w1_std - expected_std) < expected_std, (
        f"w1 std ({w1_std:.4f}) should be close to d_model**-0.5 ({expected_std:.4f})"
    )


def test_normalized_init_uses_d_model_scaling_glu():
    module = _build_ff(FeedForwardType.GLU)

    gen = torch.Generator().manual_seed(42)
    InitMethod.NORMALIZED.init_feed_forward(module, D_MODEL, N_LAYERS, generator=gen)

    expected_std = D_MODEL ** -0.5
    w1_std = module.w1.weight.std().item()
    assert abs(w1_std - expected_std) < expected_std, (
        f"w1 std ({w1_std:.4f}) should be close to d_model**-0.5 ({expected_std:.4f})"
    )


# ---------------------------------------------------------------------------
# init_attention: smoke test across init methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("init_method", list(InitMethod))
def test_init_attention_runs_without_error(init_method: InitMethod):
    cache = BufferCache()
    att_config = AttentionConfig(n_heads=N_HEADS)
    att = att_config.build(d_model=D_MODEL, cache=cache)

    gen = torch.Generator().manual_seed(0)
    init_method.init_attention(att, N_LAYERS, generator=gen)


# ---------------------------------------------------------------------------
# Full model build: ensures init_weights completes for all recipe families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ff_type,init_method",
    [
        (FeedForwardType.MLP, InitMethod.NORMAL),
        (FeedForwardType.GLU, InitMethod.LLAMA),
        (FeedForwardType.NORMALIZED_MLP, InitMethod.NORMALIZED),
        (FeedForwardType.NORMALIZED_GLU, InitMethod.NORMALIZED),
    ],
)
def test_full_model_init_weights(ff_type, init_method):
    """CoreConfig.build() -> model.init_weights() must succeed for representative combos."""
    from core.models.model_config import CoreConfig, CoreType
    from core.modules.layer_norm import LayerNormConfig

    is_normalized = init_method == InitMethod.NORMALIZED
    config = CoreConfig(
        transformer_type=CoreType.NORMALIZED if is_normalized else CoreType.BASE,
        n_layers=2,
        d_model=D_MODEL,
        vocab_size=100,
        attention=AttentionConfig(
            n_heads=N_HEADS,
            type=AttentionType.NORMALIZED if is_normalized else AttentionType.DEFAULT,
        ),
        feed_forward=FeedForwardConfig(
            ff_hidden_size=HIDDEN_SIZE,
            feed_forward_type=ff_type,
        ),
        layer_norm=LayerNormConfig(),
        max_sequence_length=64,
        dropout=0.0,
        init_method=init_method,
    )

    model = config.build()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    with torch.no_grad():
        output = model(input_ids=input_ids)
    assert output.logits.shape == (batch_size, seq_len, 100)
