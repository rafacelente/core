"""Tests that combinable features (XSA, gate, Noble, normalized, GQA, qk-norm,
clip-qkv) compose correctly or raise clear errors when incompatible.

Each "works" test verifies:
  1. The model builds without error.
  2. A forward pass produces the expected output shape.
  3. A backward pass runs without error (gradients flow).
"""

import itertools

import pytest
import torch

from core.models.model_config import CoreConfig, CoreType
from core.models.model import CoreModel, NormalizedCoreModel
from core.models.model_utils import CoreOutput
from core.modules.attention import AttentionConfig
from core.modules.feed_forward import FeedForwardConfig, FeedForwardType
from core.modules.layer_norm import LayerNormConfig
from core.modules.linear import LinearConfig, LinearType
from core.modules.noble import NobleConfig, LinearNoble
from core.modules.activation import ActivationType
from core.modules.rope import RoPEConfig

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
VOCAB = 128
SEQ = 16
BATCH = 2

NOBLE_LINEAR = LinearConfig(
    type=LinearType.NOBLE,
    noble=NobleConfig(r=8, activation_type=ActivationType.COSNET, apply_to=["all"]),
)

QK_NORM = LayerNormConfig(layer_norm_type="default")


def _build_config(
    *,
    xsa: bool = False,
    gate: bool = False,
    noble: bool = False,
    normalized: bool = False,
    ff_type: FeedForwardType = FeedForwardType.GLU,
    gqa: bool = False,
    qk_norm: bool = False,
    clip_qkv: float | None = None,
) -> CoreConfig:
    n_kv_heads = 2 if gqa else None
    return CoreConfig(
        transformer_type=CoreType.NORMALIZED if normalized else CoreType.BASE,
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        attention=AttentionConfig(
            n_heads=N_HEADS,
            n_kv_heads=n_kv_heads,
            rope=RoPEConfig(type="default"),
            use_xsa=xsa,
            use_post_sdpa_gate=gate,
            gate_activation_type=ActivationType.SIGMOID,
            qk_norm=QK_NORM if qk_norm else None,
            clip_qkv=clip_qkv,
        ),
        feed_forward=FeedForwardConfig(
            feed_forward_type=ff_type,
            ff_hidden_size=D_MODEL * 2,
            activation_type=ActivationType.SILU,
        ),
        linear=NOBLE_LINEAR if noble else LinearConfig(),
        layer_norm=LayerNormConfig(layer_norm_type="rms"),
        output_norm=LayerNormConfig(layer_norm_type="rms"),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        init_method="llama" if not normalized else "normalized",
    )


def _forward_backward(config: CoreConfig):
    model = config.build()
    model.train()
    ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    labels = torch.randint(0, VOCAB, (BATCH, SEQ))
    out = model(ids, labels=labels)
    assert isinstance(out, CoreOutput)
    assert out.logits.shape == (BATCH, SEQ, VOCAB)
    assert out.loss is not None
    out.loss.backward()
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0, "No gradients computed"
    return model, out


# ---------------------------------------------------------------------------
# Pairwise combos of {xsa, gate, noble} on BASE transformer
# ---------------------------------------------------------------------------

class TestPairwiseBase:
    def test_xsa_only(self):
        _forward_backward(_build_config(xsa=True))

    def test_gate_only(self):
        _forward_backward(_build_config(gate=True))

    def test_noble_only(self):
        _forward_backward(_build_config(noble=True))

    def test_xsa_gate(self):
        _forward_backward(_build_config(xsa=True, gate=True))

    def test_xsa_noble(self):
        _forward_backward(_build_config(xsa=True, noble=True))

    def test_gate_noble(self):
        _forward_backward(_build_config(gate=True, noble=True))

    def test_xsa_gate_noble(self):
        _forward_backward(_build_config(xsa=True, gate=True, noble=True))


# ---------------------------------------------------------------------------
# Pairwise combos of {xsa, noble} on NORMALIZED transformer
# (gate + normalized is invalid, tested separately)
# ---------------------------------------------------------------------------

class TestPairwiseNormalized:
    def test_normalized_baseline(self):
        _forward_backward(_build_config(normalized=True))

    def test_normalized_xsa(self):
        _forward_backward(_build_config(normalized=True, xsa=True))

    def test_normalized_noble(self):
        model, _ = _forward_backward(_build_config(normalized=True, noble=True))
        assert isinstance(model.blocks["0"].attention.w_q, LinearNoble)

    def test_normalized_xsa_noble(self):
        _forward_backward(_build_config(normalized=True, xsa=True, noble=True))


# ---------------------------------------------------------------------------
# Gate + Normalized must raise
# ---------------------------------------------------------------------------

class TestIncompatibleCombos:
    def test_gate_normalized_raises(self):
        with pytest.raises(ValueError, match="use_post_sdpa_gate.*not compatible.*normalized"):
            _build_config(normalized=True, gate=True)

    def test_gate_normalized_xsa_raises(self):
        with pytest.raises(ValueError, match="use_post_sdpa_gate.*not compatible.*normalized"):
            _build_config(normalized=True, gate=True, xsa=True)

    def test_gate_normalized_noble_raises(self):
        with pytest.raises(ValueError, match="use_post_sdpa_gate.*not compatible.*normalized"):
            _build_config(normalized=True, gate=True, noble=True)


# ---------------------------------------------------------------------------
# GQA interacts with everything
# ---------------------------------------------------------------------------

class TestGQACombos:
    def test_gqa_base(self):
        _forward_backward(_build_config(gqa=True))

    def test_gqa_xsa(self):
        _forward_backward(_build_config(gqa=True, xsa=True))

    def test_gqa_noble(self):
        model, _ = _forward_backward(_build_config(gqa=True, noble=True))
        att = model.blocks["0"].attention
        assert att.w_q.weight.shape[0] != att.w_k.weight.shape[0]

    def test_gqa_gate(self):
        _forward_backward(_build_config(gqa=True, gate=True))

    def test_gqa_normalized(self):
        _forward_backward(_build_config(gqa=True, normalized=True))

    def test_gqa_xsa_noble(self):
        _forward_backward(_build_config(gqa=True, xsa=True, noble=True))

    def test_gqa_normalized_noble(self):
        _forward_backward(_build_config(gqa=True, normalized=True, noble=True))


# ---------------------------------------------------------------------------
# QK-norm combos
# ---------------------------------------------------------------------------

class TestQKNormCombos:
    def test_qk_norm_base(self):
        _forward_backward(_build_config(qk_norm=True))

    def test_qk_norm_xsa(self):
        _forward_backward(_build_config(qk_norm=True, xsa=True))

    def test_qk_norm_noble(self):
        _forward_backward(_build_config(qk_norm=True, noble=True))

    def test_qk_norm_normalized(self):
        _forward_backward(_build_config(qk_norm=True, normalized=True))

    def test_qk_norm_gqa_xsa_noble(self):
        _forward_backward(_build_config(qk_norm=True, gqa=True, xsa=True, noble=True))


# ---------------------------------------------------------------------------
# Clip QKV combos
# ---------------------------------------------------------------------------

class TestClipQKVCombos:
    def test_clip_base(self):
        _forward_backward(_build_config(clip_qkv=1.0))

    def test_clip_xsa(self):
        _forward_backward(_build_config(clip_qkv=1.0, xsa=True))

    def test_clip_noble(self):
        _forward_backward(_build_config(clip_qkv=1.0, noble=True))

    def test_clip_xsa_noble_gate(self):
        _forward_backward(_build_config(clip_qkv=1.0, xsa=True, noble=True, gate=True))


# ---------------------------------------------------------------------------
# Feed-forward type interactions
# ---------------------------------------------------------------------------

class TestFFTypeCombos:
    @pytest.mark.parametrize("ff_type", [FeedForwardType.MLP, FeedForwardType.GLU])
    def test_base_noble(self, ff_type):
        _forward_backward(_build_config(noble=True, ff_type=ff_type))

    @pytest.mark.parametrize("ff_type", [FeedForwardType.MLP, FeedForwardType.GLU])
    def test_normalized_noble(self, ff_type):
        _forward_backward(_build_config(noble=True, normalized=True, ff_type=ff_type))

    @pytest.mark.parametrize("ff_type", [FeedForwardType.MLP, FeedForwardType.GLU])
    def test_xsa_noble(self, ff_type):
        _forward_backward(_build_config(xsa=True, noble=True, ff_type=ff_type))


# ---------------------------------------------------------------------------
# Noble apply_to scoping + feature combos
# ---------------------------------------------------------------------------

class TestNobleApplyToWithFeatures:
    def _noble_cfg(self, apply_to):
        return LinearConfig(
            type=LinearType.NOBLE,
            noble=NobleConfig(r=8, activation_type=ActivationType.COSNET, apply_to=apply_to),
        )

    def test_noble_att_only_with_xsa(self):
        cfg = CoreConfig(
            transformer_type=CoreType.BASE,
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            attention=AttentionConfig(
                n_heads=N_HEADS,
                rope=RoPEConfig(type="default"),
                use_xsa=True,
            ),
            feed_forward=FeedForwardConfig(
                feed_forward_type=FeedForwardType.GLU,
                ff_hidden_size=D_MODEL * 2,
                activation_type=ActivationType.SILU,
            ),
            linear=self._noble_cfg(["att"]),
            layer_norm=LayerNormConfig(layer_norm_type="rms"),
            output_norm=LayerNormConfig(layer_norm_type="rms"),
            vocab_size=VOCAB,
            max_sequence_length=SEQ,
        )
        model = cfg.build()
        assert isinstance(model.blocks["0"].attention.w_q, LinearNoble)
        assert not isinstance(model.blocks["0"].feed_forward.w1, LinearNoble)
        _forward_backward(cfg)

    def test_noble_ff_only_with_gate(self):
        cfg = CoreConfig(
            transformer_type=CoreType.BASE,
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            attention=AttentionConfig(
                n_heads=N_HEADS,
                rope=RoPEConfig(type="default"),
                use_post_sdpa_gate=True,
            ),
            feed_forward=FeedForwardConfig(
                feed_forward_type=FeedForwardType.GLU,
                ff_hidden_size=D_MODEL * 2,
                activation_type=ActivationType.SILU,
            ),
            linear=self._noble_cfg(["ff"]),
            layer_norm=LayerNormConfig(layer_norm_type="rms"),
            output_norm=LayerNormConfig(layer_norm_type="rms"),
            vocab_size=VOCAB,
            max_sequence_length=SEQ,
        )
        model = cfg.build()
        assert not isinstance(model.blocks["0"].attention.w_q, LinearNoble)
        assert isinstance(model.blocks["0"].feed_forward.w1, LinearNoble)
        _forward_backward(cfg)

    def test_noble_lm_head_only(self):
        cfg = CoreConfig(
            transformer_type=CoreType.BASE,
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            attention=AttentionConfig(
                n_heads=N_HEADS,
                rope=RoPEConfig(type="default"),
            ),
            feed_forward=FeedForwardConfig(
                feed_forward_type=FeedForwardType.GLU,
                ff_hidden_size=D_MODEL * 2,
                activation_type=ActivationType.SILU,
            ),
            linear=self._noble_cfg(["lm_head"]),
            layer_norm=LayerNormConfig(layer_norm_type="rms"),
            output_norm=LayerNormConfig(layer_norm_type="rms"),
            vocab_size=VOCAB,
            max_sequence_length=SEQ,
        )
        model = cfg.build()
        assert not isinstance(model.blocks["0"].attention.w_q, LinearNoble)
        assert not isinstance(model.blocks["0"].feed_forward.w1, LinearNoble)
        assert isinstance(model.lm_head, LinearNoble)
        _forward_backward(cfg)


# ---------------------------------------------------------------------------
# Normalized + Noble: normalize_matrices must not crash
# ---------------------------------------------------------------------------

class TestNormalizedNobleMatrixNorm:
    def test_normalize_matrices_with_noble(self):
        config = _build_config(normalized=True, noble=True)
        model = config.build()
        assert isinstance(model, NormalizedCoreModel)
        model.normalize_matrices()
        model.post_optim_step()
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)
        assert not torch.isnan(out.logits).any()

    def test_normalize_matrices_with_noble_gqa(self):
        config = _build_config(normalized=True, noble=True, gqa=True)
        model = config.build()
        model.normalize_matrices()
        model.post_optim_step()
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        with torch.no_grad():
            out = model(ids)
        assert not torch.isnan(out.logits).any()


# ---------------------------------------------------------------------------
# Full kitchen-sink combos (base only, since gate is invalid with normalized)
# ---------------------------------------------------------------------------

class TestKitchenSink:
    def test_all_base_features(self):
        """XSA + gate + noble + GQA + qk_norm + clip on base GLU."""
        _forward_backward(_build_config(
            xsa=True, gate=True, noble=True, gqa=True,
            qk_norm=True, clip_qkv=1.0,
        ))

    def test_all_normalized_features(self):
        """XSA + noble + GQA + qk_norm on normalized GLU (no gate)."""
        _forward_backward(_build_config(
            normalized=True, xsa=True, noble=True, gqa=True,
            qk_norm=True,
        ))

    def test_all_base_mlp(self):
        """All features on base MLP."""
        _forward_backward(_build_config(
            xsa=True, gate=True, noble=True, gqa=True,
            qk_norm=True, clip_qkv=1.0, ff_type=FeedForwardType.MLP,
        ))
