"""Tests for the HF conversion layer (core.hf) and convert_to_hf script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from core.models.model_config import CoreConfig, CoreType
from core.modules.attention import AttentionConfig
from core.modules.feed_forward import FeedForwardConfig
from core.modules.layer_norm import LayerNormConfig
from core.modules.rope import RoPEConfig

from core.hf.configuration_core import CoreModelConfig
from core.hf.modeling_core import CoreModelForCausalLM


# ------------------------------------------------------------------
# Helpers: minimal CoreConfig dicts for tests
# ------------------------------------------------------------------

def _base_config_dict() -> dict:
    """Vanilla base transformer config (GPT-style, small)."""
    return {
        "transformer_type": "base",
        "n_layers": 2,
        "d_model": 64,
        "attention": {"n_heads": 4, "rope": {"type": "default"}},
        "feed_forward": {
            "feed_forward_type": "mlp",
            "ff_hidden_size": 128,
            "activation_type": "gelu",
        },
        "layer_norm": {"layer_norm_type": "default"},
        "output_norm": {"layer_norm_type": "default"},
        "vocab_size": 256,
        "max_sequence_length": 64,
    }


def _llama_config_dict() -> dict:
    """LLaMA-style config with GLU + RMSNorm."""
    return {
        "transformer_type": "base",
        "n_layers": 2,
        "d_model": 64,
        "attention": {"n_heads": 4, "rope": {"type": "default"}},
        "feed_forward": {
            "feed_forward_type": "glu",
            "ff_hidden_size": 128,
            "activation_type": "silu",
        },
        "layer_norm": {"layer_norm_type": "rms"},
        "output_norm": {"layer_norm_type": "rms"},
        "vocab_size": 256,
        "max_sequence_length": 64,
        "init_method": "llama",
    }


def _noble_config_dict() -> dict:
    """LLaMA-style config with Noble linear layers."""
    cfg = _llama_config_dict()
    cfg["linear"] = {
        "type": "noble",
        "noble": {"r": 8, "activation_type": "cosnet", "apply_to": ["all"]},
    }
    return cfg


def _xsa_config_dict() -> dict:
    """LLaMA-style config with XSA enabled."""
    cfg = _llama_config_dict()
    cfg["attention"]["use_xsa"] = True
    return cfg


def _normalized_config_dict() -> dict:
    """Normalized (nGPT-style) transformer config."""
    return {
        "transformer_type": "normalized",
        "n_layers": 2,
        "d_model": 64,
        "attention": {"n_heads": 4, "rope": {"type": "default"}},
        "feed_forward": {
            "feed_forward_type": "glu",
            "ff_hidden_size": 128,
            "activation_type": "silu",
        },
        "layer_norm": {"layer_norm_type": "rms"},
        "output_norm": {"layer_norm_type": "rms"},
        "vocab_size": 256,
        "max_sequence_length": 64,
        "init_method": "normalized",
    }


# ==================================================================
# 1. Config round-trip
# ==================================================================


class TestConfigRoundTrip:
    """CoreConfig -> CoreModelConfig -> JSON -> reload -> CoreConfig."""

    def test_round_trip_base(self):
        cfg_dict = _base_config_dict()
        hf_cfg = CoreModelConfig(core_config=cfg_dict)

        with tempfile.TemporaryDirectory() as td:
            hf_cfg.save_pretrained(td)
            loaded = CoreModelConfig.from_pretrained(td)

        assert loaded.core_config == cfg_dict
        assert loaded.vocab_size == 256
        assert loaded.hidden_size == 64
        assert loaded.num_hidden_layers == 2
        assert loaded.num_attention_heads == 4

    def test_round_trip_noble(self):
        cfg_dict = _noble_config_dict()
        hf_cfg = CoreModelConfig(core_config=cfg_dict)

        with tempfile.TemporaryDirectory() as td:
            hf_cfg.save_pretrained(td)
            loaded = CoreModelConfig.from_pretrained(td)

        assert loaded.core_config["linear"]["type"] == "noble"
        assert loaded.core_config["linear"]["noble"]["r"] == 8

    def test_round_trip_normalized(self):
        cfg_dict = _normalized_config_dict()
        hf_cfg = CoreModelConfig(core_config=cfg_dict)

        with tempfile.TemporaryDirectory() as td:
            hf_cfg.save_pretrained(td)
            loaded = CoreModelConfig.from_pretrained(td)

        assert loaded.core_config["transformer_type"] == "normalized"

    def test_to_core_config_builds(self):
        hf_cfg = CoreModelConfig(core_config=_base_config_dict())
        core_cfg = hf_cfg.to_core_config()
        assert isinstance(core_cfg, CoreConfig)
        assert core_cfg.n_layers == 2
        assert core_cfg.d_model == 64

    def test_config_json_contains_core_config(self):
        cfg_dict = _base_config_dict()
        hf_cfg = CoreModelConfig(core_config=cfg_dict)

        with tempfile.TemporaryDirectory() as td:
            hf_cfg.save_pretrained(td)
            with open(Path(td) / "config.json") as f:
                raw = json.load(f)

        assert "core_config" in raw
        assert raw["core_config"]["n_layers"] == 2
        assert raw["model_type"] == "core"


# ==================================================================
# 2. Forward-pass equivalence
# ==================================================================


class TestForwardEquivalence:
    """CoreModelForCausalLM must produce the same logits as raw CoreModel."""

    @pytest.fixture(params=["base", "llama", "noble", "xsa", "normalized"])
    def variant(self, request):
        return request.param

    def _config_dict_for(self, variant: str) -> dict:
        return {
            "base": _base_config_dict,
            "llama": _llama_config_dict,
            "noble": _noble_config_dict,
            "xsa": _xsa_config_dict,
            "normalized": _normalized_config_dict,
        }[variant]()

    def test_logits_match(self, variant):
        cfg_dict = self._config_dict_for(variant)

        core_cfg = CoreConfig(**cfg_dict)
        core_model = core_cfg.build()
        core_model.eval()

        hf_cfg = CoreModelConfig(core_config=cfg_dict)
        hf_model = CoreModelForCausalLM(hf_cfg)

        hf_model.core_model.load_state_dict(core_model.state_dict())
        hf_model.eval()

        input_ids = torch.randint(0, cfg_dict["vocab_size"], (2, 16))

        with torch.no_grad():
            core_out = core_model(input_ids=input_ids)
            hf_out = hf_model(input_ids=input_ids)

        torch.testing.assert_close(hf_out.logits, core_out.logits)

    def test_loss_matches(self, variant):
        cfg_dict = self._config_dict_for(variant)

        core_cfg = CoreConfig(**cfg_dict)
        core_model = core_cfg.build()
        core_model.eval()

        hf_cfg = CoreModelConfig(core_config=cfg_dict)
        hf_model = CoreModelForCausalLM(hf_cfg)
        hf_model.core_model.load_state_dict(core_model.state_dict())
        hf_model.eval()

        input_ids = torch.randint(0, cfg_dict["vocab_size"], (2, 16))
        labels = input_ids.clone()

        with torch.no_grad():
            core_out = core_model(input_ids=input_ids, labels=labels)
            hf_out = hf_model(input_ids=input_ids, labels=labels)

        torch.testing.assert_close(hf_out.loss, core_out.loss)


# ==================================================================
# 3. State-dict loading (simulated checkpoint formats)
# ==================================================================


class TestCheckpointLoading:

    def test_load_raw_state_dict(self):
        """Simulate a raw .pt checkpoint (keys without ``model.`` prefix)."""
        from scripts.convert_to_hf import _extract_state_dict

        fake_sd = {"embeddings.weight": torch.randn(10, 4), "blocks.0.att_norm.weight": torch.randn(4)}
        ckpt = {"state_dict": fake_sd}

        result = _extract_state_dict(ckpt, is_lightning=False)
        assert "embeddings.weight" in result
        assert "blocks.0.att_norm.weight" in result

    def test_load_lightning_state_dict(self):
        """Simulate a Lightning .ckpt (keys with ``model.`` prefix)."""
        from scripts.convert_to_hf import _extract_state_dict

        fake_sd = {
            "model.embeddings.weight": torch.randn(10, 4),
            "model.blocks.0.att_norm.weight": torch.randn(4),
        }
        ckpt = {"state_dict": fake_sd}

        result = _extract_state_dict(ckpt, is_lightning=True)
        assert "embeddings.weight" in result
        assert "blocks.0.att_norm.weight" in result
        assert "model.embeddings.weight" not in result

    def test_lightning_strips_only_model_prefix(self):
        """Non-``model.`` keys (optimizer states, etc.) pass through."""
        from scripts.convert_to_hf import _extract_state_dict

        fake_sd = {
            "model.embeddings.weight": torch.randn(10, 4),
            "some_other_key": torch.randn(3),
        }
        ckpt = {"state_dict": fake_sd}

        result = _extract_state_dict(ckpt, is_lightning=True)
        assert "embeddings.weight" in result
        assert "some_other_key" in result


# ==================================================================
# 4. YAML config extraction
# ==================================================================


class TestYAMLExtraction:

    def test_training_yaml(self, tmp_path):
        """Extract core_config from a training YAML with model.core_config."""
        from scripts.convert_to_hf import _extract_core_config_from_yaml

        yaml_content = """
model:
  core_config:
    n_layers: 4
    d_model: 128
    attention:
      n_heads: 4
    feed_forward:
      feed_forward_type: mlp
      ff_hidden_size: 256
    layer_norm:
      layer_norm_type: default
    vocab_size: 1000
    max_sequence_length: 512
  sequence_length: 512
training:
  batch_size: 8
"""
        yaml_file = tmp_path / "train.yaml"
        yaml_file.write_text(yaml_content)

        result = _extract_core_config_from_yaml(str(yaml_file))
        assert result["n_layers"] == 4
        assert result["d_model"] == 128
        assert result["vocab_size"] == 1000

    def test_bare_core_config_yaml(self, tmp_path):
        """Extract core_config from a bare CoreConfig YAML."""
        from scripts.convert_to_hf import _extract_core_config_from_yaml

        yaml_content = """
n_layers: 6
d_model: 256
attention:
  n_heads: 8
feed_forward:
  feed_forward_type: glu
  ff_hidden_size: 512
layer_norm:
  layer_norm_type: rms
vocab_size: 2000
max_sequence_length: 1024
"""
        yaml_file = tmp_path / "core.yaml"
        yaml_file.write_text(yaml_content)

        result = _extract_core_config_from_yaml(str(yaml_file))
        assert result["n_layers"] == 6
        assert result["d_model"] == 256

    def test_invalid_yaml_raises(self, tmp_path):
        """A YAML with neither layout should raise."""
        from scripts.convert_to_hf import _extract_core_config_from_yaml

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("some_key: 42\n")

        with pytest.raises(ValueError, match="Could not locate"):
            _extract_core_config_from_yaml(str(yaml_file))


# ==================================================================
# 5. save_pretrained / from_pretrained round-trip
# ==================================================================


class TestSaveLoadRoundTrip:

    @pytest.fixture(params=["base", "noble", "xsa", "normalized"])
    def variant(self, request):
        return request.param

    def _config_dict_for(self, variant: str) -> dict:
        return {
            "base": _base_config_dict,
            "noble": _noble_config_dict,
            "xsa": _xsa_config_dict,
            "normalized": _normalized_config_dict,
        }[variant]()

    def test_save_load_produces_same_logits(self, variant, tmp_path):
        cfg_dict = self._config_dict_for(variant)

        hf_cfg = CoreModelConfig(core_config=cfg_dict)
        hf_model = CoreModelForCausalLM(hf_cfg)
        hf_model.eval()

        input_ids = torch.randint(0, cfg_dict["vocab_size"], (1, 8))
        with torch.no_grad():
            original_logits = hf_model(input_ids=input_ids).logits

        save_dir = tmp_path / "hf_model"
        hf_model.save_pretrained(save_dir)

        loaded_model = CoreModelForCausalLM.from_pretrained(save_dir)
        loaded_model.eval()

        with torch.no_grad():
            loaded_logits = loaded_model(input_ids=input_ids).logits

        torch.testing.assert_close(loaded_logits, original_logits)


# ==================================================================
# 6. generate() smoke test
# ==================================================================


class TestGenerate:

    def test_greedy_generate_base(self):
        cfg_dict = _base_config_dict()
        hf_cfg = CoreModelConfig(core_config=cfg_dict)
        hf_model = CoreModelForCausalLM(hf_cfg)
        hf_model.eval()

        input_ids = torch.randint(0, cfg_dict["vocab_size"], (1, 4))
        with torch.no_grad():
            output = hf_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )

        assert output.shape[0] == 1
        assert output.shape[1] == 4 + 5

    def test_greedy_generate_noble(self):
        cfg_dict = _noble_config_dict()
        hf_cfg = CoreModelConfig(core_config=cfg_dict)
        hf_model = CoreModelForCausalLM(hf_cfg)
        hf_model.eval()

        input_ids = torch.randint(0, cfg_dict["vocab_size"], (1, 4))
        with torch.no_grad():
            output = hf_model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
            )

        assert output.shape[1] == 4 + 3

    def test_sampling_generate(self):
        cfg_dict = _base_config_dict()
        hf_cfg = CoreModelConfig(core_config=cfg_dict)
        hf_model = CoreModelForCausalLM(hf_cfg)
        hf_model.eval()

        input_ids = torch.randint(0, cfg_dict["vocab_size"], (1, 4))
        with torch.no_grad():
            output = hf_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.8,
                top_k=10,
            )

        assert output.shape[0] == 1
        assert output.shape[1] == 4 + 5



class TestEmbeddingAccessors:

    def test_get_input_embeddings(self):
        hf_cfg = CoreModelConfig(core_config=_base_config_dict())
        hf_model = CoreModelForCausalLM(hf_cfg)

        emb = hf_model.get_input_embeddings()
        assert emb is hf_model.core_model.embeddings

    def test_get_output_embeddings(self):
        hf_cfg = CoreModelConfig(core_config=_base_config_dict())
        hf_model = CoreModelForCausalLM(hf_cfg)

        out_emb = hf_model.get_output_embeddings()
        assert out_emb is hf_model.core_model.lm_head
