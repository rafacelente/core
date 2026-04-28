import os
import tempfile

import pytest
import torch
import yaml

from core.models.model_config import CoreConfig, CoreType
from core.models.model import CoreModel
from core.models.model_utils import CoreOutput
from core.modules.linear import LinearType
from core.modules.noble import LinearNoble
from core.training.train import create_model_config


def _make_yaml(content: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(content, f)
    return path


MINIMAL_INLINE_CORE_CONFIG = {
    "transformer_type": "base",
    "n_layers": 2,
    "d_model": 64,
    "attention": {"n_heads": 4},
    "feed_forward": {"ff_hidden_size": 64, "activation_type": "gelu"},
    "layer_norm": {"layer_norm_type": "default"},
    "vocab_size": 128,
    "max_sequence_length": 32,
}

NOBLE_INLINE_CORE_CONFIG = {
    "transformer_type": "base",
    "n_layers": 2,
    "d_model": 64,
    "attention": {"n_heads": 4, "rope": {"type": "default"}},
    "feed_forward": {
        "feed_forward_type": "glu",
        "ff_hidden_size": 128,
        "activation_type": "silu",
    },
    "linear": {
        "type": "noble",
        "noble": {"r": 8, "activation_type": "cosnet", "apply_to": ["all"]},
    },
    "layer_norm": {"layer_norm_type": "rms"},
    "output_norm": {"layer_norm_type": "rms"},
    "vocab_size": 128,
    "max_sequence_length": 32,
    "init_method": "llama",
}


class TestCoreConfigFromDict:
    def test_build_from_minimal_dict(self):
        config = CoreConfig(**MINIMAL_INLINE_CORE_CONFIG)
        assert config.d_model == 64
        assert config.n_layers == 2
        model = config.build()
        assert isinstance(model, CoreModel)

    def test_build_noble_from_dict(self):
        config = CoreConfig(**NOBLE_INLINE_CORE_CONFIG)
        assert config.linear.type == LinearType.NOBLE
        model = config.build()
        assert isinstance(model, CoreModel)
        assert isinstance(model.blocks["0"].attention.w_q, LinearNoble)
        assert isinstance(model.blocks["0"].feed_forward.w1, LinearNoble)

    def test_noble_forward_pass(self):
        config = CoreConfig(**NOBLE_INLINE_CORE_CONFIG)
        model = config.build()
        model.eval()
        ids = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out = model(ids)
        assert isinstance(out, CoreOutput)
        assert out.logits.shape == (2, 16, 128)

    def test_noble_forward_with_loss(self):
        config = CoreConfig(**NOBLE_INLINE_CORE_CONFIG)
        model = config.build()
        model.eval()
        ids = torch.randint(0, 128, (2, 16))
        labels = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out = model(ids, labels=labels)
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_noble_apply_to_att_only(self):
        cfg = {**NOBLE_INLINE_CORE_CONFIG}
        cfg["linear"] = {
            "type": "noble",
            "noble": {"r": 8, "activation_type": "cosnet", "apply_to": ["att"]},
        }
        config = CoreConfig(**cfg)
        model = config.build()
        assert isinstance(model.blocks["0"].attention.w_q, LinearNoble)
        assert isinstance(model.blocks["0"].feed_forward.w1, torch.nn.Linear)
        assert not isinstance(model.blocks["0"].feed_forward.w1, LinearNoble)

    def test_noble_apply_to_ff_only(self):
        cfg = {**NOBLE_INLINE_CORE_CONFIG}
        cfg["linear"] = {
            "type": "noble",
            "noble": {"r": 8, "activation_type": "cosnet", "apply_to": ["ff"]},
        }
        config = CoreConfig(**cfg)
        model = config.build()
        assert isinstance(model.blocks["0"].feed_forward.w1, LinearNoble)
        assert isinstance(model.blocks["0"].attention.w_q, torch.nn.Linear)
        assert not isinstance(model.blocks["0"].attention.w_q, LinearNoble)

    def test_noble_param_count_exceeds_default(self):
        baseline = CoreConfig(**MINIMAL_INLINE_CORE_CONFIG).build()
        noble = CoreConfig(**NOBLE_INLINE_CORE_CONFIG).build()
        assert noble.num_parameters() > baseline.num_parameters()


class TestTrainingConfigInlineCoreConfig:
    """Tests that the YAML training config parser correctly picks up
    an inline ``core_config`` dict under the ``model:`` section and
    that ``create_model_config`` builds a working model from it."""

    def _training_yaml(self, core_config_dict: dict) -> str:
        content = {
            "model": {
                "core_config": core_config_dict,
                "sequence_length": 32,
            },
            "training": {"batch_size": 2, "max_epochs": 1},
            "optimizer": {"optimizer": "adamw"},
        }
        return _make_yaml(content)

    def test_from_yaml_parses_inline_core_config(self):
        from core.training.training_config import TrainingConfig

        path = self._training_yaml(MINIMAL_INLINE_CORE_CONFIG)
        try:
            tc = TrainingConfig.from_yaml(path)
            assert tc.core_config is not None
            assert isinstance(tc.core_config, dict)
            assert tc.core_config["d_model"] == 64
        finally:
            os.unlink(path)

    def test_create_model_config_from_inline(self):
        from core.training.training_config import TrainingConfig

        path = self._training_yaml(MINIMAL_INLINE_CORE_CONFIG)
        try:
            tc = TrainingConfig.from_yaml(path)
            config = create_model_config(tc, vocab_size=200, pad_token_id=0)
            assert isinstance(config, CoreConfig)
            assert config.vocab_size == 200
            assert config.pad_token_id == 0
            assert config.max_sequence_length == 32
        finally:
            os.unlink(path)

    def test_create_model_config_noble_inline(self):
        from core.training.training_config import TrainingConfig

        path = self._training_yaml(NOBLE_INLINE_CORE_CONFIG)
        try:
            tc = TrainingConfig.from_yaml(path)
            config = create_model_config(tc, vocab_size=200, pad_token_id=0)
            assert config.linear.type == LinearType.NOBLE
            model = config.build()
            assert isinstance(model.blocks["0"].attention.w_q, LinearNoble)
        finally:
            os.unlink(path)

    def test_noble_yaml_file_loads(self):
        """Smoke-test that the actual noble.yaml config file parses."""
        from core.training.training_config import TrainingConfig

        noble_path = os.path.join(os.path.dirname(__file__), "..", "config", "noble.yaml")
        if not os.path.exists(noble_path):
            pytest.skip("noble.yaml not found")

        tc = TrainingConfig.from_yaml(noble_path)
        assert tc.core_config is not None
        assert tc.core_config["n_layers"] == 12
        assert tc.core_config["linear"]["type"] == "noble"

    def test_recipe_still_works_without_core_config(self):
        """When core_config is absent the recipe registry path is used."""
        from core.training.training_config import TrainingConfig

        content = {
            "model": {"model_type": "gpt-tiny", "sequence_length": 32},
            "training": {"batch_size": 2, "max_epochs": 1},
            "optimizer": {"optimizer": "adamw"},
        }
        path = _make_yaml(content)
        try:
            tc = TrainingConfig.from_yaml(path)
            assert tc.core_config is None
            config = create_model_config(tc, vocab_size=200, pad_token_id=0)
            assert isinstance(config, CoreConfig)
            assert config.n_layers == 10  # gpt-tiny
        finally:
            os.unlink(path)
