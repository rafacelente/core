"""Tests for TrainingConfig and CLI argument wiring.

Covers:
- max_grad_norm is present and has the correct default.
- YAML round-trip preserves max_grad_norm.
- CLI argument mapping includes max_grad_norm.
- All TrainingConfig fields that appear in _CLI_ARG_TO_CONFIG_FIELD
  actually exist on the dataclass.
"""

import dataclasses
import importlib
import tempfile
import os
import sys

import yaml
import pytest

from core.training.training_config import TrainingConfig


lightning_available = importlib.util.find_spec("lightning") is not None

try:
    from core.training.train import _CLI_ARG_TO_CONFIG_FIELD, _build_parser
    _train_importable = True
except Exception:
    _train_importable = False


def test_max_grad_norm_default():
    config = TrainingConfig()
    assert config.max_grad_norm == 1.0


def test_max_grad_norm_custom():
    config = TrainingConfig(max_grad_norm=0.5)
    assert config.max_grad_norm == 0.5


def test_max_grad_norm_from_yaml():
    yaml_content = {
        "training": {
            "max_grad_norm": 2.0,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        path = f.name

    try:
        config = TrainingConfig.from_yaml(path)
        assert config.max_grad_norm == 2.0
    finally:
        os.unlink(path)


def test_max_grad_norm_cli_override():
    yaml_content = {
        "training": {
            "max_grad_norm": 2.0,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        path = f.name

    try:
        config = TrainingConfig.from_yaml_with_overrides(path, {"max_grad_norm": 0.3})
        assert config.max_grad_norm == 0.3
    finally:
        os.unlink(path)


def test_yaml_round_trip_preserves_all_training_fields():
    """All scalar training fields survive a YAML round-trip."""
    yaml_content = {
        "training": {
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "max_epochs": 3,
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "dropout": 0.1,
            "max_grad_norm": 0.7,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        path = f.name

    try:
        config = TrainingConfig.from_yaml(path)
        for key, expected in yaml_content["training"].items():
            assert getattr(config, key) == expected, f"{key} mismatch"
    finally:
        os.unlink(path)


@pytest.mark.skipif(not _train_importable, reason="lightning import chain unavailable")
def test_max_grad_norm_in_cli_arg_mapping():
    assert "max_grad_norm" in _CLI_ARG_TO_CONFIG_FIELD.values(), (
        "max_grad_norm should be reachable via CLI"
    )


@pytest.mark.skipif(not _train_importable, reason="lightning import chain unavailable")
def test_cli_parser_has_max_grad_norm_arg():
    parser = _build_parser()
    args = parser.parse_args(["--max-grad-norm", "0.5"])
    assert args.max_grad_norm == 0.5


@pytest.mark.skipif(not _train_importable, reason="lightning import chain unavailable")
def test_all_cli_fields_exist_on_training_config():
    """Every config field referenced in the CLI mapping must exist on TrainingConfig."""
    field_names = {f.name for f in dataclasses.fields(TrainingConfig)}
    for cli_name, config_field in _CLI_ARG_TO_CONFIG_FIELD.items():
        assert config_field in field_names, (
            f"CLI arg '{cli_name}' maps to '{config_field}' which is not a "
            f"field on TrainingConfig"
        )
