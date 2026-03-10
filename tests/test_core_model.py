import pytest
import torch

from core.modules.attention import AttentionConfig, AttentionType
from core.modules.feed_forward import FeedForwardConfig, FeedForwardType, ActivationType
from core.modules.layer_norm import LayerNormConfig
from core.modules.loss import LossConfig
from core.utils import DType
from core.models.model_config import CoreConfig, CoreType
from core.models.model import CoreModel
from core.models.model_utils import CoreOutput


@pytest.fixture(scope="module")
def core_config() -> CoreConfig:
    """Provides a minimal CoreConfig for testing."""
    d_model = 32  # Small dimension for testing
    n_heads = 4
    n_layers = 2
    vocab_size = 100
    return CoreConfig(
        n_layers=n_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        attention=AttentionConfig(
            n_heads=n_heads,
        ),
        feed_forward=FeedForwardConfig(
            ff_hidden_size=d_model,
        ),
        layer_norm=LayerNormConfig(),
        max_sequence_length=128,
    )


def test_config_creation(core_config: CoreConfig):
    """Tests if the CoreConfig fixture can be created successfully."""
    assert core_config is not None
    assert core_config.d_model == 32
    assert core_config.n_layers == 2
    assert core_config.attention.n_heads == 4
    assert core_config.feed_forward.ff_hidden_size == 32


def test_model_building(core_config: CoreConfig):
    """Tests if a CoreModel can be built successfully from the config."""
    vocab_size = 100  # Example vocab size
    try:
        model = CoreModel(
            d_model=core_config.d_model,
            n_layers=core_config.n_layers,
            vocab_size=vocab_size,
            attention_config=core_config.attention,
            feed_forward_config=core_config.feed_forward,
            layer_norm_config=core_config.layer_norm,
            dropout=core_config.dropout,
            dtype=core_config.dtype,
            init_method=core_config.init_method,
            init_seed=core_config.init_seed,
        )
        model.init_weights(max_seq_len=128)
        model = model.to(device="cpu")
        assert model is not None
        assert model.d_model == core_config.d_model
        assert model.n_layers == core_config.n_layers
        assert model.vocab_size == vocab_size
        assert next(model.parameters()).device == torch.device("cpu")

    except Exception as e:
        pytest.fail(f"Model building failed with error: {e}")


def test_simple_forward(core_config: CoreConfig):
    """Tests a simple forward pass on the CPU without errors."""
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    device = torch.device("cpu")

    model = CoreModel(
        d_model=core_config.d_model,
        n_layers=core_config.n_layers,
        vocab_size=vocab_size,
        attention_config=core_config.attention,
        feed_forward_config=core_config.feed_forward,
        layer_norm_config=core_config.layer_norm,
        dtype=core_config.dtype,  # Ensure dtype is consistent
    )
    model.init_weights(max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids)

        assert output is not None
        assert isinstance(output, CoreOutput)
        # Check output shape: (batch_size, seq_len, vocab_size)
        assert output.logits.shape == (batch_size, seq_len, vocab_size)
        # Check output device
        assert output.logits.device == device

    except Exception as e:
        pytest.fail(f"Model forward pass failed with error: {e}")


def test_forward_with_labels(core_config: CoreConfig):
    """Tests forward pass with labels computes a non-negative loss."""
    batch_size = 2
    seq_len = 10
    device = torch.device("cpu")

    model = CoreModel(
        d_model=core_config.d_model,
        n_layers=core_config.n_layers,
        vocab_size=core_config.vocab_size,
        attention_config=core_config.attention,
        feed_forward_config=core_config.feed_forward,
        layer_norm_config=core_config.layer_norm,
        loss_config=core_config.loss,
        dtype=core_config.dtype,
    )
    model.init_weights(max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, core_config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, core_config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels)

    assert isinstance(output, CoreOutput)
    assert output.logits.shape == (batch_size, seq_len, core_config.vocab_size)
    assert output.loss is not None
    assert output.loss.item() >= 0


def test_config_build_creates_model(core_config: CoreConfig):
    """Tests that CoreConfig.build() produces a working CoreModel."""
    model = core_config.build()

    assert isinstance(model, CoreModel)
    assert model.d_model == core_config.d_model
    assert model.n_layers == core_config.n_layers
    assert model.vocab_size == core_config.vocab_size

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, core_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids=input_ids)

    assert output.logits.shape == (batch_size, seq_len, core_config.vocab_size)


def test_model_parameter_counts(core_config: CoreConfig):
    """Tests that parameter counting methods return consistent values."""
    model = core_config.build()

    total_params = model.num_parameters()
    trainable_params = model.num_trainable_parameters()

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params

    manual_count = sum(p.numel() for p in model.parameters())
    assert total_params == manual_count


def test_all_public_config_classes_importable():
    """Validates that the key configuration classes are importable from their
    canonical module paths — the same paths used by the rest of the test suite.
    """
    assert issubclass(AttentionConfig, object)
    assert issubclass(FeedForwardConfig, object)
    assert issubclass(LayerNormConfig, object)
    assert issubclass(LossConfig, object)
    assert issubclass(CoreConfig, object)

    assert hasattr(AttentionType, "DEFAULT")
    assert hasattr(AttentionType, "NORMALIZED")
    assert hasattr(FeedForwardType, "MLP")
    assert hasattr(FeedForwardType, "GLU")
    assert hasattr(ActivationType, "GELU")
    assert hasattr(ActivationType, "SILU")
    assert hasattr(CoreType, "BASE")
    assert hasattr(CoreType, "NORMALIZED")
    assert hasattr(DType, "FLOAT32")
    assert hasattr(DType, "BFLOAT16")