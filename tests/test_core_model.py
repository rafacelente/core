import pytest
import torch

from core.config import AttentionConfig, FeedForwardConfig, LayerNormConfig
from core.model import CoreConfig, CoreModel, CoreOutput


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