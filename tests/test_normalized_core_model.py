import pytest
import torch
import math

from core.config import AttentionConfig, FeedForwardConfig, LayerNormConfig, DType
from core.model import CoreConfig, NormalizedCoreModel, CoreModel, CoreOutput
from core.modules.attention import AttentionType
from core.modules.feed_forward import FeedForwardType, ActivationType
from core.modules.init import InitMethod


@pytest.fixture(scope="module")
def normalized_core_config() -> CoreConfig:
    """Provides a CoreConfig configured for normalized components."""
    d_model = 32  # Small dimension for testing
    n_heads = 4
    n_layers = 2
    vocab_size = 100
    return CoreConfig(
        n_layers=n_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        attention=AttentionConfig(
            type=AttentionType.NORMALIZED,
            n_heads=n_heads,
        ),
        feed_forward=FeedForwardConfig(
            ff_hidden_size=d_model,
            feed_forward_type=FeedForwardType.NORMALIZED_MLP,
        ),
        layer_norm=LayerNormConfig(),
        max_sequence_length=128,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def regular_core_config() -> CoreConfig:
    """Provides a regular CoreConfig for comparison tests."""
    d_model = 32
    n_heads = 4
    n_layers = 2
    vocab_size = 100
    return CoreConfig(
        n_layers=n_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        attention=AttentionConfig(
            type=AttentionType.DEFAULT,
            n_heads=n_heads,
        ),
        feed_forward=FeedForwardConfig(
            ff_hidden_size=d_model,
            feed_forward_type=FeedForwardType.MLP,
        ),
        layer_norm=LayerNormConfig(),
        max_sequence_length=128,
        dropout=0.0,
    )


def test_normalized_config_creation(normalized_core_config: CoreConfig):
    """Tests if the normalized CoreConfig fixture can be created successfully."""
    assert normalized_core_config is not None
    assert normalized_core_config.d_model == 32
    assert normalized_core_config.n_layers == 2
    assert normalized_core_config.attention.n_heads == 4
    assert normalized_core_config.attention.type == AttentionType.NORMALIZED
    assert normalized_core_config.feed_forward.feed_forward_type == FeedForwardType.NORMALIZED_MLP
    assert normalized_core_config.dropout == 0.0


def test_dropout_validation():
    """Tests that NormalizedCoreModel raises ValueError when dropout > 0."""
    d_model = 32
    n_heads = 4
    n_layers = 2
    vocab_size = 100
    
    attention_config = AttentionConfig(
        type=AttentionType.NORMALIZED,
        n_heads=n_heads,
    )
    feed_forward_config = FeedForwardConfig(
        ff_hidden_size=d_model,
        feed_forward_type=FeedForwardType.NORMALIZED_MLP,
    )
    layer_norm_config = LayerNormConfig()
    
    with pytest.raises(ValueError, match="NormalizedCoreModel does not support dropout"):
        NormalizedCoreModel(
            d_model=d_model,
            n_layers=n_layers,
            vocab_size=vocab_size,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
            layer_norm_config=layer_norm_config,
            dropout=0.1,
        )
    
    try:
        model = NormalizedCoreModel(
            d_model=d_model,
            n_layers=n_layers,
            vocab_size=vocab_size,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
            layer_norm_config=layer_norm_config,
            dropout=0.0,
        )
        assert model is not None
    except Exception as e:
        pytest.fail(f"NormalizedCoreModel creation with dropout=0.0 failed: {e}")


def test_normalized_model_building(normalized_core_config: CoreConfig):
    """Tests if a NormalizedCoreModel can be built successfully from the config."""
    try:
        model = NormalizedCoreModel(
            d_model=normalized_core_config.d_model,
            n_layers=normalized_core_config.n_layers,
            vocab_size=normalized_core_config.vocab_size,
            attention_config=normalized_core_config.attention,
            feed_forward_config=normalized_core_config.feed_forward,
            layer_norm_config=normalized_core_config.layer_norm,
            dropout=normalized_core_config.dropout,
            dtype=normalized_core_config.dtype,
            init_method=InitMethod.NORMALIZED,
            init_seed=normalized_core_config.init_seed,
        )
        model.init_weights(max_seq_len=128)
        model = model.to(device="cpu")
        assert model is not None
        assert model.d_model == normalized_core_config.d_model
        assert model.n_layers == normalized_core_config.n_layers
        assert model.vocab_size == normalized_core_config.vocab_size
        assert model.init_method == InitMethod.NORMALIZED
        assert next(model.parameters()).device == torch.device("cpu")

    except Exception as e:
        pytest.fail(f"NormalizedCoreModel building failed with error: {e}")


def test_matrix_normalization(normalized_core_config: CoreConfig):
    """Tests that matrices are properly normalized after initialization."""
    model = NormalizedCoreModel(
        d_model=normalized_core_config.d_model,
        n_layers=normalized_core_config.n_layers,
        vocab_size=normalized_core_config.vocab_size,
        attention_config=normalized_core_config.attention,
        feed_forward_config=normalized_core_config.feed_forward,
        layer_norm_config=normalized_core_config.layer_norm,
        dropout=normalized_core_config.dropout,
        dtype=normalized_core_config.dtype,
        init_method=InitMethod.NORMALIZED,
        init_seed=42,  # Fixed seed for reproducibility
    )
    model.init_weights(max_seq_len=128)
    model = model.to(device="cpu")
    
    def is_normalized(tensor: torch.Tensor, dim: int = -1, tolerance: float = 1e-5) -> bool:
        """Check if a tensor is normalized along the specified dimension."""
        norms = tensor.norm(p=2, dim=dim, keepdim=True)
        return torch.allclose(norms, torch.ones_like(norms), atol=tolerance)
    
    # Check that embedding weights are normalized
    assert is_normalized(model.embeddings.weight, dim=-1), "Embedding weights should be normalized"
    
    # Check that lm_head weights are normalized
    assert is_normalized(model.lm_head.weight, dim=-1), "LM head weights should be normalized along dim=-1"
    
    # Check attention matrices in each block
    for block_idx, block in enumerate(model.blocks.values()):
        assert is_normalized(block.attention.w_q.weight, dim=-1), f"Block {block_idx} attention w_q should be normalized"
        assert is_normalized(block.attention.w_k.weight, dim=-1), f"Block {block_idx} attention w_k should be normalized"


def test_post_optim_step(normalized_core_config: CoreConfig):
    """Tests that post_optim_step properly re-normalizes matrices."""
    model = NormalizedCoreModel(
        d_model=normalized_core_config.d_model,
        n_layers=normalized_core_config.n_layers,
        vocab_size=normalized_core_config.vocab_size,
        attention_config=normalized_core_config.attention,
        feed_forward_config=normalized_core_config.feed_forward,
        layer_norm_config=normalized_core_config.layer_norm,
        dropout=normalized_core_config.dropout,
        dtype=normalized_core_config.dtype,
        init_method=InitMethod.NORMALIZED,
        init_seed=42,
    )
    model.init_weights(max_seq_len=128)
    model = model.to(device="cpu")
    
    # Artificially modify some weights to break normalization
    with torch.no_grad():
        model.embeddings.weight[0] *= 2.0
        model.lm_head.weight[:, 0] *= 0.5
        if model.blocks:
            first_block = next(iter(model.blocks.values()))
            first_block.attention.w_q.weight[0] *= 1.5
    
    def is_normalized(tensor: torch.Tensor, dim: int = -1, tolerance: float = 1e-5) -> bool:
        norms = tensor.norm(p=2, dim=dim, keepdim=True)
        return torch.allclose(norms, torch.ones_like(norms), atol=tolerance)
    
    assert not is_normalized(model.embeddings.weight, dim=-1)
    assert not is_normalized(model.lm_head.weight, dim=0)
    if model.blocks:
        first_block = next(iter(model.blocks.values()))
        assert not is_normalized(first_block.attention.w_q.weight, dim=-1)
    
    model.post_optim_step()
    
    assert is_normalized(model.embeddings.weight, dim=-1), "Embeddings should be normalized after post_optim_step"
    assert is_normalized(model.lm_head.weight, dim=-1), "LM head should be normalized after post_optim_step"
    if model.blocks:
        first_block = next(iter(model.blocks.values()))
        assert is_normalized(first_block.attention.w_q.weight, dim=-1), "Attention w_q should be normalized after post_optim_step"


def test_simple_forward(normalized_core_config: CoreConfig):
    """Tests a simple forward pass with NormalizedCoreModel without errors."""
    batch_size = 2
    seq_len = 10
    device = torch.device("cpu")

    model = NormalizedCoreModel(
        d_model=normalized_core_config.d_model,
        n_layers=normalized_core_config.n_layers,
        vocab_size=normalized_core_config.vocab_size,
        attention_config=normalized_core_config.attention,
        feed_forward_config=normalized_core_config.feed_forward,
        layer_norm_config=normalized_core_config.layer_norm,
        dropout=normalized_core_config.dropout,
        dtype=normalized_core_config.dtype,
        init_method=InitMethod.NORMALIZED,
    )
    model.init_weights(max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, normalized_core_config.vocab_size, (batch_size, seq_len), device=device)

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids)

        assert output is not None
        assert isinstance(output, CoreOutput)
        # Check output shape: (batch_size, seq_len, vocab_size)
        assert output.logits.shape == (batch_size, seq_len, normalized_core_config.vocab_size)
        assert output.logits.device == device

    except Exception as e:
        pytest.fail(f"NormalizedCoreModel forward pass failed with error: {e}")


def test_forward_with_labels(normalized_core_config: CoreConfig):
    """Tests forward pass with labels to compute loss."""
    batch_size = 2
    seq_len = 10
    device = torch.device("cpu")

    model = NormalizedCoreModel(
        d_model=normalized_core_config.d_model,
        n_layers=normalized_core_config.n_layers,
        vocab_size=normalized_core_config.vocab_size,
        attention_config=normalized_core_config.attention,
        feed_forward_config=normalized_core_config.feed_forward,
        layer_norm_config=normalized_core_config.layer_norm,
        dropout=normalized_core_config.dropout,
        dtype=normalized_core_config.dtype,
        init_method=InitMethod.NORMALIZED,
    )
    model.init_weights(max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, normalized_core_config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, normalized_core_config.vocab_size, (batch_size, seq_len), device=device)

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=labels)

        assert output is not None
        assert isinstance(output, CoreOutput)
        assert output.logits.shape == (batch_size, seq_len, normalized_core_config.vocab_size)
        assert output.loss is not None
        assert output.loss.item() >= 0

    except Exception as e:
        pytest.fail(f"NormalizedCoreModel forward pass with labels failed with error: {e}")


def test_comparison_with_regular_model(normalized_core_config: CoreConfig, regular_core_config: CoreConfig):
    """Tests that NormalizedCoreModel produces different outputs than regular CoreModel."""
    batch_size = 2
    seq_len = 10
    device = torch.device("cpu")

    normalized_model = NormalizedCoreModel(
        d_model=normalized_core_config.d_model,
        n_layers=normalized_core_config.n_layers,
        vocab_size=normalized_core_config.vocab_size,
        attention_config=normalized_core_config.attention,
        feed_forward_config=normalized_core_config.feed_forward,
        layer_norm_config=normalized_core_config.layer_norm,
        dropout=normalized_core_config.dropout,
        init_method=InitMethod.NORMALIZED,
        init_seed=42,
    )
    
    regular_model = CoreModel(
        d_model=regular_core_config.d_model,
        n_layers=regular_core_config.n_layers,
        vocab_size=regular_core_config.vocab_size,
        attention_config=regular_core_config.attention,
        feed_forward_config=regular_core_config.feed_forward,
        layer_norm_config=regular_core_config.layer_norm,
        dropout=regular_core_config.dropout,
        init_method=InitMethod.NORMAL,
        init_seed=42,
    )

    normalized_model.init_weights(max_seq_len=seq_len)
    regular_model.init_weights(max_seq_len=seq_len)
    
    normalized_model.eval()
    regular_model.eval()

    input_ids = torch.randint(0, normalized_core_config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        normalized_output = normalized_model(input_ids=input_ids)
        regular_output = regular_model(input_ids=input_ids)

    assert not torch.allclose(normalized_output.logits, regular_output.logits, atol=1e-5), \
        "Normalized and regular models should produce different outputs"


def test_different_feedforward_types():
    """Tests NormalizedCoreModel with different normalized feed forward types."""
    d_model = 32
    n_heads = 4
    n_layers = 1
    vocab_size = 100
    
    for ff_type in [FeedForwardType.NORMALIZED_MLP, FeedForwardType.NORMALIZED_GLU]:
        attention_config = AttentionConfig(
            type=AttentionType.NORMALIZED,
            n_heads=n_heads,
        )
        feed_forward_config = FeedForwardConfig(
            ff_hidden_size=d_model,
            feed_forward_type=ff_type,
        )
        layer_norm_config = LayerNormConfig()
        
        try:
            model = NormalizedCoreModel(
                d_model=d_model,
                n_layers=n_layers,
                vocab_size=vocab_size,
                attention_config=attention_config,
                feed_forward_config=feed_forward_config,
                layer_norm_config=layer_norm_config,
                dropout=0.0,
                init_method=InitMethod.NORMALIZED,
            )
            model.init_weights(max_seq_len=64)
            
            batch_size = 1
            seq_len = 8
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids=input_ids)
            
            assert output.logits.shape == (batch_size, seq_len, vocab_size)
            
        except Exception as e:
            pytest.fail(f"NormalizedCoreModel with {ff_type} failed: {e}")


def test_model_parameter_counts(normalized_core_config: CoreConfig):
    """Tests that parameter counting methods work correctly."""
    model = NormalizedCoreModel(
        d_model=normalized_core_config.d_model,
        n_layers=normalized_core_config.n_layers,
        vocab_size=normalized_core_config.vocab_size,
        attention_config=normalized_core_config.attention,
        feed_forward_config=normalized_core_config.feed_forward,
        layer_norm_config=normalized_core_config.layer_norm,
        dropout=normalized_core_config.dropout,
        dtype=normalized_core_config.dtype,
        init_method=InitMethod.NORMALIZED,
    )
    model.init_weights(max_seq_len=128)
    
    total_params = model.num_parameters()
    trainable_params = model.num_trainable_parameters()
    
    assert total_params > 0, "Model should have parameters"
    assert trainable_params > 0, "Model should have trainable parameters"
    assert trainable_params <= total_params, "Trainable parameters should not exceed total parameters"
    
    manual_count = sum(p.numel() for p in model.parameters())
    assert total_params == manual_count, "num_parameters() should match manual count"
