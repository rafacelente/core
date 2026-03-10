import pytest
import torch
import torch.nn as nn

from core.optimizers.muon import Muon, configure_muon
from core.optimizers.manifold_muon import ManifoldMuon, configure_manifold_muon
from core.optimizers.optimizer_utils import OptimizerName, get_optimizer


class _TinyModel(nn.Module):
    """Minimal model with embeddings, lm_head, 2D matrix, and 1D params."""

    def __init__(self, vocab_size: int = 64, d_model: int = 16, hidden: int = 32):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, hidden, bias=False)
        self.layer_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)


# ---------------------------------------------------------------------------
# configure_muon
# ---------------------------------------------------------------------------

def test_configure_muon_returns_muon_instance():
    model = _TinyModel()
    optimizer = configure_muon(model, lr=0.02)
    assert isinstance(optimizer, Muon)


def test_configure_muon_assigns_all_params():
    model = _TinyModel()
    optimizer = configure_muon(model, lr=0.02)

    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))

    model_param_ids = {id(p) for p in model.parameters()}
    assert opt_param_ids == model_param_ids


def test_configure_muon_separates_matrix_and_scalar_params():
    model = _TinyModel()
    optimizer = configure_muon(model, lr=0.02)

    muon_params = []
    adamw_params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if optimizer.state[p].get("use_muon"):
                muon_params.append(p)
            else:
                adamw_params.append(p)

    for p in muon_params:
        assert p.ndim == 2, "Muon params should be 2D matrices"

    adamw_ids = {id(p) for p in adamw_params}
    assert id(model.embeddings.weight) in adamw_ids, "Embeddings should use AdamW"
    assert id(model.lm_head.weight) in adamw_ids, "lm_head should use AdamW"


# ---------------------------------------------------------------------------
# configure_manifold_muon
# ---------------------------------------------------------------------------

def test_configure_manifold_muon_returns_manifold_muon_instance():
    model = _TinyModel()
    optimizer = configure_manifold_muon(model, lr=0.02)
    assert isinstance(optimizer, ManifoldMuon), (
        f"Expected ManifoldMuon, got {type(optimizer).__name__}"
    )
    assert not isinstance(optimizer, Muon), (
        "ManifoldMuon should not be an instance of Muon"
    )


def test_configure_manifold_muon_assigns_all_params():
    model = _TinyModel()
    optimizer = configure_manifold_muon(model, lr=0.02)

    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))

    model_param_ids = {id(p) for p in model.parameters()}
    assert opt_param_ids == model_param_ids


def test_configure_manifold_muon_separates_matrix_and_scalar_params():
    model = _TinyModel()
    optimizer = configure_manifold_muon(model, lr=0.02)

    muon_params = []
    adamw_params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if optimizer.state[p].get("use_muon"):
                muon_params.append(p)
            else:
                adamw_params.append(p)

    for p in muon_params:
        assert p.ndim == 2, "Manifold Muon params should be 2D matrices"

    adamw_ids = {id(p) for p in adamw_params}
    assert id(model.embeddings.weight) in adamw_ids
    assert id(model.lm_head.weight) in adamw_ids


# ---------------------------------------------------------------------------
# get_optimizer dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["adam", "adamw", "sgd"])
def test_get_optimizer_standard(name):
    model = _TinyModel()
    optimizer = get_optimizer(name, model, lr=1e-3)
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_get_optimizer_muon():
    model = _TinyModel()
    optimizer = get_optimizer("muon", model, lr=0.02)
    assert isinstance(optimizer, Muon)


def test_get_optimizer_invalid_name():
    model = _TinyModel()
    with pytest.raises(ValueError, match="not found"):
        get_optimizer("nonexistent", model, lr=1e-3)
