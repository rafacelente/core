import logging
from typing import Union
import torch
from core.optim import Muon

try:
    import bitsandbytes as bnb
except ImportError:
    logging.warning("bitsandbytes not installed, using torch.optim")
    bnb = None

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "bnb_adam_8bit": bnb.optim.Adam8bit,
    "bnb_adam": bnb.optim.Adam,
    "muon": Muon,
    "adamw": torch.optim.AdamW,
}

def get_optimizer(name: str, model: torch.nn.Module, lr: float, **kwargs) -> Union[torch.optim.Optimizer, list[torch.optim.Optimizer]]:
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer {name} not found")
    if name.startswith("bnb_"):
        if bnb is None:
            raise ImportError(f"bitsandbytes not installed but {name} optimizer selected.")
    if name == "muon":
        return _configure_muon(model, lr, **kwargs)
    return OPTIMIZERS[name](model.parameters(), lr=lr, **kwargs)

def _configure_muon(model: torch.nn.Module, lr: float, **kwargs) -> list[torch.optim.Optimizer]:
    embed_params: list[torch.nn.Parameter] = []
    lm_head_params: list[torch.nn.Parameter] = []
    matrix_params: list[torch.nn.Parameter] = []
    scalar_params: list[torch.nn.Parameter] = []

    assigned_param_ids: set[int] = set()

    for possible_attr in ("embed", "embeddings", "wte"):
        if hasattr(model, possible_attr):
            params = list(getattr(model, possible_attr).parameters())
            embed_params.extend(params)
            assigned_param_ids.update(id(p) for p in params)

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        lm_head_params.append(model.lm_head.weight)
        assigned_param_ids.add(id(model.lm_head.weight))

    # Iterate over all remaining parameters and bucket them by dimensionality
    scalar_param_names = []
    muon_param_names = []
    for name, p in model.named_parameters():
        if id(p) in assigned_param_ids:
            continue

        if p.ndim == 2:
            matrix_params.append(p)
            muon_param_names.append(name)
        else:
            scalar_params.append(p)
            scalar_param_names.append(name)

    betas = (0.8, 0.95)
    optimizers: list[torch.optim.Optimizer] = []

    if embed_params:
        optimizers.append(torch.optim.Adam(embed_params, lr=0.0001, betas=betas))

    if lm_head_params:
        optimizers.append(torch.optim.Adam(lm_head_params, lr=0.0001, betas=betas))

    if matrix_params:
        optimizers.append(Muon(matrix_params, lr=0.0001, momentum=0.95))

    if scalar_params:
        optimizers.append(torch.optim.AdamW(scalar_params, lr=0.0001, betas=betas, weight_decay=0.01))

    _all_param_ids = set()
    for opt in optimizers:
        for group in opt.param_groups:
            _all_param_ids.update(id(p) for p in group["params"])

    assert len(_all_param_ids) == len(list(model.parameters())), (
        "Some parameters were not assigned to any optimizer or were duplicated."
    )

    return optimizers
