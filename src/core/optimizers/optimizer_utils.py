from typing import Union
from enum import Enum
import torch
from .muon import Muon, configure_muon


class OptimizerName(str, Enum):
    """Enum listing the identifiers of the supported optimizers."""

    ADAM = "adam"
    MUON = "muon"
    MUON_8BIT = "muon_8bit"
    ADAMW = "adamw"
    SGD = "sgd"


OPTIMIZER_MAPPING: dict[OptimizerName, type[torch.optim.Optimizer]] = {
    OptimizerName.ADAM: torch.optim.Adam,
    OptimizerName.MUON: Muon,
    OptimizerName.ADAMW: torch.optim.AdamW,
    OptimizerName.SGD: torch.optim.SGD,
}


def get_optimizer(
    name: Union[str, OptimizerName],
    model: torch.nn.Module,
    lr: float,
    **kwargs,
) -> Union[torch.optim.Optimizer, list[torch.optim.Optimizer]]:
    """Instantiate and return the requested optimizer.

    Args:
        name: Identifier of the optimizer, either as an *OptimizerName* enum
            member or its corresponding string value.
        model: Target model whose parameters will be optimized.
        lr: Learning rate.
        **kwargs: Extra keyword arguments forwarded to the underlying
            optimizer constructor.

    Raises:
        ValueError: If *name* does not correspond to any supported optimizer.
    """

    if isinstance(name, str):
        try:
            name_enum = OptimizerName(name)
        except ValueError as err:
            raise ValueError(f"Optimizer {name} not found") from err
    else:
        name_enum = name

    if name_enum is OptimizerName.MUON:
        return configure_muon(model, lr, **kwargs)
    elif name_enum is OptimizerName.MUON_8BIT:
        try:
            from .muon import Muon8bit
            return configure_muon(model, lr, optimizer_class=Muon8bit, **kwargs)
        except ImportError:
            raise ValueError("Muon8bit optimizer not available. Please ensure it's implemented in the muon module.")

    optimizer_cls = OPTIMIZER_MAPPING[name_enum]
    return optimizer_cls(model.parameters(), lr=lr, **kwargs)  # type: ignore[arg-type]