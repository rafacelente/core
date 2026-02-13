"""
Adding a new optimization:

1. Add a `bool` field to `KernelOptimizations`.
2. Write a `KernelOptimization` subclass decorated with
   `@KernelOptimization.register("<field_name>")`.
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from core.models.model import CoreModel
    from core.models.model_config import CoreConfig

logger = logging.getLogger(__name__)

class KernelOptimization(ABC):
    """Base class for a single kernel-optimization handler.

    Subclass, implement both `apply_to_config` and `apply_to_model`,
    and decorate with `@KernelOptimization.register("<field_name>")`
    where `<field_name>` matches the corresponding boolean field on
    `KernelOptimizations`.
    """

    _registry: ClassVar[Dict[str, KernelOptimization]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(handler_cls):
            if name in cls._registry:
                raise ValueError(f"Optimization handler '{name}' is already registered")
            cls._registry[name] = handler_cls()
            return handler_cls
        return decorator

    @classmethod
    def get_handler(cls, name: str) -> Optional[KernelOptimization]:
        return cls._registry.get(name)

    @classmethod
    def registered_names(cls) -> list[str]:
        return list(cls._registry.keys())

    @abstractmethod
    def apply_to_config(self, config: CoreConfig, enabled: bool) -> None:
        """Mutate `<config>` in place to enable or disable this optimization."""
        ...

    @abstractmethod
    def apply_to_model(self, model: CoreModel, enabled: bool) -> None:
        """Mutate model in place to enable or disable this optimization."""
        ...


@dataclass
class KernelOptimizations:
    """Independently toggle each fused-kernel optimization.

    All flags default to `False` (standard PyTorch implementations).
    The convenience constructors `all` and `none`, as well as
    `any_enabled`, are driven by `dataclasses.fields` so they
    automatically cover any fields added in the future.

    fused_rope : bool
        Use the Triton fused rotary-position-embedding kernel.
    fused_cross_entropy : bool
        Use the fused cross-entropy loss (CUTE kernel).
    fused_rms_norm : bool
        Use the fused RMSNorm kernel (requires SM100+ / Blackwell).
        Silently ignored for models that use standard `LayerNorm`
    """

    fused_rope: bool = False
    fused_cross_entropy: bool = False
    fused_rms_norm: bool = False

    @classmethod
    def all(cls) -> KernelOptimizations:
        return cls(**{f.name: True for f in dataclasses.fields(cls)})

    @classmethod
    def none(cls) -> KernelOptimizations:
        return cls()

    def any_enabled(self) -> bool:
        return any(getattr(self, f.name) for f in dataclasses.fields(self))

def apply_kernel_optimizations(
    model: CoreModel,
    optimizations: KernelOptimizations,
) -> None:
    """Apply (or revert) kernel optimizations on `model` in-place.

    Iterates every field of `optimizations` and delegates to the
    corresponding registered `KernelOptimization` handler.

    Args:
    model:
        The :class:`CoreModel` (or :class:`NormalizedCoreModel`) to mutate.
    optimizations:
        Which optimizations should be active after this call.
        Passing `KernelOptimizations.none()` reverts everything to
        standard implementations.
    """
    for f in dataclasses.fields(optimizations):
        handler = KernelOptimization.get_handler(f.name)
        if handler is not None:
            handler.apply_to_model(model, getattr(optimizations, f.name))


def revert_kernel_optimizations(model: CoreModel) -> None:
    """Revert all kernel optimizations to standard implementations.

    Convenience wrapper equivalent to:
    apply_kernel_optimizations(model, KernelOptimizations.none())
    """
    apply_kernel_optimizations(model, KernelOptimizations.none())



@KernelOptimization.register("fused_rope")
class FusedRoPE(KernelOptimization):
    def apply_to_config(self, config: CoreConfig, enabled: bool) -> None:
        from core.modules.rope import RoPEType

        if config.attention.rope is not None:
            config.attention.rope.type = (
                RoPEType.FUSED if enabled else RoPEType.DEFAULT
            )

    def apply_to_model(self, model: CoreModel, enabled: bool) -> None:
        for block in model.blocks.values():
            att = block.attention
            if hasattr(att, "rope"):
                att.rope.use_fused = enabled
        logger.debug("RoPE implementation set to %s", "fused" if enabled else "default")


@KernelOptimization.register("fused_cross_entropy")
class FusedCrossEntropy(KernelOptimization):
    """Toggle the fused cross-entropy loss (CUTE kernel).

    The kernel requires `vocab_size` to be a multiple of
    `VOCAB_ALIGNMENT` (64). `apply_to_config` automatically pads
    the config's `vocab_size` when enabling.
    """

    VOCAB_ALIGNMENT = 64

    @staticmethod
    def pad_vocab_size(vocab_size: int, alignment: int = 64) -> int:
        """Round *vocab_size* **up** to the nearest multiple of *alignment*."""
        return ((vocab_size + alignment - 1) // alignment) * alignment

    def apply_to_config(self, config: CoreConfig, enabled: bool) -> None:
        from core.modules.loss import LossType

        config.loss.type = (
            LossType.FUSED_CROSS_ENTROPY if enabled else LossType.CROSS_ENTROPY
        )

        if enabled:
            aligned = self.pad_vocab_size(config.vocab_size, self.VOCAB_ALIGNMENT)
            if aligned != config.vocab_size:
                logger.debug(
                    "Padding vocab_size from %d to %d (multiple of %d) "
                    "for fused cross-entropy kernel",
                    config.vocab_size,
                    aligned,
                    self.VOCAB_ALIGNMENT,
                )
                config.vocab_size = aligned

    def apply_to_model(self, model: CoreModel, enabled: bool) -> None:
        from core.modules.loss import CrossEntropyLoss, FusedCrossEntropyLoss

        if enabled and model.vocab_size % self.VOCAB_ALIGNMENT != 0:
            raise ValueError(
                f"Fused cross-entropy requires vocab_size to be a multiple of "
                f"{self.VOCAB_ALIGNMENT}, but the model was built with "
                f"vocab_size={model.vocab_size}. Rebuild the model with a "
                f"padded vocab_size (use CoreConfig.with_kernel_optimizations) "
                f"or pad to {self.pad_vocab_size(model.vocab_size)}."
            )

        ignore_index = model.loss_fn.ignore_index
        if enabled and not isinstance(model.loss_fn, FusedCrossEntropyLoss):
            model.loss_fn = FusedCrossEntropyLoss(ignore_index=ignore_index)
        elif not enabled and not isinstance(model.loss_fn, CrossEntropyLoss):
            model.loss_fn = CrossEntropyLoss(ignore_index=ignore_index)
        logger.debug(
            "Cross-entropy implementation set to %s",
            "fused" if enabled else "default",
        )


@KernelOptimization.register("fused_rms_norm")
class FusedRMSNorm(KernelOptimization):
    """Toggle the fused RMSNorm kernel (QuackRMSNorm, SM100+).

    Only affects ``nn.RMSNorm`` modules; standard ``nn.LayerNorm``
    (GPT family) is left untouched.
    """

    def apply_to_config(self, config: CoreConfig, enabled: bool) -> None:
        from core.modules.layer_norm import LayerNormType

        rms_variants = {LayerNormType.RMS, LayerNormType.RMS_FAST}
        target = LayerNormType.RMS_FAST if enabled else LayerNormType.RMS

        if config.layer_norm.layer_norm_type in rms_variants:
            config.layer_norm.layer_norm_type = target
        if (
            config.output_norm is not None
            and config.output_norm.layer_norm_type in rms_variants
        ):
            config.output_norm.layer_norm_type = target
        if (
            config.attention.qk_norm is not None
            and config.attention.qk_norm.layer_norm_type in rms_variants
        ):
            config.attention.qk_norm.layer_norm_type = target

    def apply_to_model(self, model: CoreModel, enabled: bool) -> None:
        try:
            from quack.rmsnorm import QuackRMSNorm
        except ImportError:
            if enabled:
                logger.warning(
                    "Cannot apply fused RMSNorm: 'quack' package is not "
                    "available. Skipping RMSNorm optimisation."
                )
            return

        swapped = 0
        for parent in list(model.modules()):
            for name, child in parent.named_children():
                if enabled and isinstance(child, nn.RMSNorm) and not isinstance(child, QuackRMSNorm):
                    new = QuackRMSNorm(child.weight.shape[0], eps=child.eps)
                    new.weight = child.weight
                    setattr(parent, name, new)
                    swapped += 1
                elif not enabled and isinstance(child, QuackRMSNorm):
                    new = nn.RMSNorm(child.weight.shape[0], eps=child.eps)
                    new.weight = child.weight
                    setattr(parent, name, new)
                    swapped += 1

        logger.debug(
            "RMSNorm implementation set to %s (%d modules swapped)",
            "fused" if enabled else "default",
            swapped,
        )
