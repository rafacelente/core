from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, model_validator, Field
from typing_extensions import Self
import torch
import yaml

from core.modules.init import InitMethod
from core.models.model import CoreModel, NormalizedCoreModel
from core.modules.attention import AttentionType, AttentionConfig
from core.modules.feed_forward import FeedForwardType, FeedForwardConfig
from core.modules.layer_norm import LayerNormConfig
from core.modules.linear import LinearConfig, LinearType
from core.modules.loss import LossConfig
from core.utils import DType

if TYPE_CHECKING:
    from core.optimizations import KernelOptimizations

class CoreType(str, Enum):
    BASE = "base"
    NORMALIZED = "normalized"


class CoreConfig(BaseModel):
    """
    Configuration for the core model.
    """

    transformer_type: CoreType = CoreType.BASE
    n_layers: int
    d_model: int
    attention: AttentionConfig
    feed_forward: FeedForwardConfig
    layer_norm: LayerNormConfig
    linear: LinearConfig = Field(default_factory=LinearConfig)
    output_norm: Optional[LayerNormConfig] = None
    dropout: float = 0.0
    dtype: DType = DType.FLOAT32
    loss: LossConfig = Field(default_factory=LossConfig)
    init_method: InitMethod = InitMethod.NORMAL
    init_seed: int = 42

    vocab_size: int
    max_sequence_length: int

    pad_token_id: int = -100
    label_weights_path: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def validate_type(self) -> Self:
        self.loss.ignore_index = self.pad_token_id

        if self.transformer_type == CoreType.NORMALIZED:
            if self.attention.dropout != 0.0:
                raise ValueError("NormalizedCoreModel does not support dropout")
            if self.attention.use_post_sdpa_gate:
                raise ValueError(
                    "use_post_sdpa_gate is not compatible with the normalized "
                    "transformer. NormalizedAttention overrides the forward pass "
                    "and the gate would be silently ignored."
                )
            self.attention.type = AttentionType.NORMALIZED
            if self.feed_forward.feed_forward_type == FeedForwardType.MLP:
                self.feed_forward.feed_forward_type = FeedForwardType.NORMALIZED_MLP
            elif self.feed_forward.feed_forward_type == FeedForwardType.GLU:
                self.feed_forward.feed_forward_type = FeedForwardType.NORMALIZED_GLU

        if self.linear.type != LinearType.DEFAULT and self.linear.noble is not None:
            apply_to = self.linear.noble.apply_to
            if "all" in apply_to or "att" in apply_to:
                self.attention.linear = self.linear
            if "all" in apply_to or "ff" in apply_to:
                self.feed_forward.linear = self.linear

        return self

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def with_kernel_optimizations(self, optimizations: KernelOptimizations) -> CoreConfig:
        """Return a deep copy of this config with kernel optimizations applied.
        """
        import dataclasses
        from core.optimizations import KernelOptimization

        config = self.model_copy(deep=True)
        for f in dataclasses.fields(optimizations):
            handler = KernelOptimization.get_handler(f.name)
            if handler is not None:
                handler.apply_to_config(config, getattr(optimizations, f.name))
        return config

    def _resolve_lm_linear(self) -> LinearConfig:
        if self.linear.type != LinearType.DEFAULT and self.linear.noble is not None:
            apply_to = self.linear.noble.apply_to
            if "all" in apply_to or "lm_head" in apply_to:
                return self.linear
        return LinearConfig()

    def build(self) -> CoreModel:
        logger = logging.getLogger(__name__)
        label_weights = None
        if self.label_weights_path is not None:
            label_weights = torch.load(self.label_weights_path, weights_only=True)
            logger.info(f"Loaded label weights from {self.label_weights_path} "
                        f"(shape={label_weights.shape})")

        lm_linear = self._resolve_lm_linear()

        if self.transformer_type == CoreType.NORMALIZED:
            model = NormalizedCoreModel(
                n_layers=self.n_layers,
                d_model=self.d_model,
                attention_config=self.attention,
                feed_forward_config=self.feed_forward,
                layer_norm_config=self.layer_norm,
                output_norm_config=self.output_norm,
                dropout=self.dropout,
                dtype=self.dtype,
                init_method=self.init_method,
                init_seed=self.init_seed,
                vocab_size=self.vocab_size,
                loss_config=self.loss,
                label_weights=label_weights,
                lm_linear_config=lm_linear,
            )
        else:
            model = CoreModel(
                n_layers=self.n_layers,
                d_model=self.d_model,
                attention_config=self.attention,
                feed_forward_config=self.feed_forward,
                layer_norm_config=self.layer_norm,
                output_norm_config=self.output_norm,
                dropout=self.dropout,
                dtype=self.dtype,
                init_method=self.init_method,
                init_seed=self.init_seed,
                vocab_size=self.vocab_size,
                loss_config=self.loss,
                label_weights=label_weights,
                lm_linear_config=lm_linear,
            )
        model.init_weights(max_seq_len=self.max_sequence_length)
        return model
