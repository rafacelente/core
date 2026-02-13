from __future__ import annotations

from enum import Enum
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, model_validator, Field
from typing_extensions import Self
import yaml

from core.modules.init import InitMethod
from core.models.model import CoreModel, NormalizedCoreModel
from core.modules.attention import AttentionType, AttentionConfig
from core.modules.feed_forward import FeedForwardType, FeedForwardConfig
from core.modules.layer_norm import LayerNormConfig
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
    output_norm: Optional[LayerNormConfig] = None
    dropout: float = 0.0
    dtype: DType = DType.FLOAT32
    loss: LossConfig = Field(default_factory= lambda:LossConfig())
    init_method: InitMethod = InitMethod.NORMAL
    init_seed: int = 42

    vocab_size: int
    max_sequence_length: int

    pad_token_id: int = -100

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def validate_type(self) -> Self:
        if self.transformer_type == CoreType.NORMALIZED:
            if self.attention.dropout != 0.0:
                raise ValueError("NormalizedCoreModel does not support dropout")
            self.attention.type = AttentionType.NORMALIZED
            if self.feed_forward.feed_forward_type == FeedForwardType.MLP:
                self.feed_forward.feed_forward_type = FeedForwardType.NORMALIZED_MLP
            elif self.feed_forward.feed_forward_type == FeedForwardType.GLU:
                self.feed_forward.feed_forward_type = FeedForwardType.NORMALIZED_GLU
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

    def build(self) -> CoreModel:
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
        )
        model.init_weights(max_seq_len=self.max_sequence_length)
        return model