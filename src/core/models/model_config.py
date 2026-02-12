from enum import Enum
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self
from core.config import AttentionConfig, DType, FeedForwardConfig, LayerNormConfig
from core.modules.init import InitMethod
from core.models.model import CoreModel, NormalizedCoreModel
from core.modules.attention import AttentionType
from core.modules.feed_forward import FeedForwardType
import yaml

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
    dropout: float = 0.0
    dtype: DType = DType.FLOAT32

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

    def build(self) -> "CoreModel":
        if self.transformer_type == CoreType.NORMALIZED:
            model = NormalizedCoreModel(
                n_layers=self.n_layers,
                d_model=self.d_model,
                attention_config=self.attention,
                feed_forward_config=self.feed_forward,
                layer_norm_config=self.layer_norm,
                dropout=self.dropout,
                dtype=self.dtype,
                init_method=self.init_method,
                init_seed=self.init_seed,
                vocab_size=self.vocab_size,
                ignore_index=self.pad_token_id
            )
        else:
            model = CoreModel(
            n_layers=self.n_layers,
            d_model=self.d_model,
            attention_config=self.attention,
            feed_forward_config=self.feed_forward,
            layer_norm_config=self.layer_norm,
            dropout=self.dropout,
            dtype=self.dtype,
            init_method=self.init_method,
            init_seed=self.init_seed,
            vocab_size=self.vocab_size,
            ignore_index=self.pad_token_id
        )
        model.init_weights(max_seq_len=self.max_sequence_length)
        return model