from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class CoreModelConfig(PretrainedConfig):
    """HuggingFace-compatible config that wraps a ``CoreConfig`` dict.

    The full ``CoreConfig`` fields are stored verbatim under
    ``self.core_config`` so they survive JSON round-trips through
    ``save_pretrained`` / ``from_pretrained``.  Standard HF fields
    (``vocab_size``, ``hidden_size``, …) are set for compatibility with
    tooling that inspects them.
    """

    model_type = "core"

    def __init__(
        self,
        core_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.core_config = core_config or {}

        kwargs.setdefault("vocab_size", self.core_config.get("vocab_size", 50257))
        kwargs.setdefault("hidden_size", self.core_config.get("d_model", 768))
        kwargs.setdefault("num_hidden_layers", self.core_config.get("n_layers", 12))
        kwargs.setdefault("tie_word_embeddings", False)

        n_heads = self.core_config.get("attention", {}).get("n_heads", 12)
        kwargs.setdefault("num_attention_heads", n_heads)
        kwargs.setdefault("dtype", "bfloat16")

        super().__init__(**kwargs)

    def to_core_config(self):
        """Reconstruct the Pydantic ``CoreConfig`` from the stored dict."""
        from core.models.model_config import CoreConfig

        return CoreConfig(**self.core_config)
