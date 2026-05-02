from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.hf.configuration_core import CoreModelConfig


def _build_core_model_without_init(core_config):
    """Construct a :class:`CoreModel` *without* calling ``init_weights``.

    HF's ``from_pretrained`` instantiates the model on the ``meta``
    device before loading real weights, so we must skip
    ``init_weights`` (which warms the RoPE cache and touches real
    tensors).  The caller is responsible for loading weights
    afterwards.
    """
    from core.models.model import CoreModel, NormalizedCoreModel
    from core.models.model_config import CoreType

    lm_linear = core_config._resolve_lm_linear()
    model_cls = NormalizedCoreModel if core_config.transformer_type == CoreType.NORMALIZED else CoreModel

    model = model_cls(
        n_layers=core_config.n_layers,
        d_model=core_config.d_model,
        attention_config=core_config.attention,
        feed_forward_config=core_config.feed_forward,
        layer_norm_config=core_config.layer_norm,
        output_norm_config=core_config.output_norm,
        dropout=core_config.dropout,
        dtype=core_config.dtype,
        init_method=core_config.init_method,
        init_seed=core_config.init_seed,
        vocab_size=core_config.vocab_size,
        loss_config=core_config.loss,
        lm_linear_config=lm_linear,
    )

    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    return model


class CoreModelForCausalLM(PreTrainedModel, GenerationMixin):
    """Thin HuggingFace wrapper around :class:`CoreModel`.

    The real model lives at ``self.core_model``; this class only
    implements the HF ``PreTrainedModel`` interface so that
    ``AutoModelForCausalLM.from_pretrained`` and ``model.generate()``
    work out of the box.
    """

    config_class = CoreModelConfig
    supports_gradient_checkpointing = False

    _no_split_modules = []

    def __init__(self, config: CoreModelConfig):
        super().__init__(config)
        core_config = config.to_core_config()
        self.core_model = _build_core_model_without_init(core_config)

    def get_input_embeddings(self) -> nn.Module:
        return self.core_model.embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.core_model.embeddings = value

    def get_output_embeddings(self) -> nn.Module:
        return self.core_model.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.core_model.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = next(self.parameters()).device
        amp_dtype = getattr(self.config, "torch_dtype", None)
        use_amp = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)

        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=use_amp):
            core_out = self.core_model(input_ids=input_ids, labels=labels)

        if not return_dict:
            output = (core_out.logits,)
            return (core_out.loss,) + output if core_out.loss is not None else output

        return CausalLMOutputWithPast(
            loss=core_out.loss,
            logits=core_out.logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        return {"input_ids": input_ids}

    def _init_weights(self, module: nn.Module) -> None:
        pass
