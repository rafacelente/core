from core.hf.configuration_core import CoreModelConfig
from core.hf.modeling_core import CoreModelForCausalLM

CoreModelConfig.register_for_auto_class()
CoreModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = ["CoreModelConfig", "CoreModelForCausalLM"]
