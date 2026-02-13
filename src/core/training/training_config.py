from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Union
from pathlib import Path

import yaml

from core.optimizers.optimizer_utils import OptimizerName
from core.modules.feed_forward import ActivationType
from core.models.model_recipes import ModelRecipe
from core.optimizations import KernelOptimizations


@dataclass
class TrainingConfig:
    # Model configuration
    model_type: str = "gpt-small"
    transformer_type: str = "base"
    use_post_sdpa_gate: bool = False
    gate_activation_type: ActivationType = ActivationType.SIGMOID
    sequence_length: int = 2048
    vocab_size: Optional[int] = None
    
    # Training configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    dropout: float = 0.0
    max_grad_norm: float = 1.0
    
    # Kernel optimizations (individually toggleable fused kernels)
    fused_rope: bool = False
    fused_cross_entropy: bool = False
    fused_rms_norm: bool = False
    
    # Optimizer configuration
    optimizer: str = "muon"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler: str = "wsd"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {"warmup_frac": 0.1, "cooldown_frac": 0.1})
    
    # Data configuration
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: Optional[str] = "sample-10BT"
    data_preprocessing_num_proc: int = 8
    max_train_size: Optional[int] = None
    max_val_size: Optional[int] = None
    
    # Hardware configuration
    precision: str = "bf16-mixed"
    strategy: str = "fsdp"
    devices: Union[int, str] = "auto"
    num_nodes: int = 1
    
    # Logging and checkpointing
    project_name: str = "muon-8bit"
    experiment_name: Optional[str] = None
    save_dir: str = "./outputs"
    log_every_n_steps: int = 50
    val_check_interval: Union[int, float] = 0.1
    save_top_k: int = 1
    monitor_metric: str = "val_loss"
    log_model: bool = False
    
    # Profiling and debugging
    enable_profiling: bool = False
    enable_model_summary: bool = True
    detect_anomaly: bool = False
    enable_throughput_measurement: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    def __post_init__(self):
        available_recipes = ModelRecipe.get_available_recipes()
        if self.model_type not in available_recipes:
            raise ValueError(f"Unknown model type: {self.model_type}. Available: {available_recipes}")
        
        if self.optimizer not in [opt.value for opt in OptimizerName]:
            raise ValueError(f"Unknown optimizer: {self.optimizer}. Available: {[opt.value for opt in OptimizerName]}")
        
        if self.transformer_type not in ["base", "normalized"]:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")
        
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def kernel_optimizations(self) -> KernelOptimizations:
        """Build a :class:`KernelOptimizations` from the individual flags.

        Uses ``dataclasses.fields`` so that new optimizations added to
        :class:`KernelOptimizations` are picked up automatically as long
        as a matching field exists on this config.
        """
        import dataclasses as dc
        return KernelOptimizations(**{
            f.name: getattr(self, f.name)
            for f in dc.fields(KernelOptimizations)
            if hasattr(self, f.name)
        })

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load a TrainingConfig from a YAML file.

        The YAML file may use a nested structure with the following top-level
        sections that map to groups of fields on this dataclass:

            model:      model_type, transformer_type, use_post_sdpa_gate,
                        gate_activation_type, sequence_length, vocab_size
            training:   batch_size, gradient_accumulation_steps, max_epochs,
                        max_steps, learning_rate, weight_decay, dropout,
                        max_grad_norm
            optimizations: `KernelOptimizations.registered_names()`
            optimizer:  optimizer, optimizer_kwargs, lr_scheduler,
                        lr_scheduler_kwargs
            data:       dataset_name, dataset_config, data_preprocessing_num_proc,
                        max_train_size, max_val_size
            hardware:   precision, strategy, devices, num_nodes
            logging:    project_name, experiment_name, save_dir,
                        log_every_n_steps, val_check_interval, save_top_k,
                        monitor_metric, log_model
            profiling:  enable_profiling, enable_model_summary, detect_anomaly
            reproducibility: seed, deterministic

        Flat key-value pairs at the top level are also accepted and are applied
        directly to the corresponding dataclass fields.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return cls()

        flat: Dict[str, Any] = {}
        valid_field_names = {f.name for f in fields(cls)}

        for key, value in raw.items():
            if isinstance(value, dict):
                # Nested section – merge its children as flat keys
                for sub_key, sub_value in value.items():
                    if sub_key in valid_field_names:
                        flat[sub_key] = sub_value
            else:
                if key in valid_field_names:
                    flat[key] = value

        return cls(**flat)

    @classmethod
    def from_yaml_with_overrides(
        cls,
        path: str,
        overrides: Dict[str, Any],
    ) -> "TrainingConfig":
        """Load from YAML, then apply CLI overrides on top.

        *overrides* should contain only those keys the user explicitly
        provided on the command line (i.e. values that are not at their
        argparse default).  They take precedence over the YAML values.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        flat: Dict[str, Any] = {}
        valid_field_names = {f.name for f in fields(cls)}

        for key, value in raw.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key in valid_field_names:
                        flat[sub_key] = sub_value
            else:
                if key in valid_field_names:
                    flat[key] = value

        # CLI overrides take precedence
        flat.update(overrides)
        return cls(**flat)
