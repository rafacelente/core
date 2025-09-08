from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from pathlib import Path
from core.optimizers.optimizer_utils import OptimizerName


@dataclass
class ModelSizeConfig:
    """Pre-defined model size configurations"""
    n_layers: int
    d_model: int
    n_heads: int
    ff_ratio: int = 4

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")


MODEL_SIZES = {
    "small": ModelSizeConfig(n_layers=12, d_model=768, n_heads=12),
    "medium": ModelSizeConfig(n_layers=24, d_model=1024, n_heads=16),
    "large": ModelSizeConfig(n_layers=36, d_model=1280, n_heads=20),
    "xl": ModelSizeConfig(n_layers=48, d_model=1600, n_heads=25),
}


@dataclass
class TrainingConfig:
    # Model configuration
    model_size: str = "small"
    transformer_type: str = "base"
    sequence_length: int = 2048
    vocab_size: Optional[int] = None
    
    # Training configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    dropout: float = 0.0
    max_grad_norm: float = 1.0
    
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
    val_check_interval: Union[int, float] = 0.25
    save_top_k: int = 1
    monitor_metric: str = "val_loss"
    
    # Profiling and debugging
    enable_profiling: bool = False
    enable_model_summary: bool = True
    detect_anomaly: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    def __post_init__(self):
        if self.model_size not in MODEL_SIZES:
            raise ValueError(f"Unknown model size: {self.model_size}. Available: {list(MODEL_SIZES.keys())}")
        
        if self.optimizer not in [opt.value for opt in OptimizerName]:
            raise ValueError(f"Unknown optimizer: {self.optimizer}. Available: {[opt.value for opt in OptimizerName]}")
        
        if self.transformer_type not in ["base", "normalized"]:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")
        
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)