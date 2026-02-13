import os
import logging
import argparse
from typing import List, Optional, Union

import torch
import lightning as L
from lightning import Trainer
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.profilers import PyTorchProfiler
import wandb

from core.models.model_config import CoreConfig
from core.training.lightning_model import CoreLightningModel
from core.optimizers.optimizer_utils import OptimizerName
from core.training.training_config import TrainingConfig
from core.models.model_recipes import ModelRecipe
from core.training.data.fineweb import FineWebDataModule
from core.training.callbacks.profiler_callback import ThroughputMeasureCallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_model_config(training_config: TrainingConfig, vocab_size: int) -> CoreConfig:
    """Create model configuration from training config using the model recipe registry.
    """
    recipe = ModelRecipe.get_recipe(training_config.model_type)
    config = recipe.build_config(
        vocab_size=vocab_size,
        max_sequence_length=training_config.sequence_length,
        dropout=training_config.dropout,
        transformer_type=training_config.transformer_type,
        use_post_sdpa_gate=training_config.use_post_sdpa_gate,
        gate_activation_type=training_config.gate_activation_type,
    )
    optimizations = training_config.kernel_optimizations
    if optimizations.any_enabled():
        config = config.with_kernel_optimizations(optimizations)
        logger.info(f"Applied kernel optimizations: {optimizations}")
    if config.vocab_size != vocab_size:
        logger.info(
            f"Vocab size padded from {vocab_size} to {config.vocab_size} "
            f"(required by fused cross-entropy kernel)"
        )
    return config


def setup_callbacks(config: TrainingConfig, num_devices: Optional[int] = None) -> List:
    """Setup training callbacks"""
    callbacks = []
    
    if not (config.enable_profiling or config.enable_throughput_measurement) and config.log_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.save_dir / "checkpoints",
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor=config.monitor_metric,
            mode="min",
            save_top_k=config.save_top_k,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
    
    if config.enable_throughput_measurement:
        assert num_devices is not None, "Number of devices must be provided to measure throughput"
        throughput_callback = ThroughputMeasureCallback(
            num_gpus=num_devices,
            batch_size=config.batch_size,
            grad_accumulation_steps=config.gradient_accumulation_steps,
            seq_len=config.sequence_length,
        )
        callbacks.append(throughput_callback)
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(
    config: TrainingConfig, 
    model_config: CoreConfig,
    tokenizer_vocab_size: int,
    num_devices: Optional[int] = None, 
    num_iterations: Optional[int] = None,
) -> WandbLogger:
    """Setup WandB logger with comprehensive config"""
    run_name = config.experiment_name or f"{config.model_type}-{config.optimizer}-{config.transformer_type}"
    
    wandb_config = {
        # Model config
        "model_type": config.model_type,
        "transformer_type": config.transformer_type,
        "n_layers": model_config.n_layers,
        "d_model": model_config.d_model,
        "n_heads": model_config.attention.n_heads,
        "vocab_size": model_config.vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "sequence_length": config.sequence_length,
        "dropout": config.dropout,
        
        # Training config
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_epochs": config.max_epochs,
        "max_steps": config.max_steps,
        
        # Optimizer config
        "optimizer": config.optimizer,
        "optimizer_kwargs": config.optimizer_kwargs,
        "lr_scheduler": config.lr_scheduler,
        "lr_scheduler_kwargs": config.lr_scheduler_kwargs,
        
        # Hardware config
        "precision": config.precision,
        "strategy": config.strategy,
        "devices": config.devices,
        "num_nodes": config.num_nodes,
        
        # Data config
        "dataset": config.dataset_name,
        "dataset_config": config.dataset_config,
        "max_train_size": config.max_train_size,
        "max_val_size": config.max_val_size,
    }
    
    if num_devices is not None:
        wandb_config["num_devices"] = num_devices
        wandb_config["effective_batch_size_total"] = config.batch_size * config.gradient_accumulation_steps * num_devices
    
    if num_iterations is not None:
        wandb_config["num_iterations"] = num_iterations
    
    return WandbLogger(
        project=config.project_name,
        name=run_name,
        config=wandb_config,
        save_dir=str(config.save_dir),
        log_model=True,
    )


def calculate_num_iterations(
    dataset_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_devices: int,
    max_epochs: int,
    max_steps: int = -1,
) -> int:
    """Calculate the total number of optimizer steps for training.
    
    Args:
        dataset_size: Total number of samples in the training dataset
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_devices: Total number of devices (GPUs) being used
        max_epochs: Maximum number of epochs
        max_steps: Maximum number of steps (if > 0, overrides epoch-based calculation)
        
    Returns:
        Total number of optimizer steps
    """
    effective_batch_size = batch_size * gradient_accumulation_steps * num_devices
    steps_per_epoch = dataset_size // effective_batch_size
    
    if max_steps > 0:
        epoch_based_steps = steps_per_epoch * max_epochs
        total_steps = min(max_steps, epoch_based_steps)
    else:
        total_steps = steps_per_epoch * max_epochs
    
    logger.info(f"Training steps calculation:")
    logger.info(f"  Dataset size: {dataset_size:,}")
    logger.info(f"  Effective batch size: {effective_batch_size} (batch_size={batch_size} × grad_accum={gradient_accumulation_steps} × devices={num_devices})")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Max epochs: {max_epochs}")
    if max_steps > 0:
        logger.info(f"  Max steps limit: {max_steps}")
    logger.info(f"  Total training steps: {total_steps}")
    
    return total_steps


def get_num_devices(devices: Union[int, str], num_nodes: int) -> int:
    """Get the total number of devices being used for training."""
    if isinstance(devices, int):
        return devices * num_nodes
    elif devices == "auto":
        if torch.cuda.is_available():
            return torch.cuda.device_count() * num_nodes
        else:
            return 1 * num_nodes
    elif devices == "-1":
        if torch.cuda.is_available():
            return torch.cuda.device_count() * num_nodes
        else:
            return 1 * num_nodes
    else:
        devices_str = str(devices).strip("[]")
        device_list = [d.strip() for d in devices_str.split(",") if d.strip()]
        return len(device_list) * num_nodes


def setup_strategy(config: TrainingConfig):
    """Setup distributed training strategy"""
    if config.strategy == "fsdp":
        return FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            activation_checkpointing=None,
            cpu_offload=False,
        )
    elif config.strategy == "ddp":
        from lightning.pytorch.strategies import DDPStrategy
        return DDPStrategy(find_unused_parameters=False)
    else:
        return config.strategy


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training script.

    All CLI arguments use ``default=None`` (except boolean flags which use
    ``store_true``) so that we can distinguish between "user explicitly set
    this" and "user left the default" when merging with a YAML config file.
    """
    parser = argparse.ArgumentParser(description="Distributed training script for GPT-style models")

    # YAML configuration file ---------------------------------------------------
    parser.add_argument("--config", type=str, default=None,
                       help="Path to a YAML configuration file. CLI arguments override YAML values.")

    # Model configuration -------------------------------------------------------
    parser.add_argument("--model-type", type=str, default=None, choices=ModelRecipe.get_available_recipes(),
                       help="Model type configuration (default: gpt-small)")
    parser.add_argument("--optimizer", type=str, default=None, choices=[opt.value for opt in OptimizerName],
                       help="Optimizer to use (default: muon)")
    parser.add_argument("--transformer-type", type=str, default=None, choices=["base", "normalized"],
                       help="Transformer architecture type (default: base)")
    parser.add_argument("--use-post-sdpa-gate", action="store_true", default=None, help="Use post SDPA gate")
    parser.add_argument("--gate-activation-type", type=str, default=None, choices=["sigmoid", "gelu", "relu"],
                       help="Gate activation type (default: sigmoid)")

    # Training configuration ----------------------------------------------------
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per device (default: 12)")
    parser.add_argument("--max-train-size", type=int, default=None, help="Maximum number of training samples")
    parser.add_argument("--max-val-size", type=int, default=None, help="Maximum number of validation samples")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                       help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (default: 3e-4)")
    parser.add_argument("--max-epochs", type=int, default=None, help="Maximum number of epochs (default: 1)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps (overrides epochs)")
    parser.add_argument("--sequence-length", type=int, default=None, help="Sequence length (default: 2048)")

    # Kernel optimizations ------------------------------------------------------
    parser.add_argument("--fused-rope", action="store_true", default=None,
                       help="Use fused Triton RoPE kernel")
    parser.add_argument("--fused-cross-entropy", action="store_true", default=None,
                       help="Use fused cross-entropy loss kernel")
    parser.add_argument("--fused-rms-norm", action="store_true", default=None,
                       help="Use fused RMSNorm kernel (requires SM100+)")

    # Logging and checkpointing -------------------------------------------------
    parser.add_argument("--project-name", type=str, default=None, help="WandB project name (default: muon-8bit)")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name for logging")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save outputs (default: ./outputs)")
    parser.add_argument("--log-model", action="store_true", default=None, help="Log model")
    parser.add_argument("--log-every-n-steps", type=int, default=None, help="Log every n steps (default: 50)")
    parser.add_argument("--val-check-interval", type=float, default=None,
                       help="Validation check interval (default: 0.1)")
    parser.add_argument("--save-top-k", type=int, default=None, help="Save top k checkpoints (default: 1)")
    parser.add_argument("--monitor-metric", type=str, default=None, help="Metric to monitor (default: val_loss)")

    # Hardware configuration ----------------------------------------------------
    parser.add_argument("--devices", type=str, default=None, help="Devices to use (default: auto)")
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes (default: 1)")
    parser.add_argument("--strategy", type=str, default=None, choices=["fsdp", "ddp", "auto"],
                       help="Distributed strategy (default: auto)")
    parser.add_argument("--precision", type=str, default=None, help="Training precision (default: bf16-mixed)")

    # Profiling and debugging ---------------------------------------------------
    parser.add_argument("--enable-profiling", action="store_true", default=None, help="Enable PyTorch profiling")
    parser.add_argument("--enable-model-summary", action="store_true", default=None, help="Enable model summary")
    parser.add_argument("--detect-anomaly", action="store_true", default=None, help="Detect anomalies")
    parser.add_argument("--enable-throughput-measurement", action="store_true", default=None, help="Enable throughput measurement")

    # Reproducibility -----------------------------------------------------------
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument("--deterministic", action="store_true", default=None, help="Deterministic training")

    return parser


_CLI_ARG_TO_CONFIG_FIELD = {
    "model_type": "model_type",
    "optimizer": "optimizer",
    "transformer_type": "transformer_type",
    "use_post_sdpa_gate": "use_post_sdpa_gate",
    "gate_activation_type": "gate_activation_type",
    "batch_size": "batch_size",
    "max_train_size": "max_train_size",
    "max_val_size": "max_val_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "learning_rate": "learning_rate",
    "max_epochs": "max_epochs",
    "max_steps": "max_steps",
    "sequence_length": "sequence_length",
    "project_name": "project_name",
    "experiment_name": "experiment_name",
    "save_dir": "save_dir",
    "log_model": "log_model",
    "log_every_n_steps": "log_every_n_steps",
    "val_check_interval": "val_check_interval",
    "save_top_k": "save_top_k",
    "monitor_metric": "monitor_metric",
    "devices": "devices",
    "num_nodes": "num_nodes",
    "strategy": "strategy",
    "precision": "precision",
    "enable_profiling": "enable_profiling",
    "enable_model_summary": "enable_model_summary",
    "detect_anomaly": "detect_anomaly",
    "enable_throughput_measurement": "enable_throughput_measurement",
    "fused_rope": "fused_rope",
    "fused_cross_entropy": "fused_cross_entropy",
    "fused_rms_norm": "fused_rms_norm",
    "seed": "seed",
    "deterministic": "deterministic",
}


def _get_explicit_cli_overrides(args: argparse.Namespace) -> dict:
    """Return only the CLI arguments that the user explicitly provided.

    Since every argument has ``default=None``, any non-None value was
    explicitly passed on the command line.
    """
    overrides = {}
    for cli_name, config_field in _CLI_ARG_TO_CONFIG_FIELD.items():
        value = getattr(args, cli_name, None)
        if value is not None:
            overrides[config_field] = value
    return overrides


def main():
    parser = _build_parser()
    args = parser.parse_args()

    cli_overrides = _get_explicit_cli_overrides(args)

    if args.config is not None:
        logger.info(f"Loading configuration from YAML file: {args.config}")
        config = TrainingConfig.from_yaml_with_overrides(args.config, cli_overrides)
    elif cli_overrides:
        config = TrainingConfig(**cli_overrides)
    else:
        config = TrainingConfig()

    accelerator_name = torch.cuda.get_device_name(0)
    accelerator_name = accelerator_name.replace(" ", "_")
    accelerator_count = torch.cuda.device_count()

    if config.experiment_name is None:
        config.experiment_name = f"{config.model_type}-{config.optimizer}-{accelerator_name}-{accelerator_count}"
    
    logger.info(f"Starting training with configuration: {config}")
    
    L.seed_everything(config.seed, workers=True)
    
    data_module = FineWebDataModule(
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        num_proc=config.data_preprocessing_num_proc,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        max_train_size=config.max_train_size,
        max_val_size=config.max_val_size,
        enable_profiling=config.enable_profiling,
    )
    
    data_module.setup("fit")
    vocab_size = data_module.tokenizer.vocab_size
    
    # Calculate num_iterations for learning rate schedulers that need it
    num_devices = get_num_devices(config.devices, config.num_nodes)
    dataset_size = len(data_module.train_dataset)
    
    num_iterations = calculate_num_iterations(
        dataset_size=dataset_size,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_devices=num_devices,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
    )
    
    updated_lr_scheduler_kwargs = config.lr_scheduler_kwargs.copy()
    if config.lr_scheduler in ["wsd", "cosine_with_warmup"]:
        updated_lr_scheduler_kwargs["num_iterations"] = num_iterations
        logger.info(f"Added num_iterations={num_iterations} to {config.lr_scheduler} scheduler kwargs")
    
    model_config = create_model_config(config, vocab_size)
    logger.info(f"Model config: {model_config}")
    
    from dataclasses import replace
    model_training_config = replace(config, lr_scheduler_kwargs=updated_lr_scheduler_kwargs)
    
    model = CoreLightningModel(training_config=model_training_config, config=model_config)
    
    total_params = model.model.num_parameters()
    trainable_params = model.model.num_trainable_parameters()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    callbacks = setup_callbacks(config, num_devices=num_devices)
    wandb_logger = setup_logger(
        config,
        model_config=model_config,
        tokenizer_vocab_size=vocab_size,
        num_devices=num_devices,
        num_iterations=num_iterations,
    )
    strategy = setup_strategy(config)
    
    profiler = None
    if config.enable_profiling:
        profiler = PyTorchProfiler(
            schedule=torch.profiler.schedule(wait=10, warmup=5, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(config.save_dir / "profiler")),
            record_shapes=True,
            profile_memory=True,
        )
    
    trainer = Trainer(
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        accelerator="auto",
        devices=config.devices,
        num_nodes=config.num_nodes,
        strategy=strategy,
        precision=config.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        profiler=profiler,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        enable_model_summary=config.enable_model_summary,
        deterministic=config.deterministic,
        detect_anomaly=config.detect_anomaly,
        num_sanity_val_steps=0,
    )
    
    try:
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        logger.info("Training completed successfully!")
        
        if wandb.run:
            wandb.log({
                "final_train_loss": trainer.logged_metrics.get("train_loss", 0),
                "final_val_loss": trainer.logged_metrics.get("val_loss", 0),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            })
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
