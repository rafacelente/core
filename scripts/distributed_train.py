#!/usr/bin/env python3

import os
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import lightning as L
from lightning import Trainer
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from lightning.pytorch.profilers import PyTorchProfiler
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import wandb

from core.model import CoreConfig, CoreType
from core.config import AttentionConfig, FeedForwardConfig, LayerNormConfig, FeedForwardType
from core.training.lightning_model import CoreLightningModel
from core.optimizers.optimizer_utils import OptimizerName

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    dropout: float = 0.0
    
    # Optimizer configuration
    optimizer: str = "muon"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler: str = "wsd"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory={"warmup_steps": 2000, "warmup_frac": 0.1, "cooldown_frac": 0.1})
    
    # Data configuration
    dataset_name: str = "openwebtext"
    dataset_config: Optional[str] = None
    data_preprocessing_num_proc: int = 8
    
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
    save_top_k: int = 3
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


class OpenWebTextDataset(Dataset):
    def __init__(self, tokenized_data, sequence_length: int):
        self.data = tokenized_data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        return input_ids, input_ids.clone()


class OpenWebTextDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        sequence_length: int = 2048,
        batch_size: int = 8,
        num_proc: int = 8,
        dataset_name: str = "openwebtext",
        dataset_config: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            logger.info(f"Loading dataset: {self.dataset_name}")
            
            try:
                if self.dataset_config:
                    dataset = load_dataset(self.dataset_name, self.dataset_config)
                else:
                    dataset = load_dataset(self.dataset_name)
            except Exception as e:
                logger.warning(f"Failed to load {self.dataset_name}, falling back to wikitext-2-v1")
                dataset = load_dataset("wikitext", "wikitext-2-v1")
            
            train_size = min(len(dataset["train"]), 100000)
            val_size = min(len(dataset.get("validation", dataset.get("test", dataset["train"]))), 5000)
            
            train_dataset = dataset["train"].select(range(train_size))
            val_dataset = dataset.get("validation", dataset.get("test", dataset["train"])).select(range(val_size))
            
            logger.info(f"Tokenizing {len(train_dataset)} training examples...")
            self.train_dataset = self._prepare_dataset(train_dataset)
            
            logger.info(f"Tokenizing {len(val_dataset)} validation examples...")
            self.val_dataset = self._prepare_dataset(val_dataset)
            
            logger.info("Dataset preparation complete")

    def _prepare_dataset(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.sequence_length,
                return_tensors=None,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        
        return OpenWebTextDataset(tokenized_dataset, self.sequence_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True,
        )


class Model(CoreLightningModel):
    def __init__(self, config: TrainingConfig, model_config: CoreConfig):
        optimizer_kwargs = config.optimizer_kwargs.copy()
        if config.transformer_type == "normalized" and config.optimizer in ["adamw", "adam"]:
            optimizer_kwargs.setdefault("weight_decay", 0.0)
        elif config.transformer_type == "base":
            optimizer_kwargs.setdefault("weight_decay", config.weight_decay)

        lr_scheduler_kwargs = config.lr_scheduler_kwargs.copy()
        if config.lr_scheduler == "cosine_with_warmup":
            lr_scheduler_kwargs.setdefault("warmup_steps", config.warmup_steps)
            lr_scheduler_kwargs.setdefault("max_steps", config.max_steps or 10000)

        super().__init__(
            config=model_config,
            learning_rate=config.learning_rate,
            optimizer_name=OptimizerName(config.optimizer),
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_name=config.lr_scheduler,
            lr_scheduler_params=lr_scheduler_kwargs,
        )
        
        self.config = config
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.automatic_optimization = False
        
        self.train_losses = []
        self.val_losses = []

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        if (batch_idx + 1) % self.gradient_accumulation_steps == 1:
            for optimizer in optimizers:
                optimizer.zero_grad()

        input_ids, labels = batch
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / self.gradient_accumulation_steps
        
        self.manual_backward(loss)
        
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            for optimizer in optimizers:
                optimizer.step()
            
            if hasattr(self.model, "post_optim_step"):
                self.model.post_optim_step()

            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            for scheduler in schedulers:
                scheduler.step()

        actual_loss = loss * self.gradient_accumulation_steps
        self.train_losses.append(actual_loss.item())
        
        self.log("train_loss", actual_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"] if not isinstance(self.optimizers(), list) else self.optimizers()[0].param_groups[0]["lr"], on_step=True, logger=True)
        
        return actual_loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def on_train_epoch_end(self):
        if self.train_losses:
            avg_train_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("epoch_train_loss_avg", avg_train_loss, logger=True)
            self.train_losses.clear()

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_val_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("epoch_val_loss_avg", avg_val_loss, logger=True)
            self.val_losses.clear()


def create_model_config(training_config: TrainingConfig, vocab_size: int) -> CoreConfig:
    """Create model configuration from training config"""
    model_size = MODEL_SIZES[training_config.model_size]
    
    return CoreConfig(
        transformer_type=CoreType(training_config.transformer_type),
        n_layers=model_size.n_layers,
        d_model=model_size.d_model,
        attention=AttentionConfig(
            n_heads=model_size.n_heads,
            dropout=training_config.dropout,
            use_rope=True,
        ),
        feed_forward=FeedForwardConfig(
            feed_forward_type=FeedForwardType.NORMALIZED_GLU if training_config.transformer_type == "normalized" else FeedForwardType.GLU,
            ff_ratio=model_size.ff_ratio,
            dropout=training_config.dropout,
        ),
        layer_norm=LayerNormConfig(eps=1e-5),
        vocab_size=vocab_size,
        dropout=training_config.dropout,
        max_sequence_length=training_config.sequence_length,
        pad_token_id=50256,
    )


def setup_callbacks(config: TrainingConfig) -> List:
    """Setup training callbacks"""
    callbacks = []
    
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
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    device_stats = DeviceStatsMonitor()
    callbacks.append(device_stats)
    
    if config.max_epochs > 5:
        early_stopping = EarlyStopping(
            monitor=config.monitor_metric,
            mode="min",
            patience=3,
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    return callbacks


def setup_logger(config: TrainingConfig) -> WandbLogger:
    """Setup WandB logger with comprehensive config"""
    model_size = MODEL_SIZES[config.model_size]
    
    run_name = config.experiment_name or f"{config.model_size}-{config.optimizer}-{config.transformer_type}"
    
    wandb_config = {
        # Model config
        "model_size": config.model_size,
        "transformer_type": config.transformer_type,
        "n_layers": model_size.n_layers,
        "d_model": model_size.d_model,
        "n_heads": model_size.n_heads,
        "ff_ratio": model_size.ff_ratio,
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
        "warmup_steps": config.warmup_steps,
        
        # Optimizer config
        "optimizer": config.optimizer,
        "optimizer_kwargs": config.optimizer_kwargs,
        "lr_scheduler": config.lr_scheduler,
        
        # Hardware config
        "precision": config.precision,
        "strategy": config.strategy,
        "devices": config.devices,
        "num_nodes": config.num_nodes,
        
        # Data config
        "dataset": config.dataset_name,
        "dataset_config": config.dataset_config,
    }
    
    return WandbLogger(
        project=config.project_name,
        name=run_name,
        config=wandb_config,
        save_dir=str(config.save_dir),
        log_model=False,
    )


def setup_strategy(config: TrainingConfig):
    """Setup distributed training strategy"""
    if config.strategy == "fsdp":
        return FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            activation_checkpointing=None,
            cpu_offload=False,
            mixed_precision=config.precision,
        )
    elif config.strategy == "ddp":
        from lightning.pytorch.strategies import DDPStrategy
        return DDPStrategy(find_unused_parameters=False)
    else:
        return config.strategy


def main():
    parser = argparse.ArgumentParser(description="Distributed training script for GPT-style models")
    parser.add_argument("--model-size", type=str, default="small", choices=list(MODEL_SIZES.keys()),
                       help="Model size configuration")
    parser.add_argument("--optimizer", type=str, default="muon", choices=[opt.value for opt in OptimizerName],
                       help="Optimizer to use")
    parser.add_argument("--transformer-type", type=str, default="normalized", choices=["base", "normalized"],
                       help="Transformer architecture type")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=1, help="Maximum number of epochs")
    parser.add_argument("--max-steps", type=int, help="Maximum number of steps (overrides epochs)")
    parser.add_argument("--sequence-length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--project-name", type=str, default="gpt-optimizer-comparison", help="WandB project name")
    parser.add_argument("--experiment-name", type=str, help="Experiment name for logging")
    parser.add_argument("--save-dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--strategy", type=str, default="fsdp", choices=["fsdp", "ddp", "auto"],
                       help="Distributed strategy")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable PyTorch profiling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(
        model_size=args.model_size,
        optimizer=args.optimizer,
        transformer_type=args.transformer_type,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        sequence_length=args.sequence_length,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision=args.precision,
        enable_profiling=args.enable_profiling,
        seed=args.seed,
    )
    
    logger.info(f"Starting training with configuration: {config}")
    
    L.seed_everything(config.seed, workers=True)
    
    data_module = OpenWebTextDataModule(
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        num_proc=config.data_preprocessing_num_proc,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
    )
    
    data_module.setup("fit")
    vocab_size = data_module.tokenizer.vocab_size
    
    model_config = create_model_config(config, vocab_size)
    logger.info(f"Model config: {model_config}")
    
    model = Model(config, model_config)
    
    total_params = model.model.num_parameters()
    trainable_params = model.model.num_trainable_parameters()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    callbacks = setup_callbacks(config)
    wandb_logger = setup_logger(config)
    strategy = setup_strategy(config)
    
    profiler = None
    if config.enable_profiling:
        profiler = PyTorchProfiler(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
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
        gradient_clip_val=1.0,
    )
    
    try:
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        logger.info("Training completed successfully!")
        
        final_model_path = config.save_dir / "final_model.pt"
        torch.save(model.model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
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
