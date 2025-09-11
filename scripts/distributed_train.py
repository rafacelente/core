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
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import wandb

from core.model import CoreConfig, CoreType
from core.config import AttentionConfig, FeedForwardConfig, LayerNormConfig, FeedForwardType
from core.training.lightning_model import CoreLightningModel
from core.optimizers.optimizer_utils import OptimizerName
from core.training.training_config import TrainingConfig, MODEL_SIZES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        max_train_size: Optional[int] = None,
        max_val_size: Optional[int] = None,
        enable_profiling: bool = False,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        # Disable multiprocessing when profiling to avoid serialization issues
        self.num_proc = 1 if enable_profiling else num_proc
        if enable_profiling and num_proc > 1:
            logger.info(f"Profiling enabled: reducing num_proc from {num_proc} to 1 to avoid serialization issues")
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_train_size = max_train_size
        self.max_val_size = max_val_size
        self.enable_profiling = enable_profiling
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
            
            if self.max_train_size is not None:
                train_size = min(len(dataset["train"]), self.max_train_size)
            else:
                train_size = len(dataset["train"])
            
            if self.max_val_size is not None:
                val_size = min(len(dataset.get("validation", dataset.get("test", dataset["train"]))), self.max_val_size)
            else:
                val_size = len(dataset.get("validation", dataset.get("test", dataset["train"])))
            
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
    
    if not config.enable_profiling:
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
    
    return callbacks


def setup_logger(
    config: TrainingConfig, 
    num_devices: Optional[int] = None, 
    num_iterations: Optional[int] = None
) -> WandbLogger:
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
    parser.add_argument("--transformer-type", type=str, default="base", choices=["base", "normalized"],
                       help="Transformer architecture type")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size per device")
    parser.add_argument("--max-train-size", type=int, default=None, help="Maximum number of training samples")
    parser.add_argument("--max-val-size", type=int, default=None, help="Maximum number of validation samples")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=1, help="Maximum number of epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum number of steps (overrides epochs)")
    parser.add_argument("--sequence-length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--project-name", type=str, default="muon-8bit", help="WandB project name")
    parser.add_argument("--experiment-name", type=str, help="Experiment name for logging")
    parser.add_argument("--save-dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--strategy", type=str, default="auto", choices=["fsdp", "ddp", "auto"],
                       help="Distributed strategy")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable PyTorch profiling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    accelerator_name = torch.cuda.get_device_name(0)
    accelerator_name = accelerator_name.replace(" ", "_")
    accelerator_count = torch.cuda.device_count()

    if args.experiment_name is None:
        args.experiment_name = f"{args.model_size}-{args.optimizer}-{accelerator_name}-{accelerator_count}"
    
    config = TrainingConfig(
        model_size=args.model_size,
        optimizer=args.optimizer,
        transformer_type=args.transformer_type,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        max_train_size=args.max_train_size,
        max_val_size=args.max_val_size,
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
    
    callbacks = setup_callbacks(config)
    wandb_logger = setup_logger(config, num_devices=num_devices, num_iterations=num_iterations)
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
    )
    
    try:
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        logger.info("Training completed successfully!")
        
        if not config.enable_profiling:
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
