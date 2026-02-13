import os
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import lightning as L

logger = logging.getLogger(__name__)

# When the dataset has no dedicated validation/test split and we must carve
# validation data out of the training split, reserve this fraction by default.
_DEFAULT_VAL_FRACTION = 0.05

class FineWebDataset(Dataset):
    def __init__(self, tokenized_data, sequence_length: int):
        self.data = tokenized_data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        return input_ids, input_ids.clone()


class FineWebDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        sequence_length: int = 2048,
        batch_size: int = 8,
        num_proc: int = 8,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
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
            if hasattr(self, "train_dataset") and hasattr(self, "val_dataset"):
                logger.info("Dataset already loaded")
                return
            
            logger.info(f"Loading dataset: {self.dataset_name}")
            
            try:
                if self.dataset_config:
                    dataset = load_dataset(self.dataset_name, self.dataset_config)
                else:
                    dataset = load_dataset(self.dataset_name)
            except Exception as e:
                logger.warning(f"Failed to load {self.dataset_name}, falling back to wikitext-2-v1")
                dataset = load_dataset("wikitext", "wikitext-2-v1")
            
            full_train = dataset["train"]
            val_split = dataset.get("validation", dataset.get("test"))
            has_dedicated_val = val_split is not None

            if has_dedicated_val:
                train_size = min(len(full_train), self.max_train_size) if self.max_train_size is not None else len(full_train)
                val_size = min(len(val_split), self.max_val_size) if self.max_val_size is not None else len(val_split)

                train_dataset = full_train.select(range(train_size))
                val_dataset = val_split.select(range(val_size))
            else:
                # fallback: carve val out of the train split
                logger.warning(
                    f"Dataset '{self.dataset_name}' has no 'validation' or 'test' split. "
                    f"Splitting the 'train' split into non-overlapping train/val partitions."
                )
                total = len(full_train)

                if self.max_train_size is not None and self.max_val_size is not None:
                    train_size = min(total, self.max_train_size)
                    val_size = min(total - train_size, self.max_val_size)
                elif self.max_train_size is not None:
                    train_size = min(total, self.max_train_size)
                    remaining = total - train_size
                    val_size = min(remaining, max(1, int(total * _DEFAULT_VAL_FRACTION)))
                elif self.max_val_size is not None:
                    val_size = min(total, self.max_val_size)
                    train_size = total - val_size
                else:
                    val_size = max(1, int(total * _DEFAULT_VAL_FRACTION))
                    train_size = total - val_size

                if val_size <= 0:
                    raise ValueError(
                        f"Not enough data to create a validation set from the 'train' split. "
                        f"Total samples: {total}, requested train_size: {train_size}. "
                        f"Reduce max_train_size or provide a dataset with a dedicated validation split."
                    )

                logger.info(
                    f"Train/val split from 'train': {train_size} train, {val_size} val "
                    f"(total available: {total})"
                )

                train_dataset = full_train.select(range(train_size))
                val_dataset = full_train.select(range(train_size, train_size + val_size))
            
            logger.info(f"Tokenizing {len(train_dataset)} training examples...")
            self.train_dataset = self._prepare_dataset(train_dataset)
            
            logger.info(f"Tokenizing {len(val_dataset)} validation examples...")
            self.val_dataset = self._prepare_dataset(val_dataset)
            
            logger.info(f"Dataset preparation complete")

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
        
        return FineWebDataset(tokenized_dataset, self.sequence_length)

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