import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import lightning as L
import litdata
import torch
from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage
from google.cloud.exceptions import Forbidden, NotFound
from litdata.streaming.item_loader import ParquetLoader


@dataclass
class TokenizedDatasetMetadata:
    """Metadata for a pre-tokenized dataset uploaded to GCS."""

    tokenizer_name: str
    vocab_size: int
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]
    bos_token_id: Optional[int]
    dataset_name: str
    dataset_sample: str
    max_length: int
    train_size: int
    val_size: Optional[int]
    seed: int
    created_at: str


def _retry_gcs_operation(
    operation,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    operation_name: str = "GCS operation",
):
    """Retry a GCS operation with exponential backoff.

    Args:
        operation: Callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        operation_name: Human-readable name for error messages.

    Returns:
        Result of the operation.

    Raises:
        NotFound: If bucket or blob doesn't exist (not retried).
        Forbidden: If permissions are insufficient (not retried).
        GoogleAPIError: If operation fails after all retries.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except NotFound as e:
            raise NotFound(
                f"{operation_name} failed: Resource not found. "
                f"Check that the bucket exists and the path is correct. "
                f"Original error: {e}"
            ) from e
        except Forbidden as e:
            raise Forbidden(
                f"{operation_name} failed: Permission denied. "
                f"Check that your credentials have access to the bucket. "
                f"Original error: {e}"
            ) from e
        except GoogleAPIError as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                logging.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                raise GoogleAPIError(
                    f"{operation_name} failed after {max_retries + 1} attempts. "
                    f"Last error: {last_exception}"
                ) from last_exception

    raise last_exception


def load_tokenized_dataset_metadata(
    gcs_base_path: str,
    max_retries: int = 3,
) -> TokenizedDatasetMetadata:
    """Load metadata.json from a pre-tokenized dataset on GCS.

    Args:
        gcs_base_path: Base GCS path containing train/, val/, and metadata.json
            (e.g., gs://bucket/pretokenized-fineweb/sample-10BT-gpt2-2048-seed42)
        max_retries: Maximum number of download retry attempts.

    Returns:
        TokenizedDatasetMetadata with tokenizer and dataset info.

    Raises:
        NotFound: If the metadata file or bucket doesn't exist.
        Forbidden: If credentials lack permission to read from the bucket.
        GoogleAPIError: If download fails after all retries.
    """
    gcs_base_path = gcs_base_path.rstrip("/")
    path_without_prefix = gcs_base_path.replace("gs://", "")
    bucket_name = path_without_prefix.split("/")[0]
    blob_path = "/".join(path_without_prefix.split("/")[1:]) + "/metadata.json"

    logging.info(f"Loading metadata from gs://{bucket_name}/{blob_path}...")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        _retry_gcs_operation(
            lambda: blob.download_to_filename(f.name),
            max_retries=max_retries,
            operation_name=f"Download metadata from gs://{bucket_name}/{blob_path}",
        )
        with open(f.name) as metadata_file:
            metadata_dict = json.load(metadata_file)

    return TokenizedDatasetMetadata(**metadata_dict)


class TokenizedFinewebDataset(litdata.StreamingDataset):
    """Streaming dataset for pre-tokenized Fineweb data stored in GCS.

    Supports checkpoint resumption via force_override_state_dict parameter.
    """

    def __init__(
        self,
        dataset_path: str,
        cache_dir: str,
        shuffle: bool = True,
        seed: int = 42,
        max_cache_size: str = "50GB",
        drop_last: bool = True,
        force_override_state_dict: bool = False,
    ):
        super().__init__(
            dataset_path,
            item_loader=ParquetLoader(),
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            force_override_state_dict=force_override_state_dict,
        )


class TokenizedFinewebDataModule(L.LightningDataModule):
    """DataModule for pre-tokenized Fineweb data stored in GCS.

    This module streams pre-tokenized data directly from GCS. Use with data
    prepared by `tokenize_and_upload.py`.

    Args:
        train_dataset_path: GCS path to training data (e.g., gs://bucket/pretokenized-fineweb/sample-10BT-train-gpt2-2048)
        val_dataset_path: GCS path to validation data. If None, uses a subset of train data.
        batch_size: Batch size per device.
        num_workers: Number of dataloader workers.
        cache_dir: Local directory for caching streamed data.
        max_cache_size: Maximum cache size (e.g., "50GB").
        shuffle: Whether to shuffle the training data.
        seed: Random seed for reproducible shuffling.
        resuming: If True, enables force_override_state_dict for checkpoint resumption.
    """

    def __init__(
        self,
        train_dataset_path: str,
        val_dataset_path: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        cache_dir: str = "/mnt/local_storage",
        max_cache_size: str = "50GB",
        shuffle: bool = True,
        seed: int = 42,
        resuming: bool = False,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.shuffle = shuffle
        self.seed = seed
        self.resuming = resuming
        self._train_dataset = None  # Cache to preserve state across calls

    def collate_fn(self, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate batch of tokenized examples into input/target tensors."""
        input_ids = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
        return input_ids, input_ids.clone()

    def _stagger_worker_start(self):
        """Stagger dataset initialization across workers to avoid GCS race conditions."""
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            time.sleep(int(local_rank) * 10)

    def _make_dataloader(
        self, dataset: litdata.StreamingDataset
    ) -> litdata.StreamingDataLoader:
        """Create a StreamingDataLoader with drop_last enforced at the PyTorch DataLoader level."""
        dl = litdata.StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
        object.__setattr__(dl, "drop_last", True)
        return dl

    def train_dataloader(self):
        if self._train_dataset is None:
            self._stagger_worker_start()
            self._train_dataset = TokenizedFinewebDataset(
                dataset_path=self.train_dataset_path,
                cache_dir=f"{self.cache_dir}/fineweb_train",
                shuffle=self.shuffle,
                seed=self.seed,
                max_cache_size=self.max_cache_size,
                drop_last=True,
                force_override_state_dict=self.resuming,
            )

        return self._make_dataloader(self._train_dataset)

    def val_dataloader(self):
        if self.val_dataset_path is None:
            return None

        self._stagger_worker_start()

        dataset = TokenizedFinewebDataset(
            dataset_path=self.val_dataset_path,
            cache_dir=f"{self.cache_dir}/fineweb_val",
            shuffle=False,
            seed=self.seed,
            max_cache_size=self.max_cache_size,
            drop_last=True,
        )
        dataset.load_state_dict = lambda state_dict: None

        return self._make_dataloader(dataset)

