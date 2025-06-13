from typing import Any, Dict, Optional
import logging


import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from torch.profiler import profile


class ProfilerCallback(Callback):
    def __init__(
        self,
        prof: profile,
    ):
        self.prof = prof
        self.step_count = 0

    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.step_count += 1
        self.prof.step()


class ThroughputMeasureCallback(Callback):
    def __init__(
        self,
        num_gpus: int,
        batch_size: int,
        grad_accumulation_steps: int,
        seq_len: int,
        log_dict: Optional[Dict[str, Any]] = None,
        starting_step: int = 30,
    ):
        self._initialized = False
        assert batch_size is not None, "Batch size must be provided to measure throughput"
        assert num_gpus is not None, "Number of GPUs must be provided to measure throughput"
        assert seq_len is not None, "Sequence length must be provided to measure throughput"
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.seq_len = seq_len
        self.grad_accumulation_steps = grad_accumulation_steps
        self.log_dict = log_dict
        self.step_count = 0
        self.starting_step = starting_step
        self.measure_started = False

    @rank_zero_only
    def setup(self, args, state, control):
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        logging.warning("Profiling callback initialized")
        self._initialized = True

    @rank_zero_only
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.ender.record()  # type: ignore
        torch.cuda.synchronize()
        counted_steps = self.step_count - self.starting_step
        elapsed_time = self.starter.elapsed_time(self.ender)
        time_per_step = elapsed_time / counted_steps
        samples_per_step = self.batch_size * self.num_gpus * self.grad_accumulation_steps  # type: ignore
        throughput = samples_per_step / (time_per_step / 1000)

        logging.warning(f"Elapsed time: {elapsed_time}")
        logging.warning(f"Total optimizer steps: {self.step_count} (counted: {counted_steps})")
        logging.warning(f"Time taken per optimizer step: {time_per_step}")
        logging.warning(f"Throughput: {throughput} samples/sec")

        max_allocated_memory = torch.cuda.max_memory_allocated()
        logging.warning(f"Max allocated memory: {max_allocated_memory}")


    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self._initialized:
            self.setup(trainer, pl_module)
        if self.starting_step >= self.step_count and not self.measure_started:
            self.starter.record()  # type: ignore
            self.measure_started = True
        self.step_count += 1