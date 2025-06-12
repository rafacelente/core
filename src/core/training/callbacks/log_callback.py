import logging

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class LogCallback(Callback):
    """A callback to log to stdout the training progress."""

    def __init__(self, what: str = "steps", every_n: int = 100):
        """
        Initialize the LogCallback.
        Args:
            what (str): Whether to log at the step level or the epoch level.
            every_n (int): The frequency at which to log.
        """
        super().__init__()
        self.what = what
        self.every_n = every_n

    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx):
        """Log the training progress at the end of each batch."""
        if self.what == "steps":
            if batch_idx % self.every_n == 0:
                train_loss = trainer.callback_metrics.get("train_loss", None)
                if train_loss is not None:
                    print(
                        f"Step: {trainer.global_step} | Loss: {train_loss:.4f} | Batch: {batch_idx} | "
                        + f"LR: {trainer.optimizers[0].param_groups[0]['lr']:.6f}"
                    )
        elif self.what == "epoch":
            if trainer.current_epoch % self.every_n == 0:
                train_loss = trainer.callback_metrics.get("train_loss", None)
                if train_loss is not None:
                    print(
                        f"Epoch: {trainer.current_epoch} | Loss: {train_loss:.4f} | "
                        + f"LR: {trainer.optimizers[0].param_groups[0]['lr']:.6f}"
                    )
                else:
                    print(
                        f"Epoch: {trainer.current_epoch} | " + f"LR: {trainer.optimizers[0].param_groups[0]['lr']:.6f}"
                    )