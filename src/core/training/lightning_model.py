from typing import Tuple

import lightning as L
import torch

from core.optimizers.optimizer_utils import get_optimizer
from core.optimizers.lr_scheduler_utils import LR_SCHEDULER_FUNCTION_MAPPING
from core.model import CoreConfig
from core.training.training_config import TrainingConfig


class CoreLightningModel(L.LightningModule):
    def __init__(
            self,
            config: CoreConfig, 
            training_config: TrainingConfig,
        ):
        super().__init__()
        # Don't save hyperparameters when profiling is enabled to avoid serialization issues
        if not training_config.enable_profiling:
            self.save_hyperparameters()
        self.config = config
        self.model = config.build()
        self.training_config = training_config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, labels = batch
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, labels = batch
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            List[torch.optim.Optimizer]: The optimizers.
        """
        optimizer = get_optimizer(self.training_config.optimizer, self.model, self.training_config.learning_rate, **self.training_config.optimizer_kwargs)

        lr_scheduler = lambda it: LR_SCHEDULER_FUNCTION_MAPPING[self.training_config.lr_scheduler](
            it,
            **self.training_config.lr_scheduler_kwargs,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
        }