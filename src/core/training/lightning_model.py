import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from core.model import CoreConfig


class CoreLightningModel(L.LightningModule):
    def __init__(self, config: CoreConfig, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = config.build()
        self.learning_rate = learning_rate

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self.model(input_ids=input_ids, labels=labels)
        self.log("train_loss", outputs.loss, on_step=True, on_epoch=False, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self.model(input_ids=input_ids, labels=labels)
        self.log("val_loss", outputs.loss)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer