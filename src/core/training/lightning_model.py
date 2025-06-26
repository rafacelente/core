import lightning as L
import torch

from core.training.utils import get_optimizer

from core.model import CoreConfig


class CoreLightningModel(L.LightningModule):
    def __init__(self, config: CoreConfig, learning_rate: float = 1e-4, optimizer_type = "muon"):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = config.build()
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.automatic_optimization = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list): optimizers = [optimizers]

        for opt in optimizers:
            opt.zero_grad()

        input_ids, labels = batch
        outputs = self.forward(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)

        # backward
        self.manual_backward(loss)

        # step every optimiser
        for opt in optimizers:
            opt.step()

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self.model(input_ids=input_ids, labels=labels)
        self.log("val_loss", outputs.loss)

    def configure_optimizers(self):
        """Configure optimizers with Muon for 2-D weight matrices and Adam for the rest.
        """
        optimizers = get_optimizer(self.optimizer_type, self.model, self.learning_rate)
        if isinstance(optimizers, list):
            return optimizers
        return [optimizers]
