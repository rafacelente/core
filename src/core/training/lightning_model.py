from typing import Any, Dict
import lightning as L
import torch
from core.optimizers.optimizer_utils import OptimizerName, get_optimizer
from core.optimizers.lr_scheduler_utils import LR_SCHEDULER_FUNCTION_MAPPING

from core.model import CoreConfig


class CoreLightningModel(L.LightningModule):
    def __init__(
            self,
            config: CoreConfig, 
            learning_rate: float = 1e-4,
            optimizer_name: OptimizerName = OptimizerName.ADAMW,
            optimizer_kwargs: Dict[str, Any] = {},
            lr_scheduler_name: str = "constant",
            lr_scheduler_params: Dict[str, Any] = {},
        ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = config.build()
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_params = lr_scheduler_params
        self.automatic_optimization = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for optimizer in optimizers:
            optimizer.zero_grad()
        
        input_ids, labels = batch
        outputs = self.forward(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, rank_zero_only=True)
        self.manual_backward(loss)
        
        for optimizer in optimizers:
            optimizer.step()
        
        if hasattr(self.model, "post_optim_step"):
            self.model.post_optim_step()

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self.model(input_ids=input_ids, labels=labels)
        self.log("val_loss", outputs.loss)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            List[torch.optim.Optimizer]: The optimizers.
        """
        optimizer = get_optimizer(self.optimizer_name, self.model, self.learning_rate, **self.optimizer_kwargs)

        # TODO: Add support for applying different schedulers to different optimizers.
        # Right now, all optimizers will use the same scheduler.
        if isinstance(optimizer, list):
            opt_lr_schedulers = []
            for opt in optimizer:
                lr_scheduler = lambda it: LR_SCHEDULER_FUNCTION_MAPPING[self.lr_scheduler_name](
                    it,
                    **self.lr_scheduler_params,
                )
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scheduler)
                opt_lr_schedulers.append({"optimizer": opt, "lr_scheduler": lr_scheduler})
            return opt_lr_schedulers
        else:
            lr_scheduler = lambda it: LR_SCHEDULER_FUNCTION_MAPPING[self.lr_scheduler_name](
                it,
                **self.lr_scheduler_params,
            )
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)
            return [{"optimizer": optimizer, "lr_scheduler": lr_scheduler}]