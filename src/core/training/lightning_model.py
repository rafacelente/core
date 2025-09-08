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
        self.save_hyperparameters()
        self.config = config
        self.model = config.build()
        self.training_config = training_config
        self.automatic_optimization = False
        self.train_losses = []
        self.val_losses = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]


        if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 1:
            for optimizer in optimizers:
                optimizer.zero_grad()
        
        input_ids, labels = batch
        outputs = self.forward(input_ids=input_ids, labels=labels)
        loss = outputs.loss / self.training_config.gradient_accumulation_steps
        self.manual_backward(loss)
        
        if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.training_config.max_grad_norm)
            for optimizer in optimizers:
                optimizer.step()
        
            if hasattr(self.model, "post_optim_step"):
                self.model.post_optim_step()

            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            for scheduler in schedulers:
                scheduler.step()

        actual_loss = loss * self.training_config.gradient_accumulation_steps
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

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            List[torch.optim.Optimizer]: The optimizers.
        """
        optimizer = get_optimizer(self.training_config.optimizer, self.model, self.training_config.learning_rate, **self.training_config.optimizer_kwargs)

        # TODO: Add support for applying different schedulers to different optimizers.
        # Right now, all optimizers will use the same scheduler.
        if isinstance(optimizer, list):
            opt_lr_schedulers = []
            for opt in optimizer:
                lr_scheduler = lambda it: LR_SCHEDULER_FUNCTION_MAPPING[self.training_config.lr_scheduler](
                    it,
                    **self.training_config.lr_scheduler_kwargs,
                )
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scheduler)
                opt_lr_schedulers.append({"optimizer": opt, "lr_scheduler": lr_scheduler})
            return opt_lr_schedulers
        else:
            lr_scheduler = lambda it: LR_SCHEDULER_FUNCTION_MAPPING[self.training_config.lr_scheduler](
                it,
                **self.training_config.lr_scheduler_kwargs,
            )
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)
            return [{"optimizer": optimizer, "lr_scheduler": lr_scheduler}]