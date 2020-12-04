import pytorch_lightning as pl
import torch

from pytorch_lightning import Trainer
from torch.optim import AdamW
from torch.utils.data import DataLoader


class LightningBase(pl.LightningModule):
    def __init__(
            self,
            model_save_path: str,
            max_len: int,
            batch_size: int,
            num_gpus: int,
            lr: float = 3e-5,
            weight_decay: float = 1e-4,
            save_step_interval: int = 1000,
            accelerator: str = "ddp",
            precision: int = 16,
            use_amp: bool = True,
    ) -> None:
        """constructor of LightningBase"""

        super().__init__()
        self.model_save_path = model_save_path
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.num_gpus = num_gpus
        self.save_step_interval = save_step_interval
        self.accelerator = accelerator
        self.precision = precision
        self.use_amp = use_amp
        self.model = None

    def configure_optimizers(self):
        """configure optimizers and lr schedulers"""
        no_decay = ["bias", "LayerNorm.weight"]
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return [self.optimizer]

    def fit(self, train_dataloader: DataLoader):
        trainer = Trainer(
            gpus=self.num_gpus,
            distributed_backend=self.accelerator,
            precision=self.precision,
            amp_backend="apex" if self.use_amp else None,
        )

        trainer.fit(
            model=self, train_dataloader=train_dataloader,
        )

    def save_model(self) -> None:
        if (
                self.trainer.global_rank == 0
                and self.global_step % self.save_step_interval == 0
        ):
            torch.save(
                self.model.state_dict(),
                self.model_save_path + "." + str(self.global_step),
            )
