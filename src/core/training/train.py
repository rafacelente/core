import torch
from datasets import load_dataset
from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from core.model import CoreConfig
from core.training.lightning_model import CoreLightningModel
from core.training.callbacks.log_callback import LogCallback


class WikiTextDataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item["input_ids"]), torch.tensor(item["labels"])


class WikiTextDataModule(LightningDataModule):
    def __init__(self, batch_size=32, max_length=512):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528", padding_side="right")
        print(f"PAD TOKEN ID: {self.tokenizer.pad_token_id}")

    def setup(self, stage=None):
        dataset = load_dataset("HuggingFaceFW/fineweb-2", name="por_Latn")
        self.train_dataset = self._prepare_dataset(dataset["train"])
        self.val_dataset = self._prepare_dataset(dataset["validation"])
        self.test_dataset = self._prepare_dataset(dataset["test"])

    def _prepare_dataset(self, dataset):
        tokenized_dataset = dataset.map(
            self._tokenize_function, batched=True, remove_columns=["text"]
        )
        return WikiTextDataset(tokenized_dataset, self.max_length)

    def _tokenize_function(self, examples):
        outputs = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return {"input_ids": outputs.input_ids, "labels": outputs.input_ids}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4
        )


def main():

    data_module = WikiTextDataModule(batch_size=16, max_length=512)

    config = CoreConfig(
        n_layers=12,
        d_model=256,
        attention=dict(n_heads=16),
        feed_forward=dict(ff_hidden_size=2048),
        layer_norm=dict(eps=1e-5),
        vocab_size=data_module.tokenizer.vocab_size,
        dropout=0.1,
        max_sequence_length=512,
        pad_token_id=1,
    )

    lightning_model = CoreLightningModel(config)

    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        callbacks=[
            LogCallback(what="steps", every_n=10)
        ]
    )

    trainer.fit(lightning_model, data_module)


if __name__ == "__main__":
    main() 