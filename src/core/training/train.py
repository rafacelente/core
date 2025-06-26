import torch
from datasets import load_dataset
from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from torch.profiler import profile, ProfilerActivity
from core.model import CoreConfig
from core.training.lightning_model import CoreLightningModel
from core.training.callbacks.log_callback import LogCallback
from core.training.callbacks.profiler_callback import ProfilerCallback, ThroughputMeasureCallback


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
    def __init__(self, batch_size=2, max_length=512):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("maritaca-ai/sabia-2-tokenizer-small", padding_side="right")
        self.tokenizer.pad_token = "[PAD]"
        print(f"PAD TOKEN ID: {self.tokenizer.pad_token_id}")

    def setup(self, stage=None):
        dataset = load_dataset("wikitext", name="wikitext-2-v1")
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

    use_profiler = True

    data_module = WikiTextDataModule(batch_size=16, max_length=512)

    config = CoreConfig(
        n_layers=24,
        d_model=1024,
        attention=dict(
            n_heads=16,
            use_rope=False,
        ),
        feed_forward=dict(ff_ratio=4),
        layer_norm=dict(eps=1e-5),
        vocab_size=data_module.tokenizer.vocab_size,
        dropout=0.1,
        max_sequence_length=2048,
        pad_token_id=1,
    )

    lightning_model = CoreLightningModel(config, optimizer_type="adam")

    print(lightning_model)

    callbacks = [
        LogCallback(what="steps", every_n=10),
    ]

    if use_profiler:
        def _on_trace_ready(p: profile):
            save_name = f"./trace.json"
            memory_save_name = "./trace_memory.html"
            p.export_chrome_trace(save_name)
            p.export_memory_timeline(memory_save_name)
            print(f"Saving trace: {save_name}")
            print(f"Saving memory trace: {memory_save_name}")

        torch_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            on_trace_ready=_on_trace_ready,
            profile_memory=True,
            with_stack=True,
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=5,
                active=3,
                repeat=1,
            ),
        )

        callbacks.append(ProfilerCallback(prof=torch_profiler))
    else:
        import contextlib
        torch_profiler = contextlib.nullcontext()

    callbacks.append(ThroughputMeasureCallback(
                batch_size=data_module.batch_size,
                num_gpus=1,
                seq_len=data_module.max_length,
                grad_accumulation_steps=1,
            ))
    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        callbacks=callbacks,
    )

    with torch_profiler:
        trainer.fit(lightning_model, data_module)


if __name__ == "__main__":
    main() 