import os
import torch
from torch import nn, optim, Tensor
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = os.getcwd(), batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            full = MNIST(self.data_dir, train=True, transform=ToTensor())
            self.train_dataset, self.val_dataset = random_split(full, [55000, 5000])
        if stage in (None, "test"):
            self.test_dataset = MNIST(self.data_dir, train=False, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class LitMNISTClassifier(L.LightningModule):
    def __init__(self, hidden_dim: int = 64, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    
    dm = MNISTDataModule(data_dir=os.getcwd(), batch_size=64, num_workers=4)
    model = LitMNISTClassifier(hidden_dim=128, lr=1e-3)

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )

    # train + val
    trainer.fit(model, datamodule=dm)
    # test
    trainer.test(model, datamodule=dm)