import pytorch_lightning as pl
from torchvision import datasets, transforms
from torchvision import datasets
import os
from torch.utils.data import DataLoader, random_split


class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir):
        super().__init__()
        self.test = None
        self.val = None
        self.train = None
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # mean + std
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.num_workers = os.cpu_count() // 2

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = random_split(full, [55000, 5000])

        if stage == "test" or stage is None:
            self.test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=self.num_workers)
