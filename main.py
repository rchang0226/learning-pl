import os
import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchinfo import summary

import classifier
import dataModule
import gan


def train_microsoft_on_mnist():
    data_module = dataModule.DataModule(os.getcwd())
    model = classifier.MicrosoftNetwork(*data_module.dims, data_module.num_classes)
    trainer = pl.Trainer(
        max_epochs=15,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )
    trainer.fit(model, data_module)


def train_gan_on_mnist():
    data_module = dataModule.DataModule(os.getcwd())
    model = gan.GAN(*data_module.dims)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=5,
        callbacks=[TQDMProgressBar(refresh_rate=20)]
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    train_gan_on_mnist()
