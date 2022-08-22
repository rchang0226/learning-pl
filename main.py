import os
import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks.progress import TQDMProgressBar

import classifier
import dataModule


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


if __name__ == '__main__':
    train_microsoft_on_mnist()
