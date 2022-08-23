import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define model
class Discriminator(pl.LightningModule):
    def __init__(self, channels, width, height, learning_rate):
        super().__init__()

        self.channels = channels
        self.width = width
        self.height = height
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 64, 3, stride=2, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * width / 4 * height / 4, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.squeeze(dim=1)
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.squeeze(dim=1)
        loss = self.loss_fn(preds, y)
        preds = torch.sigmoid(preds)
        pred_labels = (preds >= 0.5).long()
        acc = accuracy(pred_labels, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        return optimizer
