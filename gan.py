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
            nn.Conv2d(3, 64, 3, stride=2, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(3, 64, 3, stride=2, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * width / 4 * height / 4, 1),
            nn.Sigmoid()  # remove this later, using a loss function with sigmoid attached
        )

