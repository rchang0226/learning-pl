import torch
import torchvision.utils
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
import os
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define discriminator model
class Discriminator(nn.Module):
    def __init__(self, channels, width, height):
        super().__init__()

        self.channels = channels
        self.width = width
        self.height = height

        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * width // 4 * height // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


# define generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, channels, width, height):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64 * width // 4 * height // 4),
            nn.BatchNorm1d(64 * width // 4 * height // 4),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (64, width // 4, height // 4)),

            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class GAN(pl.LightningModule):
    def __init__(
            self,
            channels,
            width,
            height,
            latent_dim: int = 100,
            lr: float = 2e-4,
            b1: float = 0.5,
            b2: float = 0.999
    ):
        super().__init__()
        self.save_hyperparameters()

        self.discriminator = Discriminator(channels, width, height)
        self.generator = Generator(latent_dim, channels, width, height)

        self.validation_z = torch.randn(8, latent_dim)

        self.example_input_array = torch.zeros(2, latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # generate images
            generated_imgs = self(z)

            # log sampled images
            sample_imgs = generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self) -> None:
        z = self.validation_z.type_as(self.generator.model[0].weight)

        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
