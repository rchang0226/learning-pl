import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define model
class MicrosoftNetwork(pl.LightningModule):
    def __init__(self, channels, width, height, num_classes, hidden_size=512, learning_rate=1e-3):
        super().__init__()

        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.ReLU()
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
