import os

import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning.callbacks import Timer


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

class LitMNIST(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS,batch_size=64, hidden_size=64, learning_rate=1e-4,resolution = 'high',num_classes = 10):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Hardcode some dataset specific attributes
        self.num_classes = num_classes
        if resolution == 'high': 
            self.dims = (1, 28, 28)
            channels, width, height = self.dims
            self.transform = transforms.Compose([
                    transforms.ToTensor()])
        elif resolution == 'low':
            self.dims = (1, 12, 12)
            channels, width, height = self.dims
            self.transform = transforms.Compose(
            [transforms.Resize(12), transforms.ToTensor()])

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()


    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            if self.num_classes==3:
                idx = (mnist_full.targets==0) | (mnist_full.targets==1)  |(mnist_full.targets==2) 
                mnist_full.targets = mnist_full.targets[idx]
                mnist_full.data = mnist_full.data[idx]
                self.mnist_train, self.mnist_val = random_split(mnist_full, [int(len(mnist_full)*0.8), len(mnist_full)-int(len(mnist_full)*0.8)])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            if self.num_classes==3:
                idx = (self.mnist_test.targets==0) | (self.mnist_test.targets==1) |(self.mnist_test.targets==2) 
                self.mnist_test.targets = self.mnist_test.targets[idx]
                self.mnist_test.data = self.mnist_test.data[idx]

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)