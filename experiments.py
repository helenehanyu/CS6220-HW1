import os
import pandas as pd
import seaborn as sn
import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning.callbacks import Timer
from src.model import LitMNIST
import argparse
import json

def main(args):
    model = LitMNIST(resolution=args.resolution,num_classes=args.num_classes,batch_size=args.batch_size)
    timer = Timer(duration="00:12:00:00")

    trainer = Trainer(
        accelerator="cpu" if args.device == "cpu" else "gpu",
        devices=1 if args.device == "gpu" else None,  # limiting got iPython runs
        max_epochs=10,
        callbacks=[TQDMProgressBar(refresh_rate=20),timer],
        logger=CSVLogger(save_dir=f"logs/{args.resolution}_resolution_{args.num_classes}_classes_{args.device}_batchsize{args.batch_size}/"),
    )
    trainer.fit(model)
    test_results = trainer.test()
    results = test_results[0]
    results['training_time'] = timer.time_elapsed("train")
    with open(f"logs/{args.resolution}_resolution_{args.num_classes}_classes_{args.device}_batchsize{args.batch_size}/results.json", "w") as outfile:
        json.dump(results, outfile, indent=4, sort_keys=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'arg parser')
    parser.add_argument('--resolution',type=str, help='resolution of training imags')
    parser.add_argument('--device',type=str, help='cpu/gpu')
    parser.add_argument('--num_classes',type=int, help='number of classes(K) in the dataset')
    parser.add_argument('--batch_size',type=int)

    args = parser.parse_args()
    main(args)