"""
@author: Adityam Ghosh
Date: 12/31/2023
"""

from typing import Dict, Any, Tuple, List, Callable, Union, Optional

import numpy as np
import polars as pol
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import pytorch_lightning as pl
import torchmetrics
import albumentations as A
import gc
import argparse

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from models import AlexNet
from dataset_builder import DrivingDataset


class LitModel(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.model = AlexNet(in_channels=in_channels, n_classes=n_actions)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_actions)
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=n_actions
        )
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=n_actions)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=n_actions)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def _common_steps(self, batch: torch.Tensor) -> Dict:
        X, y = batch["img"], batch["label"]
        X /= 255.0
        yhat = self.forward(X)
        loss = self.criterion(yhat, y)

        preds = F.softmax(yhat, dim=-1)

        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1_score(preds, y)

        return {
            "loss": loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        res = self._common_steps(batch=batch)

        self.log(
            f"loss",
            res["loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            f"accuracy",
            res["accuracy"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            f"precision",
            res["precision"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            f"recall",
            res["recall"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            f"f1_score",
            res["f1"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        return res["loss"]

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        res = self._common_steps(batch=batch)

        self.log(
            f"val_loss",
            res["loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            f"val_accuracy",
            res["accuracy"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            f"val_precision",
            res["precision"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            f"val_recall",
            res["recall"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            f"val_f1_score",
            res["f1"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        val_res = dict()

        for k, v in res.items():
            val_res[f"val_{k}"] = v

        return val_res

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.1, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-D",
        required=True,
        type=str,
        help="the folder containing all the training data.",
    )
    parser.add_argument(
        "--model_folder",
        "-M",
        required=False,
        type=str,
        default="./models",
        help="the folder to store the model",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        required=False,
        type=str,
        default="baseline",
        help="name of the model to save with",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        required=False,
        type=int,
        default=32,
        help="the batch size of the data.",
    )
    parser.add_argument(
        "--wandb_log_name",
        "-w",
        required=False,
        type=str,
        default="baseline",
        help="the name with which to log into wandb",
    )
    parser.add_argument(
        "--n_workers",
        "-nw",
        required=False,
        type=int,
        default=4,
        help="the number of parallel workers to create for data processing",
    )

    parser.add_argument(
        "--n_epochs",
        "-ne",
        required=False,
        type=int,
        default=30_000,
        help="the maximum number of epochs to train the model.",
    )

    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        required=False,
        default=1e-3,
        help="the learning rate of the model.",
    )

    parser.add_argument(
        "--accumulate_grad_batches",
        "-agb",
        type=int,
        required=False,
        default=1,
        help="number of batches to accumulate for gradient accumulation.",
    )

    parser.add_argument(
        "--convert_to_grayscale",
        "-g",
        type=int,
        required=False,
        default=0,
        help="whether to convert to grayscale or not.",
    )

    args = parser.parse_args()

    data_path = args.data_path
    model_folder = args.model_folder
    model_name = args.model_name
    batch_size = args.batch_size
    wandb_log_name = args.wandb_log_name
    n_workers = args.n_workers
    n_epochs = args.n_epochs
    lr = args.learning_rate
    accumulate_grad_batches = args.accumulate_grad_batches
    convert_to_grayscale = args.convert_to_grayscale

    in_channels = 1 if convert_to_grayscale else 3

    data = pol.read_parquet(data_path)

    train_data, val_data = train_test_split(
        data,
        test_size=0.001,
        shuffle=True,
        stratify=data.select("label"),
        random_state=32,
    )

    transformation = A.Compose([A.Normalize(always_apply=True)])
    train_dataset = DrivingDataset(
        data=train_data,
        convert_to_grayscale=bool(convert_to_grayscale),
        resize=227,
        transforms=None,
    )
    val_dataset = DrivingDataset(
        data=val_data,
        convert_to_grayscale=bool(convert_to_grayscale),
        resize=227,
        transforms=None,
    )

    train_dataloader = td.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    val_dataloader = td.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    logger = WandbLogger(name=wandb_log_name, project="PyWatchDogs2")
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_folder,
        filename=f"{model_name}_{n_epochs}_{lr}",
        verbose=True,
    )

    prog_bar = RichProgressBar()

    lit_model = LitModel(in_channels=in_channels, n_actions=9, lr=lr)

    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        precision=16,
        logger=logger,
        callbacks=[model_checkpoint, prog_bar],
        max_epochs=n_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
