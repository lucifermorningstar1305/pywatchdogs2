"""
@author: Adityam Ghosh
Date: 12/31/2023
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        self.cnn_model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=96, kernel_size=11, stride=4
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=1),
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=1),
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=1),
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=4096),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=n_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        img_feats = self.cnn_model(X)
        out = self.fc_layer(img_feats)

        return out
