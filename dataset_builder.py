"""
@author: Adityam Ghosh
Date: 12/31/2023
"""

from typing import Dict, Any, Tuple, List, Callable, Union, Optional

import numpy as np
import torch
import torch.utils.data as td
import albumentations as A
import cv2
import polars as pl

from PIL import Image


class DrivingDataset(td.Dataset):
    def __init__(
        self,
        data: pl.DataFrame,
        convert_to_grayscale: bool,
        resize: int | Tuple = 227,
        transforms: Optional[A.Compose] = None,
    ):
        self.data = data
        self.convert_to_grayscale = convert_to_grayscale
        self.resize = (resize, resize) if isinstance(resize, int) else resize
        self.transforms = transforms

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        img_file = self.data.item(idx, "img_file")
        label = self.data.item(idx, "label")

        if self.convert_to_grayscale:
            img = Image.open(img_file).convert("L")
        else:
            img = Image.open(img_file).convert("RGB")

        img = img.resize(self.resize, resample=Image.Resampling.NEAREST)

        img = np.array(img)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        if self.transforms is not None:
            img_obj = self.transforms(image=img)
            img = img_obj["image"]

        img = np.transpose(img, (2, 0, 1))  # (C, H, W)

        img = torch.from_numpy(img).float()
        label = torch.tensor(label, dtype=torch.long)

        return {"img": img, "label": label}
