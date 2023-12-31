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
        path: str,
        resize: int | Tuple = 227,
        transforms: Optional[A.Compose] = None,
    ):
        self.path = path
        self.resize = (resize, resize) if isinstance(resize, int) else resize
        self.transforms = transforms

        self.data = self._load_data(path)

    def _load_data(self, path: str) -> np.ndarray:
        data = pl.read_parquet(path)
        return data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        img_file = self.data.item(idx, "img_file")
        label = self.data.item(idx, "label")

        img = Image.open(img_file).convert("RGB")

        img = img.resize(self.resize, resample=Image.Resampling.NEAREST)

        img = np.array(img)

        if self.transforms is not None:
            img_obj = self.transforms(image=img)
            img = img_obj["image"]

        else:
            img = img.astype(np.float32)
            img /= 255.0

        img = np.transpose(img, (2, 0, 1))  # (C, H, W)

        img = torch.from_numpy(img).float()
        label = torch.tensor(label, dtype=torch.long)

        return {"img": img, "label": label}
