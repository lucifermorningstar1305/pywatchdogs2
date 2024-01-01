"""
@author: Adityam Ghosh
Date: 12/31/2023
"""

import numpy as np
import polars as pl
import os
import argparse

from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-D",
        type=str,
        required=True,
        help="the path containing the data.",
    )

    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        required=True,
        help="the path to save the metadata",
    )

    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path

    data = defaultdict(lambda: list())

    if not os.path.exists("./data/"):
        os.mkdir("./data/")

    for root, folders, files in os.walk(data_path):
        for filename in files:
            label = filename.split(".")[0].split("_")[-1]

            data["img_file"].append(os.path.join(root, filename))
            data["label"].append(int(label))

    df = pl.from_dict(data)
    df.write_parquet(save_path)
