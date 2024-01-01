"""
@author: Adityam Ghosh
Date: 01/01/2024
"""

import numpy as np
import polars as pl
import gc
import argparse


def show_value_counts(data: pl.DataFrame):
    value_counts = (
        data.select(pl.col("label")).group_by(pl.col("label")).agg(pl.count())
    )

    return value_counts


def resample(data: pl.DataFrame, save_path: str):
    """Function to balance the data"""

    value_counts = show_value_counts(data)
    print("---Previous Value Counts---")
    print(value_counts)
    print("\n\n")

    count_map = dict()
    max_val = float("-inf")
    for row in value_counts.iter_rows():
        count_map[row[0]] = row[1]
        if row[1] > max_val:
            max_val = row[1]

    diff_map = {k: max_val - v for k, v in count_map.items()}

    sub_datas = [data]
    for k, v in diff_map.items():
        if v == 0:
            continue
        sub_data = data.filter(pl.col("label") == k).sample(
            n=v, seed=32, with_replacement=True
        )
        sub_datas.append(sub_data)
        del sub_data
        gc.collect()

    final_data = pl.concat(sub_datas, how="vertical")

    value_counts = show_value_counts(final_data)
    print("---Final Value Counts---")
    print(value_counts)
    print("\n\n")

    final_data.sample(fraction=1.0, shuffle=True, seed=32)

    final_data.write_parquet(save_path)


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
    df = pl.read_parquet(data_path)
    resample(df, save_path)
