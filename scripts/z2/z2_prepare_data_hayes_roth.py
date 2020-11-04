"""Prepare Hayes-Roth dataset
Author: Rafał Safin
Group: z2

Example:

$ python scripts/z2/z2_prepare_data_hayes_roth.py --input-dir datasets/z2/Hayes-Roth/ --output-dir datasets_prepared/z2/Hayes-Roth_prepared/

Simple presentation:
$ python scripts_learn/classify_simple.py --input-dir datasets_prepared/z2/Hayes-Roth_prepared/

"""

import argparse
import os
import pickle
from typing import Any, Tuple

import numpy as np
import pandas as pd


def read_data(input_dir: str) -> pd.DataFrame:
    data_path = os.path.join(input_dir, "hayes-roth.data")
    data_df = pd.read_csv(data_path, header=None, sep=",")
    return data_df

def read_test_data(input_dir: str) -> pd.DataFrame:
    data_path = os.path.join(input_dir, "hayes-roth.test")
    data_df = pd.read_csv(data_path, header=None, sep=",")
    return data_df

def convert_data(data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    data_all = data_df.to_numpy()
    data = data_all[:, 1:4]
    data_classes = data_all[:, 5]
    return data, data_classes

def convert_test_data(data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    data_all = data_df.to_numpy()
    data = data_all[:, :3]
    data_classes = data_all[:, 4]
    return data, data_classes

def save_obj(obj: Any, output_dir: str, filename: str):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filename}.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    output_dir: str,
):
    x_train_all_dict = dict(data=x_train, classes=y_train)
    save_obj(x_train_all_dict, output_dir, "train_data")

    x_test_all_dict = dict(data=x_test, classes=y_test)
    save_obj(x_test_all_dict, output_dir, "test_data")

    save_obj(classes, output_dir, "class_names")

    print(f"Pickles saved in {output_dir}")


def main(input_dir: str, output_dir: str):
    data_df = read_data(input_dir)
    data_test_df = read_test_data(input_dir)

    x_train, y_train = convert_data(data_df)
    x_test, y_test = convert_test_data(data_test_df)
    classes = np.unique(y_test)

    save_data(x_train, y_train, x_test, y_test, classes, output_dir)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hayes-Roth dataset")
    parser.add_argument(
        "--input-dir", default="", required=True, help="data dir (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        default="",
        required=True,
        help="output dir (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_dir, args.output_dir)
