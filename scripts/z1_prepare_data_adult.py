"""Prepare Adult dataset
Author: Dawid Wegner
Group: z1

Example:
    $ python scripts/z1_prepare_data_adult.py --input-dir datasets/adult/ --output-dir datasets_prepared/adult/
"""

import argparse
import logging
import os
import pickle
from typing import Any
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(input_dir: str) -> pd.DataFrame:
    data_path = os.path.join(input_dir, "adult.data")
    data_df = pd.read_csv(data_path, header=None, sep=",")
    return data_df


def process_data(data_df: pd.DataFrame) -> pd.DataFrame:
    categorical_df = data_df[[1, 3, 5, 6, 7, 8, 9, 13, 14]]
    continuous_df = data_df[[0, 2, 4, 10, 11, 12]]
    factorized_categorical_df = categorical_df.apply(lambda x: pd.factorize(x)[0])
    return pd.concat([continuous_df, factorized_categorical_df], axis=1)


def convert_data(data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    data_all = data_df.to_numpy()
    data = data_all[:, :-1]
    data_classes = data_all[:, -1]
    return data, data_classes


def save_obj(obj: Any, output_dir: str, filename: str):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filename}.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_data(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        classes: np.ndarray,
        output_dir: str,
):
    X_train_all_dict = dict(data=X_train, classes=y_train)
    save_obj(X_train_all_dict, output_dir, "train_data")

    X_test_all_dict = dict(data=X_test, classes=y_test)
    save_obj(X_test_all_dict, output_dir, "test_data")

    save_obj(classes, output_dir, "class_names")

    logging.info(f"Pickles saved in {output_dir}")


def main(input_dir: str, output_dir: str, fraction: float):
    data_df = read_data(input_dir)
    processed_df = process_data(data_df)
    data, data_classes = convert_data(processed_df)
    classes = np.unique(data_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        data, data_classes, test_size=fraction, random_state=42
    )
    save_data(X_train, y_train, X_test, y_test, classes, output_dir)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mushroom dataset")
    parser.add_argument(
        "--input-dir", default="", required=True, help="data dir (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        default="",
        required=True,
        help="output dir (default: %(default)s)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        required=False,
        help="size of test set (fration) (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_dir, args.output_dir, args.fraction)
