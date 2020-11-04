"""Prepare car dataset variant hashing encoding
Author: Rafał Safin
Group: z2

Example:

$ python scripts/z2/z2_prepare_data_car_hashing.py --input-dir datasets/z2/car/ --output-dir datasets_prepared/z2/car_hashing_prepared/

Simple presentation:
$ python scripts_learn/classify_simple.py --input-dir datasets_prepared/z2/car_hashing_prepared/


"""

import argparse
import os
import pickle
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce

def read_data(input_dir: str) -> pd.DataFrame:
    data_path = os.path.join(input_dir, "car.data")
    data_df = pd.read_csv(data_path, header=None, names=['b', 'm', 'd', 'p', 'l', 's', 'c'])
    return data_df


def process_data(data_df: pd.DataFrame) -> pd.DataFrame:
    encoder=ce.HashingEncoder(cols=['b', 'm', 'd', 'p', 'l', 's'],n_components=10, return_df=True)
    data_encoded_df =encoder.fit_transform(data_df)
    
    print(data_encoded_df)
    return data_encoded_df


def convert_data(data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    data_all = data_df.to_numpy()
    data = data_all[:, :-1]
    data_classes = data_all[:, -1:]
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
