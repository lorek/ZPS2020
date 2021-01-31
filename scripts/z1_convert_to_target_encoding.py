"""Convert integer encoding to mean encoding
Author: Tomasz Nanowski
Group: z1

Example:
    $ python scripts/z1_convert_to_target_encoding.py --input-dir datasets_prepared/adult/ \
        --output-dir datasets_prepared/adult_one_hot/ --categorical-features-indexes 0 1 2 3 4 5 6 7 8
"""

import argparse
import os
import pickle
from shutil import copyfile
from typing import List

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from category_encoders.wrapper import PolynomialWrapper

INPUT_FILENAMES = ["train_data.pkl", "test_data.pkl"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert integer encoding to mean encoding")
    parser.add_argument(
        "--input-dir", required=True, type=str, help="Input directory"
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--categorical-features-indexes", nargs="+", type=int, required=True,
        help="The indexes of categorical features"
    )
    parser.add_argument(
        "--smoothing", default=1.0, type=float, required=False,
        help="Smoothing parameter for target encoding"
    )
    return parser.parse_args()


def read_serialized_dataset(path):
    with open(path, "rb") as file_stream:
        saved_data = pickle.load(file_stream)
        return saved_data['data'], saved_data['classes']


def load_original_dataset(input_dir: str,
                          filename: str, categorical_features_indexes: List[int]):
    input_path = os.path.join(input_dir, filename)
    x, y = read_serialized_dataset(input_path)
    x = pd.DataFrame(x)
    x.iloc[:, categorical_features_indexes] = x.iloc[:, categorical_features_indexes].astype('object')
    return x, y


def save_dataset(
        x_array: np.ndarray,
        y_array: np.ndarray,
        output_dir: str,
        filename: str,
):
    path = os.path.join(output_dir, filename)
    data_dict = dict(data=x_array, classes=y_array)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file_stream:
        pickle.dump(data_dict, file_stream, protocol=pickle.HIGHEST_PROTOCOL)


def copy_dataset(input_dir: str, output_dir: str, filename='class_names.pkl'):
    src = os.path.join(input_dir, filename)
    dst = os.path.join(output_dir, filename)
    copyfile(src, dst)


def create_target_encoder(x: pd.DataFrame, y: np.ndarray,
                          categorical_features_indexes: List[int],
                          smoothing: float = 1.0) -> PolynomialWrapper:
    encoder = PolynomialWrapper(TargetEncoder(cols=categorical_features_indexes, smoothing=smoothing))
    encoder.fit(X=x, y=y)
    return encoder


def transform_dataset(encoder, data):
    data = encoder.transform(data)
    return data.to_numpy()


def run_script(input_dir, output_dir, categorical_features_indexes, smoothing: float = 1.0):
    x_train, y_train = load_original_dataset(input_dir, "train_data.pkl", categorical_features_indexes)
    x_test, y_test = load_original_dataset(input_dir, "test_data.pkl", categorical_features_indexes)

    encoder = create_target_encoder(x_train, y_train, categorical_features_indexes, smoothing)
    x_train = transform_dataset(encoder, x_train)
    x_test = transform_dataset(encoder, x_test)

    save_dataset(x_train, y_train, output_dir, "train_data.pkl")
    save_dataset(x_test, y_test, output_dir, "test_data.pkl")
    copy_dataset(input_dir, output_dir, "class_names.pkl")


if __name__ == "__main__":
    args = parse_arguments()
    run_script(args.input_dir, args.output_dir, args.categorical_features_indexes, args.smoothing)
