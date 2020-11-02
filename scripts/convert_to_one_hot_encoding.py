"""Prepare Adult dataset
Author: Dawid Wegner
Group: z1

Example:
    $ python scripts/z1_prepare_data_adult.py --input-dir datasets_prepared/adult/ --input-files train_data.pkl \
        test_data.pkl --output-dir datasets_prepared/adult_one_hot/ --categorical-features-indexes 0 1 2 3 4 5 6 7 8 \
        --categorical-features-classes 5 5 5 5 5 5 5 5 5
"""

import argparse
import os
import pickle
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mushroom dataset")
    parser.add_argument(
        "--input-dir", required=True, type=str, help="Input directory"
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--input-files", nargs="+", required=True, type=str, help="The names of files that will be processed"
    )
    parser.add_argument(
        "--categorical-features-indexes", nargs="+", type=int, required=True,
        help="The indexes of categorical features"
    )
    parser.add_argument(
        "--categorical-features-classes", nargs="+", type=int, required=True,
        help="The number of classes of categorical features"
    )
    return parser.parse_args()


def read_serialized_dataset(path):
    with open(path, "rb") as file_stream:
        saved_data = pickle.load(file_stream)
        return saved_data['data'], saved_data['classes']


def save_dataset_slice(dataset_slice: Any, output_dir: str, filename: str):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filename}.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(dataset_slice, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    output_dir: str,
):
    x_train_all_dict = dict(data=x_train, classes=y_train)
    save_dataset_slice(x_train_all_dict, output_dir, "train_data")
    x_test_all_dict = dict(data=x_test, classes=y_test)
    save_dataset_slice(x_test_all_dict, output_dir, "test_data")
    save_dataset_slice(classes, output_dir, "class_names")


def process_script_for_file(
        input_dir: str,
        output_dir: str,
        filename: str,
        categorical_features_indexes: Sequence[int],
        categorical_features_classes: Sequence[int],
):
    input_path = os.path.join(input_dir, filename)
    x_array, y_array = read_serialized_dataset(input_path)
    features_df = pd.DataFrame(x_array)
    categorical_features_df = features_df[categorical_features_indexes]
    continuous_features_indexes = list(set(features_df.columns).difference(categorical_features_indexes))
    continuous_features_df = features_df[continuous_features_indexes]


def run_script():
    input_args = parse_arguments()
    if len(input_args.categorical_features_classes) != len(input_args.categorical_features_indexes):
        raise ValueError("The number of features classes must match the number of features indexes")
    for filename in input_args.input_files:
        process_script_for_file(
            input_args.input_dir,
            input_args.output_dir,
            filename,
            input_args.categorical_features_indexes,
            input_args.categorical_features_classes,
        )


if __name__ == "__main__":
    run_script()

