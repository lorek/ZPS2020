"""Prepare Adult dataset
Author: Dawid Wegner
Group: z1

Example:
    $ python scripts/z1_prepare_data_adult.py --input-dir datasets_prepared/adult/ \
        --output-dir datasets_prepared/adult_one_hot/ --categorical-features-indexes 0 1 2 3 4 5 6 7 8 \
        --categorical-features-classes 5 5 5 5 5 5 5 5 5
"""

import argparse
import os
import pickle
from shutil import copyfile
from typing import Sequence

import numpy as np
import pandas as pd

INPUT_FILENAMES = ["train_data.pkl", "test_data.pkl"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mushroom dataset")
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
        "--categorical-features-classes", nargs="+", type=int, required=True,
        help="The number of classes of categorical features"
    )
    parser.add_argument(
        "--first-class", default=0, type=int, required=False,
        help="First class of integer-encoded attribute (0 or 1)"
    )
    return parser.parse_args()


def read_serialized_dataset(path):
    with open(path, "rb") as file_stream:
        saved_data = pickle.load(file_stream)
        return saved_data['data'], saved_data['classes']


def save_dataset(
        x_array: np.ndarray,
        y_array: np.ndarray,
        path: str,
):
    data_dict = dict(data=x_array, classes=y_array)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file_stream:
        pickle.dump(data_dict, file_stream, protocol=pickle.HIGHEST_PROTOCOL)


def copy_dataset(input_dir: str, output_dir: str, filename='class_names.pkl'):
    src = os.path.join(input_dir, filename)
    dst = os.path.join(output_dir, filename)
    copyfile(src, dst)


def create_one_hot_features_df(features_df: pd.DataFrame, features_classes: Sequence[int]):
    columns = list(features_df.columns)
    one_hot_dataframes = []
    for column, feature_classes in zip(columns, features_classes):
        feature_array = np.array(features_df[column])
        one_hot_array = np.zeros(shape=(feature_array.shape[0], feature_classes))
        one_hot_array[np.arange(feature_array.shape[0]), feature_array] = 1.0
        feature_columns = [f'{column}_{feature_class}' for feature_class in range(feature_classes)]
        one_hot_dataframes.append(pd.DataFrame(one_hot_array, columns=feature_columns))
    return pd.concat(one_hot_dataframes, axis=1)


def process_script_for_file(
        input_dir: str,
        output_dir: str,
        filename: str,
        categorical_features_indexes: Sequence[int],
        categorical_features_classes: Sequence[int],
        first_class: int = 0
):
    input_path = os.path.join(input_dir, filename)
    x_array, y_array = read_serialized_dataset(input_path)
    features_df = pd.DataFrame(x_array)
    categorical_features_df = features_df[categorical_features_indexes] - first_class
    continuous_features_indexes = list(set(features_df.columns).difference(categorical_features_indexes))
    continuous_features_df = features_df[continuous_features_indexes]
    one_hot_features_df = create_one_hot_features_df(categorical_features_df, categorical_features_classes)
    converted_features_df = pd.concat([continuous_features_df, one_hot_features_df], axis=1)
    converted_x_array = converted_features_df.to_numpy()
    output_path = os.path.join(output_dir, filename)
    save_dataset(converted_x_array, y_array, output_path)


def run_script(input_dir, output_dir, categorical_features_indexes, categorical_features_classes, first_class=0):
    if len(categorical_features_classes) != len(categorical_features_indexes):
        raise ValueError("The number of features classes must match the number of features indexes")
    for filename in INPUT_FILENAMES:
        process_script_for_file(
            input_dir, output_dir, filename, categorical_features_indexes, categorical_features_classes, first_class
        )
    copy_dataset(input_dir, output_dir, "class_names.pkl")


if __name__ == "__main__":
    args = parse_arguments()
    run_script(args.input_dir, args.output_dir, args.categorical_features_indexes, args.categorical_features_classes,
               args.first_class)
