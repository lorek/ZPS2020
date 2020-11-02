"""Prepare Adult dataset
Author: Tomasz Nanowski
Group: z1

Example:
    $ python scripts/z1_prepare_data_tae_one_hot.py --input-dir datasets_prepared/teaching_assistant_evaluation \
        --output-dir datasets_prepared/teaching_assistant_evaluation_one_hot
"""

import argparse

import z1_convert_to_one_hot_encoding

FIRST_CLASS = 1
CATEGORICAL_FEATURES_INDEXES = [0, 1, 2, 3]
CATEGORICAL_FEATURES_CLASSES = [2, 25, 26, 2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input-dir", required=True, type=str, help="Input directory"
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Output directory"
    )
    return parser.parse_args()


def run_script(input_dir, output_dir):
    z1_convert_to_one_hot_encoding.run_script(input_dir,
                                              output_dir,
                                              CATEGORICAL_FEATURES_INDEXES,
                                              CATEGORICAL_FEATURES_CLASSES,
                                              FIRST_CLASS)


if __name__ == "__main__":
    args = parse_arguments()
    run_script(args.input_dir, args.output_dir)
