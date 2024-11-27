import os
import argparse
from tqdm import tqdm
import shutil
import random


def main(input_dir: str, train_dir: str, test_dir: str, ratio: int):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    fnames = os.listdir(input_dir)
    random.shuffle(fnames)
    for i, name in enumerate(tqdm(fnames)):
        input_path = os.path.join(input_dir, name)
        is_test = i % ratio == 0 
        if is_test:
            shutil.copy(input_path, test_dir)
        else:
            shutil.copy(input_path, train_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--ratio', type=int, default=9) # x to 1 ratio
    parser.add_argument('--output_prefix', type=str, default='./data/data')
    args = parser.parse_args()

    main( args.input_dir,
        f"{args.output_prefix}_train",
        f"{args.output_prefix}_test",
        args.ratio)

