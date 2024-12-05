import argparse
from easydict import EasyDict
import yaml
from tqdm import tqdm
import os
import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of Music2Dance')
    parser.add_argument('--config', default='')
    parser.add_argument('--input_dir', default='./data/all_split_train')
    parser.add_argument('--output_dir', default='./data/all_split_train_interval')
    parser.add_argument('--interval', type=int)
    parser.add_argument('--move', type=int)
    return parser.parse_args()


def build_data_aist(data_dir, save_dir, interval=120, move=40, rotmat=False, external_wav=None, external_wav_rate=1, music_normalize=False, wav_padding=0):
    print(f"spliting data from {data_dir} with interval {interval}, and move of {move}")
    fnames = sorted(os.listdir(data_dir))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for fname in tqdm(fnames):
        dance_data = []
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            json_data = json.loads(f.read())
            if type(json_data) == dict:
                # print(f"{path} is DICT")
                json_data = json_data['dance_array']
            else:
                # print(f"{path} is NOT")
                pass

            np_dance = np.array(json_data)

            if not rotmat:
                root = np_dance[:, :3]  # the root
                np_dance = np_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                np_dance[:, :3] = root

            if interval is None:
                dance_data.append(np_dance)
                continue

            seq_len, _ = np_dance.shape
            for i in range(0, seq_len, move):
                dance_sub_seq = np_dance[i: i + interval]

                if len(dance_sub_seq) == interval:
                    dance_data.append(dance_sub_seq)

        raw_name = ".".join(fname.split(".")[:-1])
        for i, data in enumerate(dance_data):
            data_name = f"{raw_name}_interval_{i}"
            full_path = os.path.join(save_dir, data_name)
            np.save(full_path, data)


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    data = config.data
    build_data_aist(
            args.input_dir,
            args.output_dir,
            interval = data.seq_len if not hasattr(args, "interval") else args.interval,
            move = args.move if hasattr(args, "move") else (config.move_train if hasattr(config, 'move_train') else 64),
            rotmat=config.rotmat,
        )
    
    
if __name__ == "__main__":
    main()