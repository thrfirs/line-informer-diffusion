import argparse
import os
import cv2
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def convert_raw_to_target(json_path: str, target_numpy_path: str):
    """
    raw: {"line":[[0, 0, 0], ...], "base_params": [xxx, ...]}

    target:
    data = np.load(fr, allow_pickle=True).item()
    line = data['line']  # (seq_len, 3): dx, dy, dz
    params = data['params']  # (1, 7): dist, span_x, span_y, span_z, start_x, start_y, start_z
    lengths = data['lengths']
    num_samples = data['num_samples']
    num_repetitions = data['num_repetitions']
    """

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    line = data['line']
    base_params = data['base_params']

    # convert line to relative distances
    distances = [(0, 0, 0)]
    for i in range(1, len(line)):
        prev_point = line[i - 1]
        cur_point = line[i]
        dx = cur_point[0] - prev_point[0]
        dy = cur_point[1] - prev_point[1]
        dz = cur_point[2] - prev_point[2]
        distances.append((dx, dy, dz))
    
    # convert base_params to target_params
    x_span = max([x for x, y, z in line]) - min([x for x, y, z in line])
    y_span = max([y for x, y, z in line]) - min([y for x, y, z in line])
    z_span = max([z for x, y, z in line]) - min([z for x, y, z in line])
    start_x, start_y, start_z = line[0]
    target_params = base_params + [x_span, y_span, z_span, start_x, start_y, start_z]

    # save to target_json_path
    data = {
        'line': np.expand_dims(np.array(distances), axis=0),
        'params': np.expand_dims(np.array(target_params), axis=0),
        'lengths': np.array([[len(line)]]),
        'num_samples': np.array([[1]]),
        'num_repetitions': np.array([[1]]),
    }

    np.save(target_numpy_path, data)


def main():
    raw_json_dir = "/root/line-informer-diffusion/data/prepared/"
    target_numpy_dir = "/root/line-informer-diffusion/data/prepared_target/"

    if os.path.exists(target_numpy_dir):
        shutil.rmtree(target_numpy_dir)
    os.makedirs(target_numpy_dir)

    for json_name in tqdm(os.listdir(raw_json_dir)):
        json_path = os.path.join(raw_json_dir, json_name)
        base_name_wo_ext = os.path.splitext(json_name)[0]
        target_numpy_path = os.path.join(target_numpy_dir, base_name_wo_ext + ".npy")
        convert_raw_to_target(json_path, target_numpy_path)


if __name__ == '__main__':
    main()
