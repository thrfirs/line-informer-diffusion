import torch
import os
import json
import random
import numpy as np
from torch.utils import data
from typing import List, Tuple, Dict, Any


def random_rotate_z_axis(line: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    center_coord = np.mean(line, axis=0)
    line = line - center_coord
    angle = np.random.uniform(0, 2 * np.pi)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    line = np.dot(line, rot_mat)
    line = line + center_coord
    return line


def get_line_relative_angles(line: List[Tuple[float, float, float]]) -> List[Tuple[float, float]]:
    # distance between two points are always the same, so only consider the angles
    result = [(0, 0)]
    for i in range(1, len(line)):
        prev_point = line[i - 1]
        cur_point = line[i]
        alpha = np.arctan2(cur_point[1] - prev_point[1], cur_point[0] - prev_point[0])
        beta = np.arctan2(cur_point[2] - prev_point[2], cur_point[0] - prev_point[0])
        result.append((alpha, beta))
    return result


class LineDataset(data.Dataset):
    def __init__(self, path: str = os.path.join("data", "dataset.txt")):
        self.index_file_path = path
        self.data = []
        self.load_data()
    
    def load_data(self):
        with open(self.index_file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        line = random_rotate_z_axis(item["line"])
        angles = get_line_relative_angles(line)
        params = item["params"]
        return {"line": angles, "params": params}


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["line"] for b in notnone_batches]  # each line is a list of Tuple[float, float, float]

    batch_size = len(databatch)
    max_len = max([len(line) for line in databatch])
    data = np.zeros((batch_size, max_len, 3), dtype=np.float32)
    for i, line in enumerate(databatch):
        data[i, : len(line), :] = line
    data = torch.from_numpy(data)

    params_batch = [b["params"] for b in notnone_batches]
    cond = {"y": {"params": torch.as_tensor(params_batch)}}

    return data, cond
