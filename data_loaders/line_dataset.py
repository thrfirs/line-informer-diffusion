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


def get_line_relative_distances(line: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    result = [(0, 0, 0)]
    for i in range(1, len(line)):
        prev_point = line[i - 1]
        cur_point = line[i]
        dx = cur_point[0] - prev_point[0]
        dy = cur_point[1] - prev_point[1]
        dz = cur_point[2] - prev_point[2]
        result.append((dx, dy, dz))
    return result


class LineDataset(data.Dataset):
    def __init__(self, path: str = os.path.join("data", "dataset.txt")):
        self.index_file_path = path
        self.data = []
        self.load_data()
        self.num_params = 7  # 1 base, 3 span, 3 start
    
    def load_data(self):
        with open(self.index_file_path, 'r') as f:
            for line in f:
                file_path = line.strip()
                with open(file_path, 'r') as f_data:
                    self.data.append(json.load(f_data))
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        line = random_rotate_z_axis(item["line"])
        #angles = get_line_relative_angles(line)
        distances = get_line_relative_distances(line)
        x_span = max([x for x, y, z in line]) - min([x for x, y, z in line])
        y_span = max([y for x, y, z in line]) - min([y for x, y, z in line])
        z_span = max([z for x, y, z in line]) - min([z for x, y, z in line])
        start_x, start_y, start_z = line[0]
        params = item["base_params"] + [x_span, y_span, z_span, start_x, start_y, start_z]
        #return {"line": angles, "params": params}
        return {"line": distances, "params": params}

    def __len__(self) -> int:
        return len(self.data)


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["line"] for b in notnone_batches]  # each line is a list of Tuple[float, float, float]

    batch_size = len(databatch)
    max_len = max([len(line) for line in databatch])
    data = np.zeros((batch_size, max_len, 3), dtype=np.float32)
    for i, line in enumerate(databatch):
        data[i, : len(line), :] = line
    data = torch.from_numpy(data)
    # convert data to float32
    data = data.float()

    params_batch = [b["params"] for b in notnone_batches]
    cond = {"y": {"params": torch.as_tensor(params_batch, dtype=torch.float32)}}

    return data, cond
