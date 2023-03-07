import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from multiprocessing import Manager, Queue
from concurrent.futures import ProcessPoolExecutor, wait


def rebuild_line(line: np.ndarray, start_point: np.ndarray) -> np.ndarray:
    result = [start_point]
    for i in range(1, line.shape[0]):
        prev_point = result[i - 1]
        cur_point = line[i]
        dx = cur_point[0] + prev_point[0]
        dy = cur_point[1] + prev_point[1]
        dz = cur_point[2] + prev_point[2]
        result.append((dx, dy, dz))
    return np.array(result)


def calc_coordinate_ranges(line: np.ndarray, expand_factor: float = 1.5) -> np.ndarray:
    x_min, x_max = min(line[:, 0]), max(line[:, 0])
    y_min, y_max = min(line[:, 1]), max(line[:, 1])
    z_min, z_max = min(line[:, 2]), max(line[:, 2])

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    x_span = (x_max - x_min) * expand_factor
    y_span = (y_max - y_min) * expand_factor
    z_span = (z_max - z_min) * expand_factor

    x_range = (x_center - x_span / 2, x_center + x_span / 2)
    y_range = (y_center - y_span / 2, y_center + y_span / 2)
    z_range = (z_center - z_span / 2, z_center + z_span / 2)

    return np.array([x_range, y_range, z_range])


def draw_line(line: np.ndarray, ranges: np.ndarray, rot_deg: float, width: int, height: int) -> np.ndarray:
    # rotate across z-axis by rot_deg
    rot_rad = rot_deg * np.pi / 180
    # use range to get center
    center = np.mean(ranges, axis=1)
    line = line - center
    rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0],
                        [np.sin(rot_rad), np.cos(rot_rad), 0],
                        [0, 0, 1]])
    line = np.dot(line, rot_mat)
    line = line + center

    # draw line using matplotlib
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    x_range, y_range, z_range = ranges
    ax.plot(line[:, 0], line[:, 1], line[:, 2])
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])
    fig.canvas.draw()

    # convert to numpy array
    w, h = fig.canvas.get_width_height()
    #img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    # Use frombuffer instead
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # got the warning: More than 20 figures have been opened.
    # should close the figure
    plt.close(fig)

    return img


def draw_line_wrapper(index: int, line: np.ndarray, ranges: np.ndarray, rot_deg: float, width: int, height: int, q: Queue):
    q.put((index, draw_line(line, ranges, rot_deg, width, height)))


def render_video(file_path: str, line: np.ndarray, ranges: np.ndarray, width: int, height: int, steps: int, fps: int, workers: int):
    # draw images concurrently
    futures = []
    manager = Manager()
    q = manager.Queue()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(steps):
            rot_deg = 360 * i / steps
            futures.append(executor.submit(draw_line_wrapper, i, line, ranges, rot_deg, width, height, q))
        imgs = []
        for _ in trange(steps, desc="Draw", dynamic_ncols=True):
            index, img = q.get()
            imgs.append((index, img))
        wait(futures)
    imgs = sorted(imgs, key=lambda x: x[0])

    # write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for _, img in tqdm(imgs, desc="Render", dynamic_ncols=True):
        out.write(img)
    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help=".npy with a generated line")
    parser.add_argument("--output_path", type=str, required=True, help=".mp4 to be rendered for line animation")
    parser.add_argument("--steps", type=int, default=60, help="Number of steps to rotate the line for full 360 degree")
    parser.add_argument("--fps", type=int, default=6, help="Frames per second for the output video")
    parser.add_argument("--width", type=int, default=640, help="Width of the output video")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to render the video")
    args = parser.parse_args()

    with open(args.input_path, 'rb') as fr:
        data = np.load(fr, allow_pickle=True).item()
        line = data['line']  # (seq_len, 3): dx, dy, dz
        params = data['params']  # (1, 7): dist, span_x, span_y, span_z, start_x, start_y, start_z
        lengths = data['lengths']
        num_samples = data['num_samples']
        num_repetitions = data['num_repetitions']
    
    line = rebuild_line(line[0], params[0, 4:7])
    ranges = calc_coordinate_ranges(line)
    render_video(args.output_path, line, ranges, args.width, args.height, args.steps, args.fps, args.workers)


if __name__ == "__main__":
    main()
