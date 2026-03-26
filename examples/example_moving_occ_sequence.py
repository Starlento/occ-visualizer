from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from occupancy_visualizer import save_occ


FRAME_RATE_HZ = 10
DURATION_SECONDS = 5
NUM_FRAMES = FRAME_RATE_HZ * DURATION_SECONDS
GRID_SHAPE = (200, 200, 16)
OUTPUT_DIR = ROOT_DIR / "outputs"

LABEL_CAR = 4
LABEL_DRIVEABLE_SURFACE = 11
LABEL_MANMADE = 15
LABEL_EMPTY = 0


def _print_progress(current, total):
    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rRendering frames: [{bar}] {current}/{total}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def _world_to_grid(x_meters, y_meters, z_meters):
    voxel = 0.5
    origin = np.array([-50.0, -50.0, -5.0], dtype=np.float32)
    world = np.array([x_meters, y_meters, z_meters], dtype=np.float32)
    return np.floor((world - origin) / voxel).astype(np.int32)


def _fill_box(grid, center_xyz, size_xyz, label):
    half_sizes = np.array(size_xyz, dtype=np.float32) / 2.0
    min_corner = _world_to_grid(*(np.array(center_xyz) - half_sizes))
    max_corner = _world_to_grid(*(np.array(center_xyz) + half_sizes))

    min_corner = np.clip(min_corner, 0, np.array(grid.shape) - 1)
    max_corner = np.clip(max_corner, 0, np.array(grid.shape) - 1)

    grid[
        min_corner[0] : max_corner[0] + 1,
        min_corner[1] : max_corner[1] + 1,
        min_corner[2] : max_corner[2] + 1,
    ] = label


def build_frame(frame_index):
    grid = np.zeros(GRID_SHAPE, dtype=np.uint8)

    grid[:, :, 0] = LABEL_DRIVEABLE_SURFACE

    _fill_box(
        grid,
        center_xyz=(8.0, 6.0, 1.5),
        size_xyz=(1.5, 1.5, 3.0),
        label=LABEL_MANMADE,
    )

    start_x = -18.0
    speed_mps = 4.0
    time_seconds = frame_index / FRAME_RATE_HZ
    car_center_x = start_x + speed_mps * time_seconds
    _fill_box(
        grid,
        center_xyz=(car_center_x, 0.0, 0.75),
        size_xyz=(4.0, 2.0, 1.5),
        label=LABEL_CAR,
    )

    return grid


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for frame_index in range(NUM_FRAMES):
        occupancy = build_frame(frame_index)
        save_occ(
            save_dir=str(OUTPUT_DIR),
            occupancy=occupancy,
            name=f"frame_{frame_index:03d}",
            sem=True,
            dataset="nusc",
            empty_label=LABEL_EMPTY,
        )
        _print_progress(frame_index + 1, NUM_FRAMES)

    print(f"Generated {NUM_FRAMES} frames in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()