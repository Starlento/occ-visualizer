import numpy as np


XC_CN_CLASS_NAMES = {
    0: "free",
    1: "vehicle",
    2: "pedestrian",
    3: "other_stable_obstacle",
    4: "other_dynamic_obstacle",
    5: "wall_door",
    6: "pillar",
    7: "curb",
    8: "stopper_locker",
    9: "higher_roadway",
    10: "unknown",
    11: "freespace",
    12: "ghost_obstacle",
}


def get_nuscenes_colormap():
    return np.array(
        [
            [0, 0, 0, 255],
            [255, 120, 50, 255],
            [255, 192, 203, 255],
            [255, 255, 0, 255],
            [0, 150, 245, 255],
            [0, 255, 255, 255],
            [255, 127, 0, 255],
            [255, 0, 0, 255],
            [255, 240, 150, 255],
            [135, 60, 0, 255],
            [160, 32, 240, 255],
            [255, 0, 255, 255],
            [139, 137, 137, 255],
            [75, 0, 75, 255],
            [150, 240, 80, 255],
            [230, 230, 250, 255],
            [0, 175, 0, 255],
        ],
        dtype=np.uint8,
    )


def get_xc_cn_colormap():
    # 13 perceptually distinct colors, one per class:
    # 0:free  1:vehicle  2:pedestrian  3:other_stable_obstacle
    # 4:other_dynamic_obstacle  5:wall_door  6:pillar  7:curb
    # 8:stopper_locker  9:higher_roadway  10:unknown  11:freespace  12:ghost_obstacle
    return np.array(
        [
            [0,   0,   0,   255],  # 0  free             — black
            [220,  20,  20,  255],  # 1  vehicle          — vivid red
            [255, 220,   0,  255],  # 2  pedestrian       — vivid yellow
            [30,  100, 220,  255],  # 3  other_stable_obs — medium blue
            [0,   210,  70,  255],  # 4  other_dynamic_obs— bright green
            [255, 140,   0,  255],  # 5  wall_door        — vivid orange
            [210,   0, 200,  255],  # 6  pillar           — magenta
            [0,   210, 220,  255],  # 7  curb             — cyan
            [120,   0, 220,  255],  # 8  stopper_locker   — deep purple
            [180,  90,   0,  255],  # 9  higher_roadway   — brown
            [255, 160, 120,  255],  # 10 unknown          — coral
            [110, 110, 110,  255],  # 11 freespace        — dark gray
            [255, 160, 120,  255],  # 12 ghost_obstacle   — coral
        ],
        dtype=np.uint8,
    )


def get_kitti_colormap():
    return np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.392, 0.392, 0.392, 1.0],
            [0.392, 0.0, 0.0, 1.0],
            [0.588, 0.274, 0.078, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.647, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.6, 0.0, 1.0],
            [0.0, 0.8, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.255, 0.412, 0.882, 1.0],
            [0.541, 0.169, 0.886, 1.0],
            [1.0, 0.078, 0.576, 1.0],
            [0.863, 0.863, 0.863, 1.0],
            [0.627, 0.322, 0.176, 1.0],
            [0.133, 0.545, 0.133, 1.0],
            [0.184, 0.31, 0.31, 1.0],
            [0.0, 0.0, 0.502, 1.0],
            [0.275, 0.51, 0.706, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def get_kitti360_colormap():
    return np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.502, 0.251, 0.502, 1.0],
            [0.957, 0.137, 0.91, 1.0],
            [0.275, 0.51, 0.706, 1.0],
            [0.0, 0.0, 0.557, 1.0],
            [0.0, 0.0, 0.902, 1.0],
            [0.0, 0.235, 0.392, 1.0],
            [0.863, 0.078, 0.235, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.392, 0.0, 1.0],
            [1.0, 0.588, 0.0, 1.0],
            [0.941, 0.902, 0.549, 1.0],
            [0.42, 0.557, 0.137, 1.0],
            [0.133, 0.545, 0.133, 1.0],
            [0.824, 0.706, 0.549, 1.0],
            [0.6, 0.6, 0.6, 1.0],
            [0.3, 0.3, 0.3, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


__all__ = [
    "XC_CN_CLASS_NAMES",
    "get_kitti360_colormap",
    "get_kitti_colormap",
    "get_nuscenes_colormap",
    "get_xc_cn_colormap",
]