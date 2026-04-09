from __future__ import annotations

import ast
from collections.abc import Iterable
from datetime import datetime
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


TIMESTAMP_FORMAT = "%Y%m%d%H%M%S.%f"

FLAT_FRAME_FILENAMES = {
    "freespace": "freespace.png",
    "freespace_height": "freespace_height.npy",
    "freespace_kpts": "freespace_kpts.npy",
    "occ": "occ.npz",
}

EXPECTED_CAMERA_NAMES = (
    "SurCam01",
    "SurCam02",
    "SurCam03",
    "SurCam04",
    "FrontCam02",
)

EXPECTED_SOURCE_FRAME_FILE_COUNT = len(FLAT_FRAME_FILENAMES) + len(EXPECTED_CAMERA_NAMES) + 1
DEFAULT_RENDER_MAX_SIDE = 1800

_CAMERA_PRESET_LOOKUP = {
    "SurCam01": {
        "position_offset": np.array([4.0, 6.0, 2.8], dtype=np.float32),
        "forward": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "target_offset": np.array([2.0, 0.0, -0.5], dtype=np.float32),
        "focal_distance": 20.0,
        "view_angle": 95.0,
    },
    "SurCam02": {
        "position_offset": np.array([-5.0, 0.0, 3.0], dtype=np.float32),
        "forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "target_offset": np.array([0.0, 0.0, -0.5], dtype=np.float32),
        "focal_distance": 20.0,
        "view_angle": 90.0,
    },
    "SurCam03": {
        "position_offset": np.array([4.0, -6.0, 2.8], dtype=np.float32),
        "forward": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "target_offset": np.array([2.0, 0.0, -0.5], dtype=np.float32),
        "focal_distance": 20.0,
        "view_angle": 95.0,
    },
    "SurCam04": {
        "position_offset": np.array([5.0, 0.0, 2.5], dtype=np.float32),
        "forward": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "target_offset": np.array([0.0, 0.0, -0.5], dtype=np.float32),
        "focal_distance": 20.0,
        "view_angle": 95.0,
    },
    "FrontCam02": {
        "position_offset": np.array([-4.0, 0.0, 2.6], dtype=np.float32),
        "forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "target_offset": np.array([0.0, 0.0, -0.25], dtype=np.float32),
        "focal_distance": 20.0,
        "view_angle": 80.0,
    },
}

_CAMERA_PRESET_DATASET_CONFIGS = {
    "nusc": {
        "voxel_size": [0.5, 0.5, 0.5],
        "vox_origin": [-50.0, -50.0, -5.0],
    },
    "kitti": {
        "voxel_size": [0.2, 0.2, 0.2],
        "vox_origin": [0.0, -25.6, -2.0],
    },
    "kitti360": {
        "voxel_size": [0.2, 0.2, 0.2],
        "vox_origin": [0.0, -25.6, -2.0],
    },
    "xc-cn": {
        "voxel_size": [0.1, 0.1, 0.1],
        "vox_origin": [-15.0, -15.0, -2.0],
        "origin_index": [150.0, 150.0, 20.0],
    },
}


def parse_timestamp_text(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def looks_like_timestamp(value: str) -> bool:
    try:
        parse_timestamp_text(value)
    except ValueError:
        return False
    return True


def is_timestamp_frame_dir(path: Path) -> bool:
    return path.is_dir() and looks_like_timestamp(path.name)


def list_timestamp_frame_dirs(clip_dir: Path) -> list[Path]:
    clip_dir = Path(clip_dir)
    return sorted(
        (path for path in clip_dir.iterdir() if is_timestamp_frame_dir(path)),
        key=lambda path: parse_timestamp_text(path.name),
    )


def is_complete_timestamp_frame_dir(frame_dir: Path) -> bool:
    frame_dir = Path(frame_dir)
    if not is_timestamp_frame_dir(frame_dir):
        return False

    files = [path for path in frame_dir.iterdir() if path.is_file()]
    if len(files) < EXPECTED_SOURCE_FRAME_FILE_COUNT:
        return False

    if not all((frame_dir / filename).is_file() for filename in FLAT_FRAME_FILENAMES.values()):
        return False

    json_files = [path for path in files if path.suffix.lower() == ".json"]
    if len(json_files) != 1:
        return False

    for camera_name in EXPECTED_CAMERA_NAMES:
        matches = [path for path in files if path.name.endswith(f"_{camera_name}.jpeg")]
        if len(matches) != 1:
            return False

    return True


def list_complete_timestamp_frame_dirs(clip_dir: Path) -> list[Path]:
    return [frame_dir for frame_dir in list_timestamp_frame_dirs(clip_dir) if is_complete_timestamp_frame_dir(frame_dir)]


def parse_timestamp_from_stem(path: Path) -> datetime:
    return parse_timestamp_text(path.stem)


def parse_timestamp_from_prefixed_stem(path: Path) -> datetime:
    return parse_timestamp_text(path.stem.split("_", 1)[0])


def get_camera_image_path(frame_dir: Path, camera_name: str = "FrontCam02") -> Path:
    frame_dir = Path(frame_dir)
    matches = sorted(path for path in frame_dir.glob(f"*_{camera_name}.jpeg") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {camera_name} image in {frame_dir}, found {len(matches)}")
    return matches[0]


def get_frame_json_path(frame_dir: Path) -> Path:
    frame_dir = Path(frame_dir)
    json_files = sorted(path for path in frame_dir.glob("*.json") if path.is_file())
    if len(json_files) != 1:
        raise FileNotFoundError(f"Expected exactly one JSON file in {frame_dir}, found {len(json_files)}")
    return json_files[0]


def load_frame_json(frame_dir: Path) -> dict:
    json_path = get_frame_json_path(frame_dir)
    return json.loads(json_path.read_text(encoding="utf-8"))


def _get_dynamic_calib_value(dynamic_calib: dict, camera_name: str) -> str:
    if camera_name.startswith("SurCam"):
        try:
            return dynamic_calib["SurCam"][camera_name]
        except KeyError as exc:
            raise KeyError(f"Missing extrinsics for {camera_name}") from exc

    if camera_name == "FrontCam02":
        front_cam_values = dynamic_calib.get("FrontCam")
        if not isinstance(front_cam_values, list) or len(front_cam_values) != 1:
            raise KeyError("Expected a single FrontCam extrinsic entry for FrontCam02")
        return front_cam_values[0]

    raise KeyError(f"Unsupported camera name for extrinsics lookup: {camera_name}")


def get_camera_extrinsic_matrix(frame_dir: Path, camera_name: str) -> np.ndarray:
    frame_metadata = load_frame_json(frame_dir)
    try:
        dynamic_calib = frame_metadata["external_info"]["dynamic_calib"]
    except KeyError as exc:
        raise KeyError(f"Missing external_info.dynamic_calib in {get_frame_json_path(frame_dir)}") from exc

    calibration_text = _get_dynamic_calib_value(dynamic_calib, camera_name)
    return np.asarray(ast.literal_eval(calibration_text), dtype=np.float32)


def _flip_plot_y_axis(values: np.ndarray) -> np.ndarray:
    flipped = np.asarray(values, dtype=np.float32).copy()
    flipped[..., 1] *= -1.0
    return flipped


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def _get_camera_preset_dataset_config(dataset: str) -> dict:
    try:
        return _CAMERA_PRESET_DATASET_CONFIGS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset for camera preset lookup: {dataset}") from exc


def _get_camera_preset_template(camera_name: str) -> dict:
    try:
        return _CAMERA_PRESET_LOOKUP[camera_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported camera name for lookup preset: {camera_name}") from exc


def _load_reference_occ_shape(
    frame_dir: Path,
    occupancy_key: str = "occ_voxel",
    transpose_zxy_to_xyz: bool = True,
) -> tuple[int, int, int]:
    occ_path = Path(frame_dir) / FLAT_FRAME_FILENAMES["occ"]
    with np.load(occ_path) as data:
        if occupancy_key not in data:
            available_keys = ", ".join(data.files)
            raise KeyError(f"Missing occupancy key '{occupancy_key}' in {occ_path}. Available keys: {available_keys}")
        occupancy = np.asarray(data[occupancy_key])

    if occupancy.ndim == 4:
        occupancy = occupancy[0]

    if transpose_zxy_to_xyz:
        occupancy = np.transpose(occupancy, (1, 2, 0))

    return tuple(int(size) for size in occupancy.shape)


def _get_plot_space_anchor(voxel_shape: tuple[int, int, int], dataset: str) -> np.ndarray:
    dataset_config = _get_camera_preset_dataset_config(dataset)
    voxel_size = np.asarray(dataset_config["voxel_size"], dtype=np.float32)
    vox_origin = np.asarray(dataset_config["vox_origin"], dtype=np.float32)
    origin_index = dataset_config.get("origin_index")

    if origin_index is None:
        dims = np.asarray(voxel_shape[:3], dtype=np.float32)
        anchor = vox_origin + (dims * voxel_size / 2.0)
    else:
        anchor = vox_origin + (np.asarray(origin_index, dtype=np.float32) * voxel_size)

    anchor[1] *= -1.0
    return anchor


def build_camera_preset_from_lookup(
    camera_name: str,
    voxel_shape: tuple[int, int, int],
    dataset: str = "xc-cn",
) -> dict:
    preset_template = _get_camera_preset_template(camera_name)
    forward_plot = _normalize_vector(preset_template["forward"])
    plot_anchor = _get_plot_space_anchor(voxel_shape, dataset=dataset)
    position = plot_anchor + np.asarray(preset_template["position_offset"], dtype=np.float32)

    focal_point = plot_anchor + np.asarray(preset_template["target_offset"], dtype=np.float32)
    focal_point = focal_point + (forward_plot * float(preset_template["focal_distance"]))

    return {
        "position": position.tolist(),
        "focal_point": focal_point.tolist(),
        "view_up": [0.0, 0.0, 1.0],
        "view_angle": float(preset_template["view_angle"]),
    }


def build_camera_preset_from_extrinsic(
    extrinsic_matrix: np.ndarray,
    camera_name: str,
) -> dict:
    preset_template = _get_camera_preset_template(camera_name)
    camera_to_ego = np.linalg.inv(np.asarray(extrinsic_matrix, dtype=np.float32))
    rotation = camera_to_ego[:3, :3]
    position = camera_to_ego[:3, 3]

    forward = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    view_up = rotation @ np.array([0.0, -1.0, 0.0], dtype=np.float32)

    position_plot = _flip_plot_y_axis(position)
    forward_plot = _normalize_vector(_flip_plot_y_axis(forward))
    view_up_plot = _normalize_vector(_flip_plot_y_axis(view_up))
    focal_point = position_plot + (forward_plot * float(preset_template["focal_distance"]))

    return {
        "position": position_plot.tolist(),
        "focal_point": focal_point.tolist(),
        "view_up": view_up_plot.tolist(),
        "view_angle": float(preset_template["view_angle"]),
    }


def get_camera_image_size(frame_dir: Path, camera_name: str) -> tuple[int, int]:
    image_path = get_camera_image_path(frame_dir, camera_name=camera_name)
    with Image.open(image_path) as image:
        return image.size


def scale_figure_size(image_size: tuple[int, int], max_side: int = DEFAULT_RENDER_MAX_SIDE) -> tuple[int, int]:
    width, height = image_size
    scale = min(1.0, max_side / max(width, height))
    return max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)


def get_clip_camera_render_settings(
    clip_dir: Path,
    camera_name: str,
    dataset: str = "xc-cn",
    occupancy_key: str = "occ_voxel",
    transpose_zxy_to_xyz: bool = True,
    max_side: int = DEFAULT_RENDER_MAX_SIDE,
) -> tuple[dict, tuple[int, int]]:
    frame_dirs = list_complete_timestamp_frame_dirs(clip_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No complete timestamp frame directories found in {clip_dir}")

    reference_frame = frame_dirs[0]
    voxel_shape = _load_reference_occ_shape(
        reference_frame,
        occupancy_key=occupancy_key,
        transpose_zxy_to_xyz=transpose_zxy_to_xyz,
    )
    camera_preset = build_camera_preset_from_lookup(
        camera_name=camera_name,
        voxel_shape=voxel_shape,
        dataset=dataset,
    )
    figure_size = scale_figure_size(get_camera_image_size(reference_frame, camera_name=camera_name), max_side=max_side)
    return camera_preset, figure_size


def _collect_image_paths(image_source: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(image_source, (str, Path)):
        source_path = Path(image_source)
        if source_path.is_dir():
            image_paths = sorted(path for path in source_path.glob("*.png") if path.is_file())
        elif source_path.is_file():
            image_paths = [source_path]
        else:
            raise FileNotFoundError(f"Image source does not exist: {source_path}")
    else:
        image_paths = [Path(path) for path in image_source]

    if not image_paths:
        raise FileNotFoundError("No PNG images found to merge.")

    return image_paths


def merge_image_sequence(image_source: Path | str | Iterable[Path | str], hz: float, output_path: Path | str | None = None) -> Path:
    image_paths = _collect_image_paths(image_source)

    if output_path is None:
        if isinstance(image_source, (str, Path)) and Path(image_source).is_dir():
            output_path = Path(image_source) / "sequence.mp4"
        else:
            raise ValueError("output_path is required when image_source is not a directory")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(output_path), fps=hz)
    try:
        for image_path in image_paths:
            writer.append_data(imageio.imread(image_path))
    finally:
        writer.close()

    return output_path