from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from occupancy_visualizer import save_occ
from utils import (
	EXPECTED_CAMERA_NAMES,
	FLAT_FRAME_FILENAMES,
	build_camera_preset_from_lookup,
	get_camera_image_size,
	scale_figure_size,
)


FRAME_DIR = ROOT_DIR / "data" / "4af3e12ea8ace222e59743f5c1370a12" / "20231125105257.100000"
CAMERA_NAME = "SurCam02"
CAMERA_VIEW_ANGLE = 100.0
CAMERA_CENTER_Z_CELL_OFFSET = 5.0
CAMERA_HEIGHT_OFFSET = 0
CAMERA_TARGET_HEIGHT_OFFSET = 0
DATASET = "xc-cn"
OCCUPANCY_KEY = "occ_voxel"
OUTPUT_NAME = "occ_vis_interactive"
EMPTY_LABEL = 0
TRANSPOSE_ZXY_TO_XYZ = True
SHOW_WINDOW = True


def _load_occupancy(npz_path: Path, occupancy_key: str, transpose_zxy_to_xyz: bool) -> np.ndarray:
	with np.load(npz_path) as data:
		if occupancy_key not in data:
			available_keys = ", ".join(data.files)
			raise KeyError(f"Missing occupancy key '{occupancy_key}' in {npz_path}. Available keys: {available_keys}")
		occupancy = np.asarray(data[occupancy_key])

	if transpose_zxy_to_xyz:
		occupancy = np.transpose(occupancy, (1, 2, 0))

	return occupancy


def _print_frame_summary(frame_dir: Path) -> None:
	print(f"Frame directory: {frame_dir}")
	print(f"Occupancy file: {frame_dir / FLAT_FRAME_FILENAMES['occ']}")
	json_files = sorted(frame_dir.glob("*.json"))
	if json_files:
		print(f"JSON file: {json_files[0]}")

	print("Camera images:")
	for camera_name in EXPECTED_CAMERA_NAMES:
		matches = sorted(frame_dir.glob(f"*_{camera_name}.jpeg"))
		if matches:
			print(f"  {camera_name}: {matches[0]}")


def main() -> None:
	frame_dir = FRAME_DIR.resolve()
	if not frame_dir.is_dir():
		raise NotADirectoryError(f"Frame directory not found: {frame_dir}")

	occ_path = frame_dir / FLAT_FRAME_FILENAMES["occ"]
	if not occ_path.is_file():
		raise FileNotFoundError(f"Occupancy file not found: {occ_path}")

	output_path = frame_dir / f"{OUTPUT_NAME}.png"

	_print_frame_summary(frame_dir)

	occupancy = _load_occupancy(
		occ_path,
		occupancy_key=OCCUPANCY_KEY,
		transpose_zxy_to_xyz=TRANSPOSE_ZXY_TO_XYZ,
	)
	camera_preset = build_camera_preset_from_lookup(
		camera_name=CAMERA_NAME,
		voxel_shape=occupancy.shape,
		dataset=DATASET,
		view_angle=CAMERA_VIEW_ANGLE,
		camera_center_z_cell_offset=CAMERA_CENTER_Z_CELL_OFFSET,
		camera_height_offset=CAMERA_HEIGHT_OFFSET,
		camera_target_height_offset=CAMERA_TARGET_HEIGHT_OFFSET,
	)
	figure_size = scale_figure_size(get_camera_image_size(frame_dir, camera_name=CAMERA_NAME))

	save_occ(
		save_dir=str(frame_dir),
		occupancy=occupancy,
		name=OUTPUT_NAME,
		sem=True,
		dataset=DATASET,
		empty_label=EMPTY_LABEL,
		show=SHOW_WINDOW,
		camera_preset=camera_preset,
		figure_size=figure_size,
	)
	print(f"Saved interactive debug render to {output_path}")


if __name__ == "__main__":
	main()
