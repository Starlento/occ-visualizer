"""
Full pipeline: render per-frame occ visualizations inside timestamp folders and
generate a merged comparison video.

Usage:
    python pipeline.py <dir> <hz>

Example:
    python pipeline.py ./data/4af3e12ea8ace222e59743f5c1370a12 10 --camera-name SurCam01
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from render_occ_npz_sequence import render_occ_npz_sequence
from utils import get_camera_image_path, list_complete_timestamp_frame_dirs


DEFAULT_OCC_VIS_FILENAME = "occ_vis.png"
DEFAULT_CAMERA_NAME = "FrontCam02"


def _get_frame_dirs(data_dir: Path) -> list[Path]:
    frame_dirs = list_complete_timestamp_frame_dirs(data_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No complete timestamp frame directories found in {data_dir}")
    return frame_dirs


def _get_occ_vis_path(frame_dir: Path, occ_vis_filename: str) -> Path:
    occ_vis_path = frame_dir / occ_vis_filename
    if not occ_vis_path.is_file():
        raise FileNotFoundError(f"Missing {occ_vis_filename} in {frame_dir}")
    return occ_vis_path


def _concat_images_horizontal(left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
    left_height = left_frame.shape[0]
    right_height = right_frame.shape[0]
    if left_height != right_height:
        new_width = int(right_frame.shape[1] * left_height / right_height)
        right_frame = np.array(Image.fromarray(right_frame).resize((new_width, left_height), Image.LANCZOS))
    return np.concatenate([left_frame, right_frame], axis=1)


def _generate_merged_video_from_frames(
    frame_dirs: list[Path],
    hz: float,
    output_path: Path,
    camera_name: str,
    occ_vis_filename: str,
) -> None:
    writer = imageio.get_writer(str(output_path), fps=hz)
    try:
        total = len(frame_dirs)
        for index, frame_dir in enumerate(frame_dirs, start=1):
            front_frame = imageio.imread(get_camera_image_path(frame_dir, camera_name=camera_name))
            occ_frame = imageio.imread(_get_occ_vis_path(frame_dir, occ_vis_filename))
            merged_frame = _concat_images_horizontal(front_frame, occ_frame)
            writer.append_data(merged_frame)
            print(f"\rMerging frames: {index}/{total}", end="", flush=True)
    finally:
        writer.close()
    print()


def run_pipeline(
    data_dir: Path,
    hz: float,
    camera_name: str = DEFAULT_CAMERA_NAME,
    regenerate_occ_vis: bool = True,
    occ_vis_filename: str = DEFAULT_OCC_VIS_FILENAME,
) -> None:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {data_dir}")

    frame_dirs = _get_frame_dirs(data_dir)
    merged_video = data_dir / "merged.mp4"

    if regenerate_occ_vis:
        print("==> Rendering occ visualizations...")
        render_occ_npz_sequence(
            input_dir=data_dir,
            output_name=Path(occ_vis_filename).stem,
            camera_name=camera_name,
            hz=hz,
            merge_video=False,
        )
    else:
        missing_occ_vis = [frame_dir for frame_dir in frame_dirs if not (frame_dir / occ_vis_filename).is_file()]
        if missing_occ_vis:
            raise FileNotFoundError(
                f"Missing {occ_vis_filename} in {len(missing_occ_vis)} timestamp folders. "
                f"First missing folder: {missing_occ_vis[0]}"
            )
        print(f"==> Reusing existing {occ_vis_filename} files in timestamp folders...")

    print("==> Generating merged video from aligned images...")
    _generate_merged_video_from_frames(
        frame_dirs,
        hz,
        merged_video,
        camera_name=camera_name,
        occ_vis_filename=occ_vis_filename,
    )

    print(f"Done. Saved merged video to {merged_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full occ visualizer pipeline.")
    parser.add_argument("dir", type=Path, help="Data directory (e.g. ./data/<hash>).")
    parser.add_argument("hz", type=float, help="Frame rate for all generated videos.")
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_CAMERA_NAME,
        help="Camera image to place on the left side and fixed directional preset to use for occ rendering. Defaults to FrontCam02.",
    )
    parser.add_argument(
        "--reuse-occ-vis",
        dest="reuse_occ_vis",
        action="store_true",
        help="Reuse existing per-frame occ_vis.png files instead of rendering them again.",
    )
    parser.add_argument(
        "--reuse-occ-png",
        dest="reuse_occ_vis",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    run_pipeline(
        args.dir,
        args.hz,
        camera_name=args.camera_name,
        regenerate_occ_vis=not args.reuse_occ_vis,
    )


if __name__ == "__main__":
    main()
