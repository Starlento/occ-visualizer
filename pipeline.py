"""
Full pipeline: render occ PNGs, extract front cam images, generate both videos,
horizontally concatenate them, and clean up the individual videos.

Usage:
    python pipeline.py <dir> <hz>

Example:
    python pipeline.py ./data/4af3e12ea8ace222e59743f5c1370a12 10
"""

import argparse
from bisect import bisect_left
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image

from render_occ_npz_sequence import render_occ_npz_sequence
from extract_from_view_image import extract_front_cam


def _parse_occ_timestamp(path: Path) -> float:
    return float(path.stem)


def _parse_front_cam_timestamp(path: Path) -> float:
    return float(path.stem.split("_", 1)[0])


def _get_occ_png_files(occ_png_dir: Path) -> list[Path]:
    png_files = sorted(
        (path for path in occ_png_dir.glob("*.png") if path.is_file()),
        key=_parse_occ_timestamp,
    )
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {occ_png_dir}")
    return png_files


def _get_front_cam_files(front_cam_dir: Path) -> list[Path]:
    image_files = sorted(
        (path for path in front_cam_dir.glob("*_FrontCam02.jpeg") if path.is_file()),
        key=_parse_front_cam_timestamp,
    )
    if not image_files:
        raise FileNotFoundError(f"No FrontCam02 images found in {front_cam_dir}")
    return image_files


def _find_nearest_front_cam(occ_timestamp: float, front_timestamps: list[float], front_files: list[Path]) -> Path:
    index = bisect_left(front_timestamps, occ_timestamp)
    if index <= 0:
        return front_files[0]
    if index >= len(front_files):
        return front_files[-1]

    prev_timestamp = front_timestamps[index - 1]
    next_timestamp = front_timestamps[index]
    if abs(prev_timestamp - occ_timestamp) <= abs(next_timestamp - occ_timestamp):
        return front_files[index - 1]
    return front_files[index]


def _concat_images_horizontal(left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
    left_height = left_frame.shape[0]
    right_height = right_frame.shape[0]
    if left_height != right_height:
        new_width = int(right_frame.shape[1] * left_height / right_height)
        right_frame = np.array(Image.fromarray(right_frame).resize((new_width, left_height), Image.LANCZOS))
    return np.concatenate([left_frame, right_frame], axis=1)


def _generate_merged_video_from_images(front_cam_dir: Path, occ_png_dir: Path, hz: float, output_path: Path) -> None:
    occ_png_files = _get_occ_png_files(occ_png_dir)
    front_cam_files = _get_front_cam_files(front_cam_dir)
    front_timestamps = [_parse_front_cam_timestamp(path) for path in front_cam_files]

    writer = imageio.get_writer(str(output_path), fps=hz)
    try:
        total = len(occ_png_files)
        for index, occ_png_path in enumerate(occ_png_files, start=1):
            occ_timestamp = _parse_occ_timestamp(occ_png_path)
            front_cam_path = _find_nearest_front_cam(occ_timestamp, front_timestamps, front_cam_files)

            front_frame = imageio.imread(front_cam_path)
            occ_frame = imageio.imread(occ_png_path)
            occ_frame = np.fliplr(occ_frame)
            merged_frame = _concat_images_horizontal(front_frame, occ_frame)
            writer.append_data(merged_frame)
            print(f"\rMerging frames: {index}/{total}", end="", flush=True)
    finally:
        writer.close()
    print()


def run_pipeline(data_dir: Path, hz: float, regenerate_occ_png: bool = True) -> None:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {data_dir}")

    occ_dir = data_dir / "occ"
    occ_png_dir = data_dir / "occ_png"
    front_cam_dir = data_dir / "front_cam"
    merged_video = data_dir / "merged.mp4"
    legacy_sequence_video = data_dir / "sequence.mp4"

    # 1. Render occ NPZ -> PNG frames
    if regenerate_occ_png:
        print("==> Rendering occ PNGs...")
        render_occ_npz_sequence(input_dir=occ_dir, output_dir=occ_png_dir, hz=hz, merge_video=False)
    else:
        if not occ_png_dir.is_dir():
            raise FileNotFoundError(f"occ_png directory not found: {occ_png_dir}")
        if not any(occ_png_dir.glob("*.png")):
            raise FileNotFoundError(f"No PNG files found in existing occ_png directory: {occ_png_dir}")
        print(f"==> Reusing existing occ PNGs in {occ_png_dir}...")

    # 2. Extract FrontCam02 images
    print("==> Extracting front cam images...")
    extract_front_cam(data_dir)

    # 3. Merge images using nearest timestamp alignment
    print("==> Generating merged video from aligned images...")
    _generate_merged_video_from_images(front_cam_dir, occ_png_dir, hz, merged_video)

    # 4. Remove legacy outputs if present
    if legacy_sequence_video.exists():
        legacy_sequence_video.unlink()

    print(f"Done. Saved merged video to {merged_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full occ visualizer pipeline.")
    parser.add_argument("dir", type=Path, help="Data directory (e.g. ./data/<hash>).")
    parser.add_argument("hz", type=float, help="Frame rate for all generated videos.")
    parser.add_argument(
        "--reuse-occ-png",
        action="store_true",
        help="Reuse existing occ_png frames instead of rendering them again.",
    )
    args = parser.parse_args()

    run_pipeline(args.dir, args.hz, regenerate_occ_png=not args.reuse_occ_png)


if __name__ == "__main__":
    main()
