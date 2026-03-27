import argparse
import sys
from pathlib import Path

import numpy as np

from occupancy_visualizer import save_occ


DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parent
    / "2a548472d8f5713112606b241325839f"
    / "occ"
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Render a directory of occupancy NPZ files to PNG frames.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per-frame occupancy NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write PNG frames into. Defaults to <input_dir>/../occ_png.",
    )
    parser.add_argument(
        "--dataset",
        default="xc-cn",
        help="Semantic dataset key for rendering. Defaults to xc-cn.",
    )
    parser.add_argument(
        "--empty-label",
        type=int,
        default=0,
        help="Semantic label to treat as empty space. Defaults to 0.",
    )
    parser.add_argument(
        "--occupancy-key",
        default="occ_voxel",
        help="NPZ key that contains the occupancy grid. Defaults to occ_voxel.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of frames to render.",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Disable the default transpose from (Z, X, Y) to (X, Y, Z).",
    )
    return parser.parse_args()


def _print_progress(current, total):
    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rRendering frames: [{bar}] {current}/{total}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def _load_occupancy(npz_path, occupancy_key, transpose_zxy_to_xyz):
    with np.load(npz_path) as data:
        if occupancy_key not in data:
            available_keys = ", ".join(data.files)
            raise KeyError(f"Missing occupancy key '{occupancy_key}' in {npz_path}. Available keys: {available_keys}")
        occupancy = np.asarray(data[occupancy_key])

    if transpose_zxy_to_xyz:
        occupancy = np.transpose(occupancy, (1, 2, 0))

    return occupancy


def render_occ_npz_sequence(
    input_dir,
    output_dir=None,
    dataset="xc-cn",
    empty_label=0,
    occupancy_key="occ_voxel",
    limit=None,
    transpose_zxy_to_xyz=True,
):
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Occupancy directory does not exist: {input_dir}")

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {input_dir}")

    if limit is not None:
        npz_files = npz_files[:limit]

    output_dir = Path(output_dir) if output_dir is not None else input_dir.parent / "occ_png"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(npz_files)
    for index, npz_path in enumerate(npz_files, start=1):
        occupancy = _load_occupancy(
            npz_path,
            occupancy_key=occupancy_key,
            transpose_zxy_to_xyz=transpose_zxy_to_xyz,
        )
        save_occ(
            save_dir=str(output_dir),
            occupancy=occupancy,
            name=npz_path.stem,
            sem=True,
            dataset=dataset,
            empty_label=empty_label,
        )
        _print_progress(index, total)

    print(f"Generated {total} PNG frames in {output_dir.resolve()}")
    return output_dir


def main():
    args = _parse_args()
    render_occ_npz_sequence(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        empty_label=args.empty_label,
        occupancy_key=args.occupancy_key,
        limit=args.limit,
        transpose_zxy_to_xyz=not args.no_transpose,
    )


if __name__ == "__main__":
    main()