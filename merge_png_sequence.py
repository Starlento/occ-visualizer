import argparse
from pathlib import Path

import imageio.v2 as imageio


def _parse_args():
    parser = argparse.ArgumentParser(description="Merge PNG frames in a folder into an MP4 file.")
    parser.add_argument("png_dir", type=Path, help="Folder containing PNG frames.")
    parser.add_argument("hz", type=float, help="Frame rate of the output video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output MP4 path. Defaults to <png_dir>/../sequence.mp4.",
    )
    return parser.parse_args()


def _get_png_files(png_dir):
    png_files = sorted(path for path in png_dir.glob("*.png") if path.is_file())
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {png_dir}")
    return png_files


def merge_png_sequence(png_dir, hz, output_path=None):
    png_dir = Path(png_dir)
    if not png_dir.exists() or not png_dir.is_dir():
        raise NotADirectoryError(f"PNG folder does not exist: {png_dir}")
    if hz <= 0:
        raise ValueError("HZ must be greater than 0.")

    png_files = _get_png_files(png_dir)
    output_path = output_path or (png_dir / ".." / "sequence.mp4")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=hz)
    try:
        total = len(png_files)
        for index, png_file in enumerate(png_files, start=1):
            writer.append_data(imageio.imread(png_file))
            print(f"\rEncoding video: {index}/{total}", end="", flush=True)
    finally:
        writer.close()

    print()
    print(f"Saved MP4 to {output_path.resolve()}")
    return output_path


def main():
    args = _parse_args()
    merge_png_sequence(args.png_dir, args.hz, args.output)


if __name__ == "__main__":
    main()