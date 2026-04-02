import argparse
from pathlib import Path

import imageio.v2 as imageio

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")


def _parse_args():
    parser = argparse.ArgumentParser(description="Merge image frames in a folder into an MP4 file.")
    parser.add_argument("image_dir", type=Path, help="Folder containing image frames.")
    parser.add_argument("hz", type=float, help="Frame rate of the output video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output MP4 path. Defaults to <image_dir>/../sequence.mp4.",
    )
    return parser.parse_args()


def _get_image_files(image_dir):
    image_files = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No supported image files found in {image_dir}")
    return image_files


def merge_image_sequence(image_dir, hz, output_path=None):
    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise NotADirectoryError(f"Image folder does not exist: {image_dir}")
    if hz <= 0:
        raise ValueError("HZ must be greater than 0.")

    image_files = _get_image_files(image_dir)
    output_path = output_path or (image_dir / ".." / "sequence.mp4")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=hz)
    try:
        total = len(image_files)
        for index, image_file in enumerate(image_files, start=1):
            writer.append_data(imageio.imread(image_file))
            print(f"\rEncoding video: {index}/{total}", end="", flush=True)
    finally:
        writer.close()

    print()
    print(f"Saved MP4 to {output_path.resolve()}")
    return output_path


def main():
    args = _parse_args()
    merge_image_sequence(args.image_dir, args.hz, args.output)


if __name__ == "__main__":
    main()
