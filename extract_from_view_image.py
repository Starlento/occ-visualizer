"""
Extract all *_FrontCam02.jpeg files from a data directory into a flat front_cam/ subfolder.

Usage:
    python extract_from_view_image.py <dir>

Example:
    python extract_from_view_image.py ./data/4af3e12ea8ace222e59743f5c1370a12
"""

import argparse
import shutil
from pathlib import Path


def extract_front_cam(data_dir: Path) -> None:
    out_dir = data_dir / "front_cam"
    out_dir.mkdir(exist_ok=True)

    files = sorted(
        path
        for path in data_dir.rglob("*_FrontCam02.jpeg")
        if out_dir not in path.parents
    )
    if not files:
        print(f"No *_FrontCam02.jpeg files found in {data_dir}")
        return

    for src in files:
        dst = out_dir / src.name
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)

    print(f"Copied {len(files)} file(s) to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FrontCam02 images into front_cam/")
    parser.add_argument("dir", type=Path, help="Target data directory")
    args = parser.parse_args()

    data_dir = args.dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Directory not found: {data_dir}")

    extract_front_cam(data_dir)


if __name__ == "__main__":
    main()
