import argparse
from datetime import datetime
from statistics import median
from pathlib import Path

from pipeline import run_pipeline


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline for all clip directories under data/.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing clip subdirectories. Defaults to ./data.",
    )
    parser.add_argument(
        "--reuse-occ-png",
        action="store_true",
        help="Reuse existing occ_png frames instead of rendering them again.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one clip fails.",
    )
    return parser.parse_args()


def _get_clip_dirs(data_dir: Path) -> list[Path]:
    clip_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not clip_dirs:
        raise FileNotFoundError(f"No clip directories found in {data_dir}")
    return clip_dirs


def _parse_occ_timestamp(npz_path: Path) -> datetime:
    return datetime.strptime(npz_path.stem, "%Y%m%d%H%M%S.%f")


def _infer_hz_from_occ_dir(occ_dir: Path) -> float:
    npz_files = sorted(path for path in occ_dir.glob("*.npz") if path.is_file())
    if len(npz_files) < 2:
        raise FileNotFoundError(f"Need at least 2 NPZ files to infer HZ in {occ_dir}")

    timestamps = [_parse_occ_timestamp(path) for path in npz_files]
    intervals = [
        (current - previous).total_seconds()
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    if not intervals:
        raise ValueError(f"Unable to infer HZ from timestamps in {occ_dir}")

    frame_interval = median(intervals)
    if frame_interval <= 0:
        raise ValueError(f"Invalid frame interval inferred from {occ_dir}: {frame_interval}")

    return 1.0 / frame_interval


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    clip_dirs = _get_clip_dirs(data_dir)
    failures: list[tuple[Path, str]] = []

    for index, clip_dir in enumerate(clip_dirs, start=1):
        print(f"[{index}/{len(clip_dirs)}] Processing {clip_dir.name}...")
        try:
            hz = _infer_hz_from_occ_dir(clip_dir / "occ")
            print(f"Inferred HZ: {hz:.3f}")
            run_pipeline(
                clip_dir,
                hz,
                regenerate_occ_png=not args.reuse_occ_png,
            )
        except Exception as exc:
            failures.append((clip_dir, str(exc)))
            print(f"Failed: {clip_dir.name}: {exc}")
            if args.fail_fast:
                raise

    if failures:
        print("\nCompleted with failures:")
        for clip_dir, message in failures:
            print(f"- {clip_dir.name}: {message}")
        raise SystemExit(1)

    print(f"\nCompleted successfully for {len(clip_dirs)} clip(s).")


if __name__ == "__main__":
    main()