import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from statistics import median
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"


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
        "--workers",
        type=int,
        default=1,
        help="Number of clips to process in parallel. Defaults to 1.",
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


def _extract_merged_video_to_data_root(data_dir: Path, clip_dir: Path) -> None:
    source = clip_dir / "merged.mp4"
    if not source.is_file():
        raise FileNotFoundError(f"Expected merged video not found: {source}")

    destination = data_dir / f"{clip_dir.name}.mp4"
    source.replace(destination)
    print(f"Moved merged video to: {destination.name}")


def _run_pipeline_subprocess(clip_dir: Path, hz: float, reuse_occ_png: bool) -> None:
    command = [
        sys.executable,
        str(ROOT_DIR / "pipeline.py"),
        str(clip_dir),
        str(hz),
    ]
    if reuse_occ_png:
        command.append("--reuse-occ-png")

    process = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        tail = (process.stderr or process.stdout or "").strip().splitlines()[-10:]
        detail = "\n".join(tail)
        raise RuntimeError(f"pipeline.py failed (exit={process.returncode})\n{detail}")


def _process_clip(data_dir: Path, clip_dir: Path, reuse_occ_png: bool) -> tuple[Path, float]:
    hz = _infer_hz_from_occ_dir(clip_dir / "occ")
    _run_pipeline_subprocess(clip_dir, hz, reuse_occ_png)
    _extract_merged_video_to_data_root(data_dir, clip_dir)
    return clip_dir, hz


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    clip_dirs = _get_clip_dirs(data_dir)
    failures: list[tuple[Path, str]] = []
    workers = max(1, args.workers)

    if workers == 1:
        for index, clip_dir in enumerate(clip_dirs, start=1):
            print(f"[{index}/{len(clip_dirs)}] Processing {clip_dir.name}...")
            try:
                _, hz = _process_clip(data_dir, clip_dir, args.reuse_occ_png)
                print(f"Inferred HZ: {hz:.3f}")
            except Exception as exc:
                failures.append((clip_dir, str(exc)))
                print(f"Failed: {clip_dir.name}: {exc}")
                if args.fail_fast:
                    raise
    else:
        print(f"Running with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_process_clip, data_dir, clip_dir, args.reuse_occ_png): clip_dir
                for clip_dir in clip_dirs
            }
            completed = 0
            for future in as_completed(future_map):
                clip_dir = future_map[future]
                completed += 1
                try:
                    _, hz = future.result()
                    print(f"[{completed}/{len(clip_dirs)}] Done {clip_dir.name} (HZ={hz:.3f})")
                except Exception as exc:
                    failures.append((clip_dir, str(exc)))
                    print(f"[{completed}/{len(clip_dirs)}] Failed {clip_dir.name}: {exc}")
                    if args.fail_fast:
                        for pending in future_map:
                            pending.cancel()
                        raise

    if failures:
        print("\nCompleted with failures:")
        for clip_dir, message in failures:
            print(f"- {clip_dir.name}: {message}")
        raise SystemExit(1)

    print(f"\nCompleted successfully for {len(clip_dirs)} clip(s).")


if __name__ == "__main__":
    main()