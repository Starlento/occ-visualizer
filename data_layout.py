from __future__ import annotations

from datetime import datetime
from pathlib import Path


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

EXPECTED_FRAME_FILE_COUNT = len(FLAT_FRAME_FILENAMES) + len(EXPECTED_CAMERA_NAMES) + 1


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
    if len(files) != EXPECTED_FRAME_FILE_COUNT:
        return False

    if not all((frame_dir / filename).is_file() for filename in FLAT_FRAME_FILENAMES.values()):
        return False

    json_files = [path for path in files if path.suffix.lower() == ".json"]
    if len(json_files) != 1:
        return False

    for camera_name in EXPECTED_CAMERA_NAMES:
        if not any(path.name.endswith(f"_{camera_name}.jpeg") for path in files):
            return False

    return True


def list_complete_timestamp_frame_dirs(clip_dir: Path) -> list[Path]:
    return [frame_dir for frame_dir in list_timestamp_frame_dirs(clip_dir) if is_complete_timestamp_frame_dir(frame_dir)]


def parse_timestamp_from_stem(path: Path) -> datetime:
    return parse_timestamp_text(path.stem)


def parse_timestamp_from_prefixed_stem(path: Path) -> datetime:
    return parse_timestamp_text(path.stem.split("_", 1)[0])
