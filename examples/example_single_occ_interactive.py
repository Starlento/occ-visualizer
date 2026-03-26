from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from example_moving_occ_sequence import LABEL_EMPTY, build_frame
from occupancy_visualizer import save_occ


OUTPUT_DIR = ROOT_DIR / "outputs"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_occ(
        save_dir=str(OUTPUT_DIR),
        occupancy=build_frame(frame_index=0),
        name="interactive_frame_000",
        sem=True,
        dataset="nusc",
        show=True,
        empty_label=LABEL_EMPTY,
    )

    print(f"Saved PNG to {(OUTPUT_DIR / 'interactive_frame_000.png').resolve()}")


if __name__ == "__main__":
    main()