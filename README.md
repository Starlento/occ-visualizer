# Occupancy Visualization Utilities

This repository contains a small set of tools for rendering occupancy grids as PNG images with Mayavi and stitching rendered PNG sequences into an MP4 video.

The repo currently includes:

- `occupancy_visualizer.py`: renders a single occupancy grid to a PNG.
- `colormaps.py`: dataset-specific semantic colormaps.
- `examples/example_moving_occ_sequence.py`: generates a simple synthetic scene sequence with a moving car.
- `examples/example_single_occ_interactive.py`: renders one frame and opens an interactive Mayavi window.
- `merge_png_sequence.py`: converts a folder of PNG frames into an MP4.

## Requirements

- Python `3.12`
- A working Mayavi installation for occupancy rendering
- `imageio` and `imageio-ffmpeg` for MP4 export

This workspace already uses a local virtual environment at `.venv`.

## Environment Setup

If you need to recreate the environment:

```powershell
uv venv --python 3.12 .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the base dependencies:

```powershell
python -m pip install numpy imageio imageio-ffmpeg
```

Install Mayavi separately according to your platform constraints.

## Render a Single Occupancy Grid

The main API is `save_occ` in `occupancy_visualizer.py`.

Expected input:

- `occupancy`: a `numpy.ndarray` or tensor-like object containing a single occupancy grid
- shape: typically `(X, Y, Z)` or `(1, X, Y, Z)`
- semantic mode: set `sem=True` when grid values are semantic class ids

Example:

```python
import numpy as np

from occupancy_visualizer import save_occ

grid = np.zeros((200, 200, 16), dtype=np.uint8)
grid[:, :, 0] = 11

save_occ(
    save_dir="outputs",
    occupancy=grid,
    name="frame_000",
    sem=True,
    dataset="nusc",
    empty_label=0,
)
```

## Generate the Example Sequence

The example script creates a simple synthetic scene:

- a flat ground plane
- one static pillar
- one car moving forward along the `+x` direction
- `10 Hz` for `5 seconds`
- total: `50` PNG frames

Run:

```powershell
python .\examples\example_moving_occ_sequence.py
```

By default, the script writes frames into:

```text
outputs/
```

The script shows a terminal progress bar while rendering.

## Interactive Single-Frame Example

Use this example when you want to inspect one occupancy frame interactively in a Mayavi window.

Run:

```powershell
python .\examples\example_single_occ_interactive.py
```

This example:

- builds the first frame from the synthetic scene
- saves `outputs/interactive_frame_000.png`
- opens an interactive Mayavi view via `save_occ(..., show=True)`

## Merge PNG Frames into MP4

Use `merge_png_sequence.py` to encode all `.png` files in a folder into an MP4.

Arguments:

- `png_dir`: path to the folder containing PNG files
- `hz`: output video frame rate
- `--output`: optional output MP4 path

Example:

```powershell
python .\merge_png_sequence.py .\outputs 10
```

Example with explicit output path:

```powershell
python .\merge_png_sequence.py .\outputs 10 --output .\outputs\moving_car.mp4
```

## Notes

- `save_occ` is designed for single-frame rendering. If you want a sequence, call it once per frame.
- Semantic rendering should pass `empty_label`, otherwise empty voxels may be rendered as a valid class.
- The current camera is a fixed oblique top-down view defined in `occupancy_visualizer.py`.
- If you want an interactive Mayavi window, call `save_occ(..., show=True)`.

## Current Workflow

The typical workflow in this repo is:

1. Generate or load occupancy grids.
2. Render them to PNG with `save_occ`.
3. Merge the PNG folder into MP4 with `merge_png_sequence.py`.