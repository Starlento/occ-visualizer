# Occupancy Visualization Utilities

This repository renders occupancy grids with Mayavi and builds comparison videos from timestamp-aligned frame folders.

## Main Files

- `occupancy_visualizer.py`: low-level single-frame Mayavi renderer.
- `render_occ_npz_sequence.py`: renders `occ.npz` into `occ_vis.png` inside each timestamp folder.
- `pipeline.py`: builds a merged video from a chosen camera image and `occ_vis.png` in each timestamp folder.
- `utils.py`: timestamp-frame helpers and PNG-to-MP4 utilities.
- `colormaps.py`: dataset-specific semantic colormaps.
- `examples/example_single_occ_interactive.py`: interactive single-frame occ debug entry point.

## Requirements

- Python `3.12`
- A working Mayavi installation for occupancy rendering
- `PyQt5` on Windows for interactive Mayavi windows
- `imageio` and `imageio-ffmpeg` for MP4 export

This workspace already uses a local virtual environment at `.venv`.

## Data Layout

Each clip directory must use the flat per-timestamp layout. Every timestamp folder must contain the 10 source files: `occ.npz`, `freespace.png`, `freespace_height.npy`, `freespace_kpts.npy`, 5 camera JPEGs, and 1 frame JSON file. Generated intermediates such as `occ_vis.png` can live in the same folder.

Example layout:

```text
data/<clip>/
    20231125105257.100000/
        occ.npz
        freespace.png
        freespace_height.npy
        freespace_kpts.npy
        20231125105257.164375_FrontCam02.jpeg
        20231125105257.156691_SurCam02.jpeg
        11143C_20231125105257.100000.json
        occ_vis.png
```

`render_occ_npz_sequence.py` and `pipeline.py` expect this layout only.

## Render Per-Frame Occ Visualization

Render `occ_vis.png` into each timestamp folder:

```powershell
python .\render_occ_npz_sequence.py .\data\4af3e12ea8ace222e59743f5c1370a12
```

To use a different output base name:

```powershell
python .\render_occ_npz_sequence.py .\data\4af3e12ea8ace222e59743f5c1370a12 --output-name custom_occ_vis
```

## Interactive Debug Render

`examples/example_single_occ_interactive.py` is a hard-coded single-frame debug script. Edit the constants at the top of the file, then run it to open one timestamp frame in an interactive Mayavi window.

```powershell
python .\examples\example_single_occ_interactive.py
```

Set `SHOW_WINDOW = False` in the script when you only want a debug PNG without opening the Mayavi window.
Set `CAMERA_NAME` in the script to switch the debug view to a different fixed per-camera preset.
If you need to tune viewpoint, edit the preset config in `utils.py` instead of overriding values in the script.

## Build the Merged Video

Generate `merged.mp4` from a camera image and `occ_vis.png` in each timestamp folder:

```powershell
python .\pipeline.py .\data\4af3e12ea8ace222e59743f5c1370a12 10 --camera-name FrontCam02
```

`--camera-name` now controls both the left image selection and the occ render viewpoint. The right-side render uses a fixed per-camera preset template: `SurCam01` left, `SurCam02` front, `SurCam03` right, `SurCam04` back, and `FrontCam02` front.

Reuse existing `occ_vis.png` files instead of re-rendering them:

```powershell
python .\pipeline.py .\data\4af3e12ea8ace222e59743f5c1370a12 10 --reuse-occ-vis
```

## Merge PNG Frames into MP4

Use `merge_image_sequence` from `utils.py` to encode `.png` frames into an MP4.

```python
from pathlib import Path

from utils import merge_image_sequence

merge_image_sequence(Path("outputs"), 10, Path("outputs/moving_car.mp4"))
```

## Notes

- `save_occ` is designed for single-frame rendering. If you want a sequence, call it once per frame.
- `render_occ_npz_sequence.py` writes `occ_vis.png` into each timestamp folder by default.
- Semantic rendering reads `empty_labels` from the selected dataset config in `occupancy_visualizer.py`.
- When `camera_name` is provided, the render viewpoint is derived from a fixed per-camera preset template. For `xc-cn`, the preset is anchored at the dataset origin index and then shifted by per-camera position and target offsets.
- If you want an interactive Mayavi window, call `save_occ(..., show=True)`.