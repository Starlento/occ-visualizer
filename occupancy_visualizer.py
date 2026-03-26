import os
from importlib import import_module

import numpy as np

from colormaps import get_kitti360_colormap, get_kitti_colormap, get_nuscenes_colormap

_OFFSCREEN = False
_DATASET_CONFIGS = {
    "nusc": {
        "voxel_size": [0.5, 0.5, 0.5],
        "vox_origin": [-50.0, -50.0, -5.0],
        "vmin": 0,
        "vmax": 16,
        "semantic_upper_bound": 17,
        "semantic_colormap": get_nuscenes_colormap(),
    },
    "kitti": {
        "voxel_size": [0.2, 0.2, 0.2],
        "vox_origin": [0.0, -25.6, -2.0],
        "vmin": 1,
        "vmax": 19,
        "semantic_upper_bound": 20,
        "semantic_colormap": (get_kitti_colormap()[1:, :] * 255).astype(np.uint8),
    },
    "kitti360": {
        "voxel_size": [0.2, 0.2, 0.2],
        "vox_origin": [0.0, -25.6, -2.0],
        "vmin": 1,
        "vmax": 18,
        "semantic_upper_bound": 19,
        "semantic_colormap": (get_kitti360_colormap()[1:, :] * 255).astype(np.uint8),
    },
}
_CAMERA_PRESET = {
    "position": [-107.15500034628069, -0.008333206176756742, 92.16667026873841],
    "focal_point": [0.008333206176757812, -0.008333206176757812, 1.399999976158142],
    "view_angle": 30.0,
    "view_up": [0.6463156430702277, -6.4549254142909245e-18, 0.7630701733934555],
    "clipping_range": [78.84362692774403, 218.2948716014858],
}


def _start_virtual_display():
    global _OFFSCREEN

    if os.environ.get("MAYAVI_OFFSCREEN", "").lower() in {"1", "true", "yes"}:
        _OFFSCREEN = True
        return

    if os.environ.get("DISPLAY"):
        _OFFSCREEN = False
        return

    if os.name == "nt":
        _OFFSCREEN = False
        return


def _get_mlab(show=False):
    _start_virtual_display()
    if show and os.environ.get("MAYAVI_OFFSCREEN", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("Interactive rendering requested, but MAYAVI_OFFSCREEN forces offscreen mode.")

    if not show:
        os.environ.setdefault("ETS_TOOLKIT", "null")

    try:
        mlab = import_module("mayavi.mlab")
    except Exception as exc:
        raise ImportError("Mayavi is required for occupancy visualization.") from exc

    mlab.options.offscreen = not show or _OFFSCREEN
    return mlab


def _should_render_offscreen(show):
    return (not show) or _OFFSCREEN


def get_grid_coords(dims, resolution):
    """Return voxel center coordinates for a dense grid."""

    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
    return (coords_grid * resolution) + resolution / 2


def _get_dataset_config(dataset):
    try:
        return _DATASET_CONFIGS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def _as_numpy_array(occupancy):
    if hasattr(occupancy, "detach"):
        occupancy = occupancy.detach().cpu().numpy()
    return np.asarray(occupancy)


def _prepare_voxels(occupancy, sem, cap):
    voxels = _as_numpy_array(occupancy)
    if voxels.ndim == 4:
        voxels = voxels[0]
    voxels = voxels.astype(np.int32, copy=True)

    if sem:
        return voxels

    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1
    voxels[..., (-cap):] = 0
    for z_index in range(voxels.shape[-1] - cap):
        mask = (voxels > 0)[..., z_index]
        voxels[..., z_index][mask] = z_index + 1
    return voxels


def _build_grid_with_values(voxels, dataset_config):
    grid_coords = get_grid_coords(voxels.shape, dataset_config["voxel_size"])
    grid_coords = grid_coords + np.array(dataset_config["vox_origin"], dtype=np.float32).reshape([1, 3])
    return np.vstack([grid_coords.T, voxels.reshape(-1)]).T


def _filter_visible_voxels(grid_coords, sem, dataset_config, empty_label):
    values = grid_coords[:, 3]

    if not sem:
        return grid_coords[(values > 0) & (values < 100)]

    visible_mask = (values >= 0) & (values < dataset_config["semantic_upper_bound"])
    if empty_label is not None:
        visible_mask &= values != empty_label
    elif dataset_config["vmin"] > 0:
        visible_mask &= values > 0
    return grid_coords[visible_mask]


def _create_figure(mlab, show):
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    figure.scene.off_screen_rendering = _should_render_offscreen(show)
    return figure


def _build_points_kwargs(dataset_config, sem):
    points_kwargs = {
        "scale_factor": sum(dataset_config["voxel_size"]) / 3,
        "mode": "cube",
        "opacity": 1.0,
    }
    if sem:
        points_kwargs.update({"vmin": dataset_config["vmin"], "vmax": dataset_config["vmax"]})
    else:
        points_kwargs.update({"colormap": "jet"})
    return points_kwargs


def _apply_semantic_lut(plot, dataset_config):
    plot.module_manager.scalar_lut_manager.lut.table = dataset_config["semantic_colormap"]


def _apply_camera_path(scene):
    scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
    scene.camera.focal_point = _CAMERA_PRESET["focal_point"]
    scene.camera.view_angle = _CAMERA_PRESET["view_angle"]
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
    scene.camera.compute_view_plane_normal()
    scene.render()

    for _ in range(26):
        scene.camera.azimuth(5)
        scene.render()

    scene.camera.azimuth(-135)
    scene.render()
    scene.camera.position = _CAMERA_PRESET["position"]
    scene.camera.focal_point = _CAMERA_PRESET["focal_point"]
    scene.camera.view_angle = _CAMERA_PRESET["view_angle"]
    scene.camera.view_up = _CAMERA_PRESET["view_up"]
    scene.camera.clipping_range = _CAMERA_PRESET["clipping_range"]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.elevation(-5)
    scene.camera.orthogonalize_view_up()
    scene.render()


def save_occ(save_dir, occupancy, name, sem=False, cap=2, dataset="nusc", show=False, empty_label=None):
    """Render an occupancy volume to a PNG using Mayavi."""

    dataset_config = _get_dataset_config(dataset)
    voxels = _prepare_voxels(occupancy, sem=sem, cap=cap)
    grid_coords = _build_grid_with_values(voxels, dataset_config)
    visible_voxels = _filter_visible_voxels(
        grid_coords,
        sem=sem,
        dataset_config=dataset_config,
        empty_label=empty_label,
    )

    mlab = _get_mlab(show=show)
    previous_offscreen = mlab.options.offscreen
    mlab.options.offscreen = _should_render_offscreen(show)
    figure = _create_figure(mlab, show=show)
    points_kwargs = _build_points_kwargs(dataset_config, sem=sem)

    plot = mlab.points3d(
        visible_voxels[:, 0],
        -visible_voxels[:, 1],
        visible_voxels[:, 2],
        visible_voxels[:, 3],
        **points_kwargs,
    )
    plot.glyph.scale_mode = "scale_by_vector"

    if sem:
        _apply_semantic_lut(plot, dataset_config)

    scene = figure.scene
    _apply_camera_path(scene)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{name}.png")
    mlab.draw(figure=figure)
    mlab.savefig(output_path)
    if show and not _OFFSCREEN:
        mlab.show()
    mlab.close(figure)
    mlab.options.offscreen = previous_offscreen


__all__ = [
    "get_grid_coords",
    "save_occ",
]