import os
from importlib import import_module

import numpy as np

from colormaps import get_kitti360_colormap, get_kitti_colormap, get_nuscenes_colormap, get_xc_cn_colormap

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
    "xc-cn": {
        "voxel_size": [0.5, 0.5, 0.5],
        "vox_origin": [-50.0, -50.0, -5.0],
        "vmin": 0,
        "vmax": 12,
        "semantic_upper_bound": 13,
        "semantic_colormap": get_xc_cn_colormap(),
    },
    "xccn": {
        "voxel_size": [0.5, 0.5, 0.5],
        "vox_origin": [-50.0, -50.0, -5.0],
        "vmin": 0,
        "vmax": 12,
        "semantic_upper_bound": 13,
        "semantic_colormap": get_xc_cn_colormap(),
    },
}
_CAMERA_PRESET = {
    "position": [-95.0, 0.0, 36.0],
    "focal_point": [0.0, 0.0, 14.0],
    "view_angle": 35.0,
    "view_up": [0.0, 0.0, 1.0],
    "clipping_range": [60.0, 220.0],
}
_FIGURE_MIN_SIDE = 1400
_FIGURE_MAX_SIDE = 2200
_CAMERA_FRAME_MARGIN = 1.02


def _normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length camera vector")
    return vector / norm


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

    if show:
        os.environ.setdefault("ETS_TOOLKIT", "qt")
        os.environ.setdefault("QT_API", "pyqt5")
    else:
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

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing="ij")
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


def _get_figure_size(voxel_shape, dataset_config):
    extents = np.array(voxel_shape[:2], dtype=np.float32) * np.array(dataset_config["voxel_size"][:2], dtype=np.float32)
    horizontal_aspect = float(extents[0] / max(extents[1], 1e-6))

    if horizontal_aspect >= 1.0:
        height = _FIGURE_MIN_SIDE
        width = int(round(height * horizontal_aspect))
        if width > _FIGURE_MAX_SIDE:
            width = _FIGURE_MAX_SIDE
            height = int(round(width / horizontal_aspect))
    else:
        width = _FIGURE_MIN_SIDE
        height = int(round(width / horizontal_aspect))
        if height > _FIGURE_MAX_SIDE:
            height = _FIGURE_MAX_SIDE
            width = int(round(height * horizontal_aspect))

    return max(width, 1), max(height, 1)


def _create_figure(mlab, show, figure_size):
    figure = mlab.figure(size=figure_size, bgcolor=(1, 1, 1))
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


def _get_scene_bounds(voxel_shape, dataset_config):
    voxel_size = np.array(dataset_config["voxel_size"], dtype=np.float32)
    origin = np.array(dataset_config["vox_origin"], dtype=np.float32)
    dims = np.array(voxel_shape, dtype=np.float32)
    mins = origin
    maxs = origin + (dims * voxel_size)
    return mins, maxs


def _get_bounds_corners(mins, maxs):
    return np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )


def _to_plot_space(coords):
    plot_coords = np.array(coords, dtype=np.float32, copy=True)
    plot_coords[:, 1] *= -1.0
    return plot_coords


def _to_plot_space_vector(vector):
    plot_vector = np.array(vector, dtype=np.float32, copy=True)
    plot_vector[1] *= -1.0
    return plot_vector


def _apply_explicit_camera_preset(scene, voxel_shape, dataset_config, camera_preset):
    position = np.array(camera_preset["position"], dtype=np.float32)
    focal_point = np.array(camera_preset["focal_point"], dtype=np.float32)
    forward = _normalize_vector(focal_point - position)
    view_up = _normalize_vector(np.array(camera_preset["view_up"], dtype=np.float32))
    right = _normalize_vector(np.cross(forward, view_up))
    view_up = _normalize_vector(np.cross(right, forward))

    scene.camera.position = position.tolist()
    scene.camera.focal_point = focal_point.tolist()
    scene.camera.view_angle = float(camera_preset.get("view_angle", _CAMERA_PRESET["view_angle"]))
    scene.camera.view_up = view_up.tolist()
    if "clipping_range" in camera_preset:
        scene.camera.clipping_range = list(camera_preset["clipping_range"])
    scene.camera.compute_view_plane_normal()
    scene.camera.orthogonalize_view_up()
    scene.render()
    if "clipping_range" not in camera_preset:
        scene.renderer.reset_camera_clipping_range()
        scene.render()


def _apply_camera_path(scene, voxel_shape, dataset_config, figure_size, camera_preset=None):
    if camera_preset is not None:
        _apply_explicit_camera_preset(scene, voxel_shape, dataset_config, camera_preset)
        return

    mins, maxs = _get_scene_bounds(voxel_shape, dataset_config)
    corners = _to_plot_space(_get_bounds_corners(mins, maxs))
    center = (corners.min(axis=0) + corners.max(axis=0)) / 2.0
    corners = corners - center

    base_focal = np.array(_CAMERA_PRESET["focal_point"], dtype=np.float32)
    base_position = np.array(_CAMERA_PRESET["position"], dtype=np.float32)
    camera_offset = base_position - base_focal
    camera_offset = camera_offset / np.linalg.norm(camera_offset)
    forward = -camera_offset
    up = np.array(_CAMERA_PRESET["view_up"], dtype=np.float32)
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    aspect_ratio = figure_size[0] / figure_size[1]
    vertical_fov = np.deg2rad(_CAMERA_PRESET["view_angle"])
    horizontal_fov = 2.0 * np.arctan(np.tan(vertical_fov / 2.0) * aspect_ratio)

    projected_x = np.abs(corners @ right)
    projected_y = np.abs(corners @ up)
    projected_z = np.abs(corners @ forward)

    projected_half_width = float(projected_x.max())
    projected_half_height = float(projected_y.max())
    projected_half_depth = float(projected_z.max())

    height_distance = projected_half_height / np.tan(vertical_fov / 2.0)
    width_distance = projected_half_width / np.tan(horizontal_fov / 2.0)
    distance = (max(height_distance, width_distance) + projected_half_depth) * _CAMERA_FRAME_MARGIN

    position = center + (camera_offset * distance)

    scene.camera.position = position.tolist()
    scene.camera.focal_point = center.tolist()
    scene.camera.view_angle = _CAMERA_PRESET["view_angle"]
    scene.camera.view_up = up.tolist()
    scene.camera.clipping_range = [max(distance - projected_half_depth * 2.5, 1.0), distance + projected_half_depth * 2.5]
    scene.camera.compute_view_plane_normal()
    scene.camera.orthogonalize_view_up()
    scene.render()


def save_occ(
    save_dir,
    occupancy,
    name,
    sem=False,
    cap=2,
    dataset="nusc",
    show=False,
    empty_label=None,
    camera_preset=None,
    figure_size=None,
):
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
    figure_size = figure_size if figure_size is not None else _get_figure_size(voxels.shape, dataset_config)
    figure = _create_figure(mlab, show=show, figure_size=figure_size)
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
    _apply_camera_path(scene, voxels.shape, dataset_config, figure_size, camera_preset=camera_preset)

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