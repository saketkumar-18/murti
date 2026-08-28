"""Tests for metrics and the image-to-3D visual hull path."""
import numpy as np

from murti.metrics import (binarize, chamfer_distance, iou, surface_points,
                           volume_stats)
from murti.pipeline import image_to_volume, silhouette_to_volume


def _ball(grid=32, r=0.5, center=(0, 0, 0)):
    ax = np.linspace(-1, 1, grid)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    return (d < r).astype(np.float32)


def test_iou_identical_and_disjoint():
    a = _ball()
    assert iou(a, a) == 1.0
    b = _ball(center=(0.9, 0.9, 0.9), r=0.2)
    assert iou(a, b) < 0.05


def test_iou_partial_overlap():
    a = _ball(r=0.5)
    b = _ball(r=0.5, center=(0.3, 0, 0))
    v = iou(a, b)
    assert 0.1 < v < 0.9


def test_chamfer_zero_for_same_cloud():
    pts = surface_points(_ball())
    assert chamfer_distance(pts, pts) < 1e-9


def test_chamfer_positive_for_different():
    a = surface_points(_ball(r=0.4))
    b = surface_points(_ball(r=0.7))
    assert chamfer_distance(a, b) > 0.005


def test_surface_points_on_boundary():
    vol = _ball()
    pts = surface_points(vol, n_points=512)
    assert pts.shape[1] == 3
    radii = np.linalg.norm(pts, axis=1)
    assert np.abs(radii - 0.5).max() < 0.12  # near the r=0.5 shell


def test_volume_stats():
    s = volume_stats(_ball(r=0.5))
    # sphere r=0.5 inside [-1,1]^3 (vol 8): (4/3)pi*0.125/8 ~ 0.0654
    assert 0.05 < s["occupancy"] < 0.08
    assert not s["empty"]


def test_silhouette_visual_hull_ball():
    """A circular silhouette should carve to a roughly ball-like hull."""
    grid = 32
    ax = np.linspace(-1, 1, grid)
    x, y = np.meshgrid(ax, ax, indexing="ij")
    sil = (np.sqrt(x**2 + y**2) < 0.5).astype(np.float32)
    vol = silhouette_to_volume(sil, grid=grid, carve_rotations=True)
    assert vol.shape == (grid, grid, grid)
    b = vol > 0.5
    assert b.sum() > 100
    # hull of a circle from all 4 rotations ~ sphere: check corner voxels empty
    assert not b[0, 0, 0] and not b[-1, -1, -1]
    # center occupied
    assert b[grid // 2, grid // 2, grid // 2]


def test_image_to_volume_from_array():
    grid = 32
    img = np.full((64, 64), 255, np.uint8)
    # dark square object on light background
    img[16:48, 16:48] = 0
    vol = image_to_volume(img, grid=grid)
    assert vol.shape == (grid, grid, grid)
    b = vol > 0.5
    assert b[grid // 2, grid // 2, grid // 2]
    # square hull: corners of the object region occupied, far corners empty
    assert not b[1, 1, 1]
