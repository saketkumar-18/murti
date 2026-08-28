"""Evaluation metrics for generated / reconstructed 3D volumes and meshes.

- IoU between binary volumes (reconstruction fidelity)
- Chamfer distance between point clouds (generation quality vs reference)
- Classifier accuracy on reconstructions (semantic preservation)
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def binarize(volume: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (volume > threshold).astype(bool)


def iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    a_b, b_b = binarize(a, threshold), binarize(b, threshold)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    return float(inter / union) if union > 0 else 1.0


def surface_points(volume: np.ndarray, n_points: int = 2048,
                   threshold: float = 0.5, seed: int = 0) -> np.ndarray:
    """Sample points on the iso-surface boundary voxels."""
    rng = np.random.default_rng(seed)
    b = binarize(volume, threshold)
    # boundary = occupied voxel with at least one empty 6-neighbor
    padded = np.pad(b, 1, mode="constant", constant_values=False)
    interior = (
        padded[2:, 1:-1, 1:-1] & padded[:-2, 1:-1, 1:-1] &
        padded[1:-1, 2:, 1:-1] & padded[1:-1, :-2, 1:-1] &
        padded[1:-1, 1:-1, 2:] & padded[1:-1, 1:-1, :-2]
    )
    boundary = b & ~interior
    idx = np.argwhere(boundary)
    if len(idx) == 0:
        idx = np.argwhere(b)
    if len(idx) == 0:
        return np.zeros((0, 3))
    if len(idx) > n_points:
        idx = idx[rng.choice(len(idx), n_points, replace=False)]
    g = volume.shape[0]
    return (idx / (g - 1)) * 2.0 - 1.0


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric chamfer distance between two (N,3) point clouds."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    # chunked to keep memory bounded
    def one_way(x, y):
        best = np.full(len(x), np.inf)
        for i in range(0, len(y), 512):
            chunk = y[i:i + 512]
            d = ((x[:, None, :] - chunk[None, :, :]) ** 2).sum(-1)
            best = np.minimum(best, d.min(axis=1))
        return float(best.mean())
    return 0.5 * (one_way(a, b) + one_way(b, a))


def volume_stats(volume: np.ndarray, threshold: float = 0.5) -> dict:
    b = binarize(volume, threshold)
    return {
        "occupancy": float(b.mean()),
        "voxels": int(b.sum()),
        "empty": bool(b.sum() == 0),
    }
