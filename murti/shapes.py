"""Procedural 3D shape engine.

Builds occupancy grids (32^3 by default) from analytic shape primitives
with randomized scale / aspect / rotation, plus compositional scenes
(stacking, side-by-side, pairs). Every builder also returns the tokens
needed to compose a ground-truth caption, so the dataset is perfectly
labelled for text-to-3D training.

Coordinates: the grid spans [-1, 1]^3, y is "up".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

GRID = 32
HALF = 0.82  # shapes stay inside this radius so meshes are watertight-ish


def _coords(grid: int = GRID) -> np.ndarray:
    ax = np.linspace(-1.0, 1.0, grid)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([x, y, z], axis=-1)


_COORD_CACHE: Dict[int, np.ndarray] = {}


def coords(grid: int = GRID) -> np.ndarray:
    if grid not in _COORD_CACHE:
        _COORD_CACHE[grid] = _coords(grid)
    return _COORD_CACHE[grid]


def rotation_matrix(rng: np.random.Generator, max_angle_deg: float = 12.0):
    """Small random rotation (keeps shapes recognizable)."""
    angle = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    c, s = np.cos(angle), np.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
        [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
        [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
    ])


def _rot(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    return p @ R.T


# ---------------------------------------------------------------------------
# Primitive occupancy builders: (p, rng, params) -> bool grid
# ---------------------------------------------------------------------------

def _sphere(p, rng, prm):
    r = prm.get("r", rng.uniform(0.45, 0.62))
    return np.linalg.norm(p, axis=-1) < r, {"r": r}


def _cube(p, rng, prm):
    b = prm.get("b", rng.uniform(0.42, 0.58))
    return np.max(np.abs(p), axis=-1) < b, {"b": b}


def _cylinder(p, rng, prm):
    r = prm.get("r", rng.uniform(0.38, 0.52))
    h = prm.get("h", rng.uniform(0.5, 0.7))
    d = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
    return (d < r) & (np.abs(p[..., 1]) < h), {"r": r, "h": h}


def _cone(p, rng, prm):
    r = prm.get("r", rng.uniform(0.45, 0.6))
    h = prm.get("h", rng.uniform(0.6, 0.78))
    t = (p[..., 1] + h) / (2 * h)  # 0 at bottom, 1 at apex
    rad = r * np.clip(1.0 - t, 0.0, 1.0)
    d = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
    return (d < rad) & (np.abs(p[..., 1]) < h), {"r": r, "h": h}


def _torus(p, rng, prm):
    R = prm.get("R", rng.uniform(0.42, 0.52))
    r = prm.get("r", rng.uniform(0.16, 0.24))
    q = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2) - R
    return np.sqrt(q ** 2 + p[..., 1] ** 2) < r, {"R": R, "r": r}


def _capsule(p, rng, prm):
    r = prm.get("r", rng.uniform(0.3, 0.4))
    h = prm.get("h", rng.uniform(0.3, 0.5))
    d = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
    dy = np.clip(np.abs(p[..., 1]) - h, 0.0, None)
    return np.sqrt(d ** 2 + dy ** 2) < r, {"r": r, "h": h}


def _pyramid(p, rng, prm):
    r = prm.get("r", rng.uniform(0.5, 0.65))
    h = prm.get("h", rng.uniform(0.6, 0.78))
    t = (p[..., 1] + h) / (2 * h)
    half = r * np.clip(1.0 - t, 0.0, 1.0)
    return (np.max(np.abs(p[..., [0, 2]]), axis=-1) < half) & (np.abs(p[..., 1]) < h), {"r": r, "h": h}


def _prism(p, rng, prm):
    """Triangular prism, extruded along y."""
    s = prm.get("s", rng.uniform(0.5, 0.68))
    h = prm.get("h", rng.uniform(0.5, 0.7))
    x, z = p[..., 0] / s, p[..., 2] / s
    tri = (z > -0.45) & (z < 0.9) & (z < -1.1 * np.abs(x) + 0.62)
    return tri & (np.abs(p[..., 1]) < h), {"s": s, "h": h}


def _star(p, rng, prm):
    """5-point star extruded along y (polar formula in the xz plane)."""
    R1 = prm.get("R1", rng.uniform(0.55, 0.68))
    R2 = prm.get("R2", R1 * rng.uniform(0.42, 0.55))
    h = prm.get("h", rng.uniform(0.22, 0.34))
    x, z = p[..., 0], p[..., 2]
    theta = np.arctan2(z, x)
    w = 0.5 + 0.5 * np.cos(5 * theta)
    rad = R2 + (R1 - R2) * w ** 3
    return (np.sqrt(x ** 2 + z ** 2) < rad) & (np.abs(p[..., 1]) < h), {"R1": R1, "R2": R2, "h": h}


def _arch(p, rng, prm):
    """Box with a half-cylinder doorway cut through it."""
    b = prm.get("b", rng.uniform(0.55, 0.68))
    d = prm.get("d", rng.uniform(0.24, 0.32))
    box = np.max(np.abs(p), axis=-1) < b
    hole = (np.sqrt(p[..., 0] ** 2 + (p[..., 1] + b) ** 2) < d) & (p[..., 1] < 0.1)
    return box & ~hole, {"b": b, "d": d}


def _cross(p, rng, prm):
    a = prm.get("a", rng.uniform(0.16, 0.24))
    L = prm.get("L", rng.uniform(0.55, 0.7))
    ax = np.abs(p)
    b1 = (ax[..., 0] < L) & (ax[..., 1] < a) & (ax[..., 2] < a)
    b2 = (ax[..., 1] < L) & (ax[..., 0] < a) & (ax[..., 2] < a)
    b3 = (ax[..., 2] < L) & (ax[..., 0] < a) & (ax[..., 1] < a)
    return b1 | b2 | b3, {"a": a, "L": L}


def _table(p, rng, prm):
    top = prm.get("top", rng.uniform(0.55, 0.68))
    th = prm.get("th", 0.09)
    leg = prm.get("leg", 0.09)
    h = prm.get("h", rng.uniform(0.35, 0.45))
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    slab = (np.abs(x) < top) & (np.abs(y - h) < th) & (np.abs(z) < top)
    legs = np.zeros(p.shape[:-1], bool)
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx, cz = sx * (top - 2 * leg), sz * (top - 2 * leg)
            legs |= (np.abs(x - cx) < leg) & (np.abs(z - cz) < leg) & (y > -h - th) & (y < h)
    return slab | legs, {"top": top, "h": h}


def _chair(p, rng, prm):
    s = prm.get("s", rng.uniform(0.42, 0.52))
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    seat_y = prm.get("seat_y", -0.1)
    seat = (np.abs(x) < s) & (np.abs(y - seat_y) < 0.08) & (np.abs(z) < s)
    back = (np.abs(x) < s) & (y > seat_y) & (y < seat_y + 2 * s) & (np.abs(z + s + 0.06) < 0.08)
    legs = np.zeros(p.shape[:-1], bool)
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx, cz = sx * (s - 0.08), sz * (s - 0.08)
            legs |= (np.abs(x - cx) < 0.07) & (np.abs(z - cz) < 0.07) & (y < seat_y) & (y > seat_y - 0.55)
    return seat | back | legs, {"s": s}


def _snowman(p, rng, prm):
    r1 = prm.get("r1", rng.uniform(0.4, 0.48))
    r2 = r1 * 0.72
    r3 = r1 * 0.5
    d = np.linalg.norm(p, axis=-1)
    y = p[..., 1]
    b1 = d < r1
    b2 = np.sqrt(p[..., 0] ** 2 + (y - r1 * 1.15) ** 2 + p[..., 2] ** 2) < r2
    b3 = np.sqrt(p[..., 0] ** 2 + (y - r1 * 1.95) ** 2 + p[..., 2] ** 2) < r3
    return (b1 | b2 | b3) & (y > -r1), {"r1": r1}


def _rocket(p, rng, prm):
    r = prm.get("r", rng.uniform(0.26, 0.34))
    h = prm.get("h", rng.uniform(0.45, 0.6))
    d = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
    y = p[..., 1]
    body = (d < r) & (np.abs(y + 0.15) < h)
    nose_t = (y - (h - 0.15)) / (2.2 * r)
    nose = (d < r * np.clip(1 - nose_t, 0, 1)) & (y > h - 0.15) & (y < h - 0.15 + 2.2 * r)
    fins = np.zeros(p.shape[:-1], bool)
    for ang in (0.0, 2.094, 4.188):
        fx, fz = np.cos(ang), np.sin(ang)
        along = p[..., 0] * fx + p[..., 2] * fz
        perp = np.abs(-p[..., 0] * fz + p[..., 2] * fx)
        fins |= (along > r * 0.5) & (along < r * 2.1) & (perp < 0.05) & (y < -0.15 - h * 0.35) & (y > -0.15 - h)
    return body | nose | fins, {"r": r, "h": h}


def _tree(p, rng, prm):
    tr = prm.get("tr", rng.uniform(0.1, 0.15))
    th = prm.get("th", rng.uniform(0.35, 0.5))
    cr = prm.get("cr", rng.uniform(0.4, 0.55))
    d = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
    y = p[..., 1]
    trunk = (d < tr) & (y > -0.8) & (y < -0.8 + 2 * th + 0.5)
    crown = np.sqrt(p[..., 0] ** 2 + (y - (-0.8 + 2 * th + 0.5 + cr * 0.55)) ** 2 + p[..., 2] ** 2) < cr
    return (trunk | crown) & (y > -0.8), {"tr": tr, "cr": cr}


def _dumbbell(p, rng, prm):
    r = prm.get("r", rng.uniform(0.26, 0.34))
    L = prm.get("L", rng.uniform(0.45, 0.6))
    bar = prm.get("bar", 0.09)
    d = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
    y = p[..., 1]
    bar_m = (d < bar) & (np.abs(y) < L)
    s1 = np.sqrt(d ** 2 + (y - L) ** 2) < r
    s2 = np.sqrt(d ** 2 + (y + L) ** 2) < r
    return bar_m | s1 | s2, {"r": r, "L": L}


def _block(p, rng, prm):
    b = np.array([rng.uniform(0.25, 0.65) for _ in range(3)])
    inside = np.all(np.abs(p) < b[None, None, None, :], axis=-1)
    return inside, {"b": b.tolist()}


BUILDERS: Dict[str, Callable] = {
    "sphere": _sphere, "cube": _cube, "cylinder": _cylinder, "cone": _cone,
    "torus": _torus, "capsule": _capsule, "pyramid": _pyramid, "prism": _prism,
    "star": _star, "arch": _arch, "cross": _cross, "table": _table,
    "chair": _chair, "snowman": _snowman, "rocket": _rocket, "tree": _tree,
    "dumbbell": _dumbbell, "block": _block,
}
SHAPE_NAMES = list(BUILDERS.keys())

# Adjective -> per-axis scale multipliers (x, y, z)
ADJECTIVES = {
    "tall": (0.85, 1.35, 0.85), "wide": (1.3, 0.85, 1.3), "flat": (1.15, 0.55, 1.15),
    "small": (0.72, 0.72, 0.72), "large": (1.22, 1.22, 1.22), "thin": (0.62, 1.1, 0.62),
    "chunky": (1.25, 0.8, 1.25), "slender": (0.7, 1.25, 0.7),
    "squat": (1.2, 0.62, 1.2), "stretched": (0.8, 1.45, 0.8),
}


def _eval_at(builder, p: np.ndarray, rng: np.random.Generator,
             offset=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), R=None) -> np.ndarray:
    """Evaluate a builder on transformed coordinates (analytic, no resampling).

    q = R @ ((p - offset) / scale); scale>1 stretches the shape along an axis.
    """
    q = p - np.asarray(offset, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)
    if not np.allclose(s, 1.0):
        q = q / s
    if R is not None:
        q = q @ R.T
    occ, _ = builder(q, rng, {})
    return occ.astype(np.float32)


@dataclass
class ShapeSample:
    occupancy: np.ndarray          # float32 (g, g, g) in [0, 1]
    caption: str
    label: int                     # class id (see CLASS_NAMES)
    kind: str                      # single | stack | side | pair
    shape: str                     # primary shape name


CLASS_NAMES = SHAPE_NAMES + ["compound"]
LABEL = {n: i for i, n in enumerate(CLASS_NAMES)}


def _smooth(occ: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(occ.astype(np.float32), sigma)


def build_shape(name: str, rng: np.random.Generator, grid: int = GRID,
                adj: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """One randomized shape, optionally modified by an adjective."""
    p = coords(grid)
    R = rotation_matrix(rng)
    scale = ADJECTIVES[adj] if adj is not None else (1.0, 1.0, 1.0)
    occ = _eval_at(BUILDERS[name], p, rng, scale=scale, R=R)
    occ = _smooth(occ)
    cap = f"a {adj} {name}" if adj else f"a {name}"
    return occ, cap


def build_scene(rng: np.random.Generator, grid: int = GRID) -> ShapeSample:
    """Random sample: single shape (60%) or composition (40%)."""
    roll = rng.random()
    use_adj = rng.random() < 0.45
    adj = rng.choice(list(ADJECTIVES.keys())) if use_adj else None
    p = coords(grid)

    if roll < 0.60:  # single shape
        name = rng.choice(SHAPE_NAMES)
        occ, cap = build_shape(name, rng, grid, adj)
        return ShapeSample(occ, cap, LABEL[name], "single", name)

    if roll < 0.78:  # stack: A on top of B
        a, b = rng.choice(SHAPE_NAMES, size=2, replace=False)
        Ra, Rb = rotation_matrix(rng), rotation_matrix(rng)
        occ_b = _eval_at(BUILDERS[b], p, rng, offset=(0, -0.34, 0),
                         scale=(0.72, 0.62, 0.72), R=Rb)
        occ_a = _eval_at(BUILDERS[a], p, rng, offset=(0, 0.42, 0),
                         scale=(0.60, 0.60, 0.60), R=Ra)
        occ = _smooth(np.maximum(occ_a, occ_b), 0.5)
        cap = f"a {a} on top of a {b}"
        return ShapeSample(occ, cap, LABEL["compound"], "stack", a)

    if roll < 0.90:  # side by side: A next to B
        a, b = rng.choice(SHAPE_NAMES, size=2, replace=False)
        Ra, Rb = rotation_matrix(rng), rotation_matrix(rng)
        occ_a = _eval_at(BUILDERS[a], p, rng, offset=(-0.44, -0.12, 0),
                         scale=(0.55, 0.55, 0.55), R=Ra)
        occ_b = _eval_at(BUILDERS[b], p, rng, offset=(0.44, -0.12, 0),
                         scale=(0.55, 0.55, 0.55), R=Rb)
        occ = _smooth(np.maximum(occ_a, occ_b), 0.5)
        cap = f"a {a} next to a {b}"
        return ShapeSample(occ, cap, LABEL["compound"], "side", a)

    # pair: two of the same
    a = rng.choice(SHAPE_NAMES)
    R1, R2 = rotation_matrix(rng), rotation_matrix(rng)
    o1 = _eval_at(BUILDERS[a], p, rng, offset=(-0.44, -0.12, 0),
                  scale=(0.55, 0.55, 0.55), R=R1)
    o2 = _eval_at(BUILDERS[a], p, rng, offset=(0.44, -0.12, 0),
                  scale=(0.55, 0.55, 0.55), R=R2)
    occ = _smooth(np.maximum(o1, o2), 0.5)
    cap = f"two {a}s"
    return ShapeSample(occ, cap, LABEL["compound"], "pair", a)
