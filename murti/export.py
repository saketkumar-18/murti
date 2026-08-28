"""Marching cubes meshing + OBJ / STL / GLB exporters.

Meshes are produced in normalized [-1, 1]^3 space and rescaled to a
configurable world size (default 1.0 unit) on export. GLB output is a
hand-rolled minimal glTF 2.0 binary writer (no external deps) so the
assets drop straight into Three.js / Unity / Blender / Godot.
"""
from __future__ import annotations

import base64
import io
import json
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray     # (F, 3) int64
    normals: Optional[np.ndarray] = None  # (V, 3) float32

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    zero = (norms[:, 0] == 0)
    norms[norms == 0] = 1.0
    normals = normals / norms
    if zero.any():
        # degenerate vertices: fall back to radial direction, then +y
        centroid = vertices.mean(axis=0)
        radial = vertices[zero] - centroid
        rn = np.linalg.norm(radial, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        fallback = radial / rn
        fallback[np.linalg.norm(fallback, axis=1) == 0] = np.array([0.0, 1.0, 0.0])
        normals[zero] = fallback
    return normals.astype(np.float32)


def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Orient a closed manifold mesh so normals point outward.

    Two-phase, robust for any topology (balls, tori, arches):
      1. BFS over face adjacency enforcing consistent winding (adjacent
         faces must traverse their shared edge in opposite directions).
      2. Signed-volume test: outward winding encloses positive volume;
         flip everything if the enclosed volume is negative.
    """
    from collections import defaultdict, deque

    n = len(faces)
    edge_faces = defaultdict(list)
    cur = [list(f) for f in faces]
    for fi, f in enumerate(cur):
        for i in range(3):
            e = (min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
            edge_faces[e].append(fi)

    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        while q:
            fi = q.popleft()
            f = cur[fi]
            for i in range(3):
                a, b = f[i], f[(i + 1) % 3]
                e = (min(a, b), max(a, b))
                for fj in edge_faces[e]:
                    if fj == fi or visited[fj]:
                        continue
                    visited[fj] = True
                    g = cur[fj]
                    # does fj traverse the shared edge in the opposite direction?
                    opposite = any(
                        g[j] == b and g[(j + 1) % 3] == a for j in range(3)
                    )
                    if not opposite:
                        cur[fj] = [g[0], g[2], g[1]]
                    q.append(fj)

    out = np.asarray(cur, dtype=faces.dtype)
    v0, v1, v2 = vertices[out[:, 0]], vertices[out[:, 1]], vertices[out[:, 2]]
    signed_vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
    if signed_vol < 0:
        out = out[:, ::-1].copy()
    return out


def marching_cubes(volume: np.ndarray, level: float = 0.5,
                   world_size: float = 1.0) -> Mesh:
    """Extract an iso-surface mesh from an occupancy volume in [0,1].

    Faces are oriented so vertex normals point outward (consistent-winding
    BFS + signed-volume test), which makes the GLB/OBJ exports render
    correctly in any engine regardless of the iso-surface topology.
    """
    from skimage.measure import marching_cubes as _mc
    verts, faces, _, _ = _mc(volume.astype(np.float32), level=level)
    g = volume.shape[0]
    verts = (verts / (g - 1)) * 2.0 - 1.0          # -> [-1, 1]
    verts = verts * (world_size / 2.0)              # -> world units
    verts = verts.astype(np.float32)
    faces = orient_faces_outward(verts, faces.astype(np.int64))
    normals = compute_vertex_normals(verts, faces)
    return Mesh(verts, faces, normals)


def mesh_stats(mesh: Mesh) -> dict:
    v0, v1, v2 = mesh.vertices[mesh.faces[:, 0]], mesh.vertices[mesh.faces[:, 1]], mesh.vertices[mesh.faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return {
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "surface_area": float(areas.sum()),
        "bounds_min": mesh.vertices.min(axis=0).tolist(),
        "bounds_max": mesh.vertices.max(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def export_obj(mesh: Mesh, path: str, name: str = "murti") -> str:
    lines = [f"# Murti generated mesh: {mesh.n_vertices} verts, {mesh.n_faces} faces",
             f"o {name}"]
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    if mesh.normals is not None:
        for n in mesh.normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for f in mesh.faces:
        if mesh.normals is not None:
            lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")
        else:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def export_stl(mesh: Mesh, path: str) -> str:
    """Binary STL (3D-print ready)."""
    v0, v1, v2 = mesh.vertices[mesh.faces[:, 0]], mesh.vertices[mesh.faces[:, 1]], mesh.vertices[mesh.faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    fn = fn / norms
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", mesh.n_faces))
        for i in range(mesh.n_faces):
            fh.write(struct.pack("<3f", *fn[i]))
            fh.write(struct.pack("<3f", *v0[i]))
            fh.write(struct.pack("<3f", *v1[i]))
            fh.write(struct.pack("<3f", *v2[i]))
            fh.write(struct.pack("<H", 0))
    return path


def _pad4(b: bytes) -> bytes:
    return b + b" " * ((4 - len(b) % 4) % 4)


def export_glb(mesh: Mesh, path: str, name: str = "murti") -> str:
    """Minimal glTF 2.0 binary (positions + normals + indices, one mesh)."""
    pos = mesh.vertices.astype(np.float32)
    nrm = (mesh.normals if mesh.normals is not None
           else compute_vertex_normals(mesh.vertices, mesh.faces)).astype(np.float32)
    idx = mesh.faces.astype(np.uint32).reshape(-1)

    pos_b = pos.tobytes()
    nrm_b = nrm.tobytes()
    idx_b = idx.tobytes()
    bin_blob = _pad4(pos_b) + _pad4(nrm_b) + _pad4(idx_b)

    pos_off, nrm_off = 0, len(_pad4(pos_b))
    idx_off = nrm_off + len(_pad4(nrm_b))
    vmin = pos.min(axis=0).tolist()
    vmax = pos.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0", "generator": "Murti 1.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                    "indices": 2, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": mesh.n_vertices,
             "type": "VEC3", "min": vmin, "max": vmax},
            {"bufferView": 1, "componentType": 5126, "count": mesh.n_vertices, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": int(idx.size), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": nrm_off, "byteLength": len(nrm_b), "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_b), "target": 34963},
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    json_b = _pad4(json.dumps(gltf, separators=(",", ":")).encode())

    with open(path, "wb") as fh:
        fh.write(b"glTF")
        fh.write(struct.pack("<II", 2, 12 + 8 + len(json_b) + 8 + len(bin_blob)))
        fh.write(struct.pack("<I", len(json_b)))
        fh.write(b"JSON")
        fh.write(json_b)
        fh.write(struct.pack("<I", len(bin_blob)))
        fh.write(b"BIN\0")
        fh.write(bin_blob)
    return path


def export_mesh(mesh: Mesh, path: str, fmt: Optional[str] = None,
                name: str = "murti") -> str:
    fmt = fmt or path.rsplit(".", 1)[-1].lower()
    if fmt == "obj":
        return export_obj(mesh, path, name)
    if fmt == "stl":
        return export_stl(mesh, path)
    if fmt in ("glb", "gltf"):
        return export_glb(mesh, path, name)
    raise ValueError(f"unsupported format: {fmt}")
