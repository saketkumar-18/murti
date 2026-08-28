"""Tests for meshing and the OBJ / STL / GLB exporters."""
import os
import struct

import numpy as np
import pytest

from murti.export import (Mesh, compute_vertex_normals, export_glb,
                          export_mesh, export_obj, export_stl,
                          marching_cubes, mesh_stats)


def _ball_volume(grid=32, r=0.5):
    ax = np.linspace(-1, 1, grid)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    return (np.sqrt(x**2 + y**2 + z**2) < r).astype(np.float32)


def test_marching_cubes_ball():
    mesh = marching_cubes(_ball_volume(), level=0.5, world_size=1.0)
    assert mesh.n_vertices > 100
    assert mesh.n_faces > 100
    assert mesh.faces.min() >= 0 and mesh.faces.max() < mesh.n_vertices
    # ball of radius 0.5 in [-1,1] space -> bounds within [-0.55, 0.55]
    assert np.abs(mesh.vertices).max() < 0.6
    stats = mesh_stats(mesh)
    # world_size=1.0 maps [-1,1] -> 1 unit, so r=0.5 -> r=0.25 world units
    # surface area = 4*pi*0.25^2 ~ 0.785
    assert 0.6 < stats["surface_area"] < 1.0


def test_vertex_normals_unit_length():
    mesh = marching_cubes(_ball_volume())
    norms = np.linalg.norm(mesh.normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    # outward-facing: normals on a ball align with position direction
    dots = (mesh.normals * mesh.vertices).sum(1)
    assert (dots > 0).mean() > 0.95


def test_export_obj(tmp_path):
    mesh = marching_cubes(_ball_volume())
    p = export_obj(mesh, str(tmp_path / "ball.obj"), name="ball")
    text = open(p).read()
    assert text.startswith("# Murti")
    assert text.count("\nv ") == mesh.n_vertices
    assert text.count("\nf ") == mesh.n_faces
    assert "o ball" in text


def test_export_stl_binary(tmp_path):
    mesh = marching_cubes(_ball_volume())
    p = export_stl(mesh, str(tmp_path / "ball.stl"))
    data = open(p, "rb").read()
    n_faces = struct.unpack("<I", data[80:84])[0]
    assert n_faces == mesh.n_faces
    assert len(data) == 84 + n_faces * 50


def test_export_glb_structure(tmp_path):
    mesh = marching_cubes(_ball_volume())
    p = export_glb(mesh, str(tmp_path / "ball.glb"))
    data = open(p, "rb").read()
    assert data[:4] == b"glTF"
    version, total_len = struct.unpack("<II", data[4:12])
    assert version == 2
    assert total_len == len(data)
    json_len = struct.unpack("<I", data[12:16])[0]
    assert data[16:20] == b"JSON"
    import json
    gltf = json.loads(data[20:20 + json_len])
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"] == 0
    assert gltf["meshes"][0]["primitives"][0]["attributes"]["NORMAL"] == 1
    # buffer lengths consistent
    assert gltf["buffers"][0]["byteLength"] <= len(data)


def test_export_mesh_dispatch(tmp_path):
    mesh = marching_cubes(_ball_volume())
    for ext in ("obj", "stl", "glb"):
        p = export_mesh(mesh, str(tmp_path / f"m.{ext}"))
        assert os.path.exists(p) and os.path.getsize(p) > 0
    with pytest.raises(ValueError):
        export_mesh(mesh, str(tmp_path / "m.xyz"))


def test_empty_volume_raises():
    with pytest.raises(Exception):
        marching_cubes(np.zeros((32, 32, 32), np.float32))
