"""End-to-end pipeline tests with tiny untrained models (structure only)."""
import numpy as np
import torch

from murti.models import LatentDiffusion, VAE3D
from murti.pipeline import Pipeline, image_to_volume


def test_pipeline_generate_structure():
    pipe = Pipeline(VAE3D(), LatentDiffusion())
    res = pipe.generate("a sphere", seed=0, cfg_scale=2.0, ddim_steps=2)
    assert res.prompt == "a sphere"
    assert res.volume.shape == (32, 32, 32)
    assert res.mesh.n_vertices > 0
    assert res.mesh.n_faces > 0
    assert res.stats["faces"] == res.mesh.n_faces
    assert res.seed == 0 and res.cfg_scale == 2.0


def test_pipeline_generate_reproducible():
    pipe = Pipeline(VAE3D(), LatentDiffusion())
    r1 = pipe.generate("a cube", seed=7, ddim_steps=2)
    r2 = pipe.generate("a cube", seed=7, ddim_steps=2)
    assert np.allclose(r1.volume, r2.volume)


def test_pipeline_reconstruct():
    pipe = Pipeline(VAE3D(), LatentDiffusion())
    ax = np.linspace(-1, 1, 32)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    vol = (np.sqrt(x**2 + y**2 + z**2) < 0.5).astype(np.float32)
    mesh = pipe.reconstruct(vol)
    assert mesh.n_faces > 0


def test_image_to_3D_full_path():
    img = np.full((64, 64), 255, np.uint8)
    img[20:44, 20:44] = 0
    vol = image_to_volume(img, grid=32)
    pipe = Pipeline(VAE3D(), LatentDiffusion())
    mesh = pipe.reconstruct(vol)
    assert mesh.n_faces > 0
