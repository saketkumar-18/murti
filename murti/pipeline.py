"""End-to-end generation pipeline: prompt -> latent diffusion -> mesh.

Also hosts the image-to-3D path: a silhouette is carved into a voxel
visual hull (space carving from the silhouette + its 90-degree rotations),
optionally refined by conditioning the diffusion model on the classified
shape name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .export import Mesh, marching_cubes, mesh_stats
from .models import LatentDiffusion, ShapeClassifier, VAE3D
from .shapes import GRID
from .text import tokenize


@dataclass
class GenerationResult:
    prompt: str
    volume: np.ndarray      # (g,g,g) float32 occupancy
    mesh: Mesh
    stats: dict
    seed: int
    cfg_scale: float
    steps: int


class Pipeline:
    """Wraps VAE + diffusion (+ optional classifier) for inference."""

    def __init__(self, vae: VAE3D, diffusion: LatentDiffusion,
                 classifier: Optional[ShapeClassifier] = None,
                 device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.vae = vae.to(self.device).eval()
        self.diffusion = diffusion.to(self.device).eval()
        self.classifier = classifier.to(self.device).eval() if classifier else None

    @classmethod
    def load(cls, ckpt_dir: str, device: Optional[torch.device] = None):
        import os
        vae = VAE3D()
        diff = LatentDiffusion()
        clf = ShapeClassifier(len(_class_names()))
        vae.load_state_dict(torch.load(os.path.join(ckpt_dir, "vae.pt"), map_location="cpu"))
        diff.load_state_dict(torch.load(os.path.join(ckpt_dir, "diffusion.pt"), map_location="cpu"))
        clf_path = os.path.join(ckpt_dir, "classifier.pt")
        if os.path.exists(clf_path):
            clf.load_state_dict(torch.load(clf_path, map_location="cpu"))
        else:
            clf = None
        return cls(vae, diff, clf, device)

    @torch.no_grad()
    def generate(self, prompt: str, seed: int = 0, cfg_scale: float = 3.0,
                 ddim_steps: int = 25, level: float = 0.5,
                 world_size: float = 1.0) -> GenerationResult:
        tokens = torch.tensor([tokenize(prompt)], dtype=torch.long, device=self.device)
        z = self.diffusion.sample(tokens, (self.vae.latent_ch, 8, 8, 8),
                                  ddim_steps=ddim_steps, cfg_scale=cfg_scale, seed=seed)
        vol = torch.sigmoid(self.vae.decode(z)).squeeze().cpu().numpy()
        mesh = marching_cubes(vol, level=level, world_size=world_size)
        return GenerationResult(prompt, vol.astype(np.float32), mesh,
                                mesh_stats(mesh), seed, cfg_scale, ddim_steps)

    @torch.no_grad()
    def reconstruct(self, volume: np.ndarray, level: float = 0.5,
                    world_size: float = 1.0) -> Mesh:
        x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        vol = self.vae.reconstruct(x).squeeze().cpu().numpy()
        return marching_cubes(vol, level=level, world_size=world_size)


def _class_names():
    from .shapes import CLASS_NAMES
    return CLASS_NAMES


# ---------------------------------------------------------------------------
# Image-to-3D: silhouette visual hull
# ---------------------------------------------------------------------------

def silhouette_to_volume(silhouette: np.ndarray, grid: int = GRID,
                         carve_rotations: bool = True) -> np.ndarray:
    """Extrude a binary silhouette along z, then carve with rotated copies.

    silhouette: (H, W) array, nonzero = foreground. Returns (grid,grid,grid)
    float32 occupancy in [0,1]. With carve_rotations the hull is intersected
    with 90-degree rotated extrusions, which rounds off the sides and gives
    a much better hull for roughly symmetric objects.
    """
    from PIL import Image
    img = Image.fromarray((silhouette > 0).astype(np.uint8) * 255).resize((grid, grid))
    sil = (np.asarray(img) > 127).astype(np.float32)

    def extrude(s: np.ndarray) -> np.ndarray:
        # s is (grid, grid) in image coords (row=y down, col=x); volume axes (x, y, z)
        vol = np.repeat(s[None, :, :], grid, axis=0)          # extrude along x
        vol = np.transpose(vol, (1, 2, 0))                    # (y, z, x) -> want (x,y,z)
        vol = np.transpose(vol, (2, 0, 1))
        return vol

    hull = extrude(sil)
    if carve_rotations:
        for k in (1, 2, 3):
            rot = np.rot90(sil, k=k)
            hull = np.minimum(hull, extrude(rot))
    # smooth slightly for nicer marching cubes
    from scipy.ndimage import gaussian_filter
    hull = gaussian_filter(hull.astype(np.float32), 0.8)
    return hull


def image_to_volume(image, grid: int = GRID, threshold: int = 200,
                    carve_rotations: bool = True) -> np.ndarray:
    """Load an image (path or ndarray), threshold to silhouette, carve hull.

    Dark-on-light or light-on-dark both work: we pick the polarity whose
    foreground is smaller (the object).
    """
    from PIL import Image
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8))
    else:
        img = Image.open(image)
    img = img.convert("L").resize((grid, grid))
    arr = np.asarray(img)
    fg_dark = arr < threshold
    fg_light = arr >= threshold
    sil = fg_dark if fg_dark.sum() <= fg_light.sum() else fg_light
    return silhouette_to_volume(sil.astype(np.float32), grid, carve_rotations)
