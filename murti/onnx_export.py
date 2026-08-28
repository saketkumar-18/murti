"""ONNX export of the full in-browser inference graph.

Exports three models into web/models/:
  encoder.onnx    voxel -> latent mean            (image-to-3D refine path)
  decoder.onnx    latent -> occupancy volume      (shared)
  diffusion.onnx  (z_noisy, t, tokens) -> noise   (25-step DDIM loop in JS)

The JS runtime (web/js/inference.js) runs DDIM with CFG by calling
diffusion.onnx twice per step (cond + uncond), exactly mirroring
LatentDiffusion.sample.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .models import LatentDiffusion, VAE3D
from .text import MAX_LEN


class EncoderOnnx(nn.Module):
    def __init__(self, vae: VAE3D):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        mu, _ = self.vae.encode(x)
        return mu


class DecoderOnnx(nn.Module):
    def __init__(self, vae: VAE3D):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return torch.sigmoid(self.vae.decode(z))


class DiffusionOnnx(nn.Module):
    def __init__(self, diffusion: LatentDiffusion):
        super().__init__()
        self.unet = diffusion.unet

    def forward(self, z_noisy, t, tokens):
        return self.unet(z_noisy, t, tokens)


def export_onnx(vae: VAE3D, diffusion: LatentDiffusion, out_dir: str,
                opset: int = 17) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    vae.eval()
    diffusion.eval()
    paths = {}

    dummy_x = torch.zeros(1, 1, 32, 32, 32)
    dummy_z = torch.zeros(1, vae.latent_ch, 8, 8, 8)
    dummy_t = torch.zeros(1, dtype=torch.long)
    dummy_tok = torch.zeros(1, MAX_LEN, dtype=torch.long)

    # dynamo=False: the legacy TorchScript exporter — the new dynamo-based
    # exporter in torch>=2.6 fails to lower SiLU (aten.mul.Scalar) for ONNX.
    torch.onnx.export(EncoderOnnx(vae), dummy_x, os.path.join(out_dir, "encoder.onnx"),
                      input_names=["voxels"], output_names=["latent"],
                      opset_version=opset, dynamo=False,
                      dynamic_axes={"voxels": {0: "batch"}, "latent": {0: "batch"}})
    torch.onnx.export(DecoderOnnx(vae), dummy_z, os.path.join(out_dir, "decoder.onnx"),
                      input_names=["latent"], output_names=["volume"],
                      opset_version=opset, dynamo=False,
                      dynamic_axes={"latent": {0: "batch"}, "volume": {0: "batch"}})
    torch.onnx.export(DiffusionOnnx(diffusion), (dummy_z, dummy_t, dummy_tok),
                      os.path.join(out_dir, "diffusion.onnx"),
                      input_names=["z_noisy", "t", "tokens"], output_names=["noise"],
                      opset_version=opset, dynamo=False,
                      dynamic_axes={"z_noisy": {0: "batch"}, "t": {0: "batch"},
                                    "tokens": {0: "batch"}, "noise": {0: "batch"}})
    for name in ("encoder", "decoder", "diffusion"):
        paths[name] = os.path.join(out_dir, f"{name}.onnx")

    # sanity check with onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(paths["diffusion"], providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "z_noisy": np.zeros((1, vae.latent_ch, 8, 8, 8), np.float32),
        "t": np.zeros((1,), np.int64),
        "tokens": np.zeros((1, MAX_LEN), np.int64),
    })[0]
    assert out.shape == (1, vae.latent_ch, 8, 8, 8), out.shape
    return paths
