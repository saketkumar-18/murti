"""Murti — text & image to 3D asset generation with latent diffusion.

Pipeline: procedural 3D dataset -> 3D-conv VAE (voxel -> latent) ->
classifier-free-guided latent diffusion conditioned on text ->
marching cubes -> OBJ / STL / GLB export. The full inference pipeline is
also exported to ONNX and runs entirely in the browser (web/).
"""

__version__ = "1.0.0"
