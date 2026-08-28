# Model Card — Murti v1.0

## Model details

- **Developed by:** Saket Kumar (capstone project #19)
- **Model type:** Conditional latent diffusion over 3D voxel occupancy,
  with a 3D-convolutional VAE latent space. Text conditioning via a
  GRU encoder; classifier-free guidance at sampling time.
- **License:** MIT
- **Version:** 1.0.0
- **Contact:** k.saket@op.iitg.ac.in

## Architecture & parameters

| Component | Params | Role |
|---|---|---|
| 3D VAE (encoder + decoder) | ~0.55M | 32³ voxel grid ↔ 4×8³ latent |
| Latent diffusion UNet | ~2.2M | ε-prediction, text-conditioned |
| Text encoder (GRU, 2-layer) | ~0.06M | prompt → 64-d context |
| Shape classifier | ~0.1M | eval + image-to-3D support |

- Diffusion schedule: 1000 steps, linear β from 1e-4 to 0.02.
- Sampling: DDIM, 25 steps default (10–50 selectable), CFG scale 3.0 default.
- Vocab: 40 tokens (18 shape nouns, 10 adjectives, structure words).

## Training data

**Fully synthetic.** 3,000 procedurally generated scenes of 32³ occupancy
grids: 18 analytic shape primitives with randomized scale, aspect ratio and
small rotations, plus compositional scenes (A on top of B, A next to B,
pairs) and adjective modifiers. Every sample has a machine-generated
ground-truth caption from a fixed grammar. No external, scraped, or
copyrighted data is used at any stage.

## Training procedure

1. VAE: BCE reconstruction + β·KL (β=1e-4), AdamW 2e-4, 8 epochs.
2. Latent diffusion: MSE noise prediction on pre-encoded latents, 15%
   condition dropout for CFG, AdamW 2e-4, 16 epochs.
3. Classifier: cross-entropy on occupancy grids, 8 epochs.
4. Evaluation + ONNX export.

Hardware: 16-thread CPU (no GPU), ~2.5 hours total. Framework: PyTorch 2.x
(CPU build). Deterministic per-seed generation in both Python and the
browser runtime.

## Evaluation

Held-out metrics are written to `web/models/config.json` (`eval` block) at
the end of every training run:

- **Reconstruction IoU** (VAE round-trip on unseen samples)
- **Generation IoU vs same-class reference** (12 class prompts)
- **Chamfer distance** between generated and reference surface point clouds
- **Mesh integrity**: watertightness, Euler characteristic, outward-normal
  ratio (verified in both Python and the JS implementation)

## Intended use

- Research and education: a compact, inspectable, fully reproducible
  text-to-3D pipeline.
- Rapid prototyping of simple 3D primitives and compositions for games,
  web scenes, and 3D printing (STL export).

## Out of scope / limitations

- **Vocabulary-bound prompts.** The model only understands its 40-token
  vocabulary; unknown words map to `<unk>` and outputs degrade toward the
  unconditional prior. Free-form natural language is not supported.
- **Resolution.** 32³ voxels → meshes capture coarse shape structure, not
  fine detail. Not competitive with large pretrained systems
  (Point-E, Shap-E, TripoSR, InstantMesh) on realism.
- **Composition.** Stacked/side-by-side scenes are learned but remain the
  hardest category; parts may merge or blur.
- **Image-to-3D** uses geometric visual-hull carving (silhouette
  extrusion + rotation carving), not learned reconstruction: it works well
  for convex-ish objects on plain backgrounds and poorly for complex or
  cluttered photos.
- Outputs are meshes without color/texture/material information.

## Ethical considerations

See ETHICS.md. Key points: synthetic-only training data eliminates
copyright and privacy exposure; inference is fully client-side (no user
data collection); generated geometry is original (not copied from any
dataset); known failure modes are documented rather than hidden.

## Citation

```bibtex
@software{murti2026,
  title  = {Murti: Text and Image to 3D Asset Generation with Latent Diffusion},
  author = {Kumar, Saket},
  year   = {2026},
  url    = {https://github.com/saketkumar-18/murti}
}
```
