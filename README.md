# Murti 🗿 — Text & Image to 3D Asset Generator

**Live: https://murti-six.vercel.app** · **Capstone #19** — generate
game/web-ready 3D models from text prompts and images using a real,
from-scratch-trained **latent diffusion** pipeline. The entire inference stack
is exported to **ONNX** and runs **100% in the browser** via WebAssembly — no
GPU server, no uploads.

> *Murti* (मूर্তि) — Sanskrit for "form, figure, embodiment".

## ✨ What it does

| Mode | Input | Output |
|---|---|---|
| **Text → 3D** | a prompt like `a rocket`, `a tall cylinder`, `a sphere on top of a cube` | textured-ready mesh |
| **Image → 3D** | a silhouette photo | visual-hull mesh carved in-browser |
| **Export** | any generated mesh | **GLB** (games/web), **OBJ**, **STL** (3D printing) |

Generation is fully controllable: seed, CFG guidance scale, and DDIM step
count. Same seed + prompt ⇒ identical mesh, in Python *and* in the browser.

## 🧠 Architecture

```
prompt ──▶ tokenizer ──▶ GRU text encoder ─┐
                                           ▼
noise z_T ──▶ 3D UNet (ε-prediction, 25-step DDIM + CFG) ──▶ latent z_0
                                                                │
                        3D-conv VAE decoder (8³ → 32³) ◀────────┘
                                   │
                          occupancy volume
                                   │
                     marching cubes (auto-oriented)
                                   │
                        mesh ──▶ GLB / OBJ / STL
```

- **Dataset**: 3,000+ procedurally generated 32³ occupancy grids — 18 shape
  primitives (sphere, cube, torus, rocket, snowman, chair, …) with random
  scale/aspect/rotation, plus compositional scenes (stacking, side-by-side,
  pairs) and adjective modifiers (tall, flat, chunky, …). Every sample
  carries a ground-truth caption.
- **3D VAE** (0.55M params): compresses 32³ voxels into 4×8³ latents
  (16× spatial compression), β-VAE objective.
- **Latent diffusion** (2.2M params): DDPM in latent space, 1000-step linear
  schedule, **classifier-free guidance** training (15% condition dropout),
  DDIM sampling at inference.
- **Shape classifier** (auxiliary): used for evaluation and the
  image-to-3D refinement path.
- **Meshing**: marching cubes with consistent-winding BFS + signed-volume
  face orientation → watertight meshes with outward normals.

## 📊 Results

Trained on CPU (16 threads) in ~2.5h. Metrics from the held-out evaluation:

| Metric | Value |
|---|---|
| VAE reconstruction IoU (mean / min) | **0.933** / 0.771 |
| Generation IoU vs same-class reference | 0.277 |
| Chamfer distance (generated vs reference) | 0.0457 |
| Mesh quality | watertight, Euler χ = 2 for genus-0 shapes (χ = 4 for two-component scenes), 100% outward normals |
| GLB export | 0 errors / 0 warnings on the official Khronos glTF validator |

*(The eval block is written into `web/models/config.json` at the end of every
training run and displayed in the deployed app.)*

## 🚀 Quick start

```bash
# install
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# tests (35 unit tests + browser marching-cubes verification)
python -m pytest tests/ -q
node web/js/verify-mc.mjs

# train from scratch (CPU, ~2.5h) → checkpoints/ + web/models/*.onnx
python -m murti.train

# or a 2-minute smoke run
python -m murti.train --quick

# generate from Python
python - <<'EOF'
from murti.pipeline import Pipeline
pipe = Pipeline.load("checkpoints")
res = pipe.generate("a rocket", seed=7, cfg_scale=3.0)
from murti.export import export_glb
export_glb(res.mesh, "rocket.glb")
print(res.stats)
EOF

# serve the web app locally
cd web && python -m http.server 8080
```

## 🌐 Web app

`web/` is a zero-backend static app:

- **Three.js** viewer with orbit controls, wireframe toggle, auto-rotate
- **ONNX Runtime Web (WASM)** runs the diffusion loop client-side
- JS marching cubes + GLB/OBJ/STL exporters mirror the Python code exactly
- image-to-3D visual hull carving happens entirely in the browser

Nothing you type or upload ever leaves your machine.

## 📁 Repository layout

```
murti/
  shapes.py        procedural 3D shape engine + captions
  dataset.py       cached PyTorch dataset
  text.py          tokenizer / vocabulary
  models.py        3D VAE, latent-diffusion UNet, classifier
  export.py        marching cubes + OBJ/STL/GLB writers
  pipeline.py      end-to-end generation + image-to-3D visual hull
  metrics.py       IoU, chamfer, surface sampling
  onnx_export.py   ONNX export for the browser
  train.py         full training pipeline (VAE → diffusion → clf → eval)
tests/             35 pytest tests
web/               static web app (Three.js + ONNX Runtime Web)
  js/              tokenizer, inference, marching cubes, exporters, app
  models/          *.onnx + config.json (written by train.py)
checkpoints/       *.pt weights + sample exports
```

## 🧪 Testing

- **35 unit tests** cover the shape engine, tokenizer, model forward passes,
  losses, DDIM sampling reproducibility, meshing, all three exporters
  (binary STL size check, glTF structure parse, OBJ counts), metrics, the
  visual-hull path, and the end-to-end pipeline.
- **Browser marching cubes** is verified in Node: watertightness (every edge
  shared by exactly 2 faces), Euler characteristic, coordinate mapping, and
  outward-normal percentage on a reference sphere.
- **CI** runs the full pytest suite, the Node mesh verification, and a smoke
  training run on every push.

## ⚖️ Ethics & limitations

See [ETHICS.md](ETHICS.md) and [MODEL_CARD.md](MODEL_CARD.md). Highlights:
synthetic-only training data (no scraped content, no copyright exposure),
fully local inference (no data collection), documented failure modes
(out-of-vocabulary prompts degrade gracefully), and an honest statement of
scope — this is a research-scale generative-3D system, not a replacement
for large pretrained models like Point-E/Shap-E/TripoSR.

## 📄 License

MIT — see [LICENSE](LICENSE).
