# Ethics Statement — Murti

## 1. Data provenance & copyright

Murti is trained **exclusively on procedurally generated synthetic data**:
analytic shape primitives (spheres, cubes, tori, …) composed and randomized
in code, with machine-generated captions from a fixed grammar.

- No scraped, crawled, or third-party datasets are used.
- No copyrighted 3D assets, meshes, or images appear in training.
- No human-generated text corpora are used; captions are template-based.

Consequently the model cannot memorize or regurgitate any existing artist's
work, and there is no licensing entanglement with dataset providers.

## 2. Privacy

Inference runs **entirely in the user's browser** (ONNX Runtime WebAssembly).
Prompts, uploaded images, and generated meshes never leave the device. The
deployed app is static — there is no server, no analytics, no cookies, and
no telemetry. The Python CLI is offline by design.

## 3. Bias & representation

The training distribution is synthetic geometry, so the social-bias surface
of large language/vision models does not apply. Residual considerations:

- The vocabulary is small and English-only; prompts outside it degrade
  gracefully to `<unk>` rather than failing silently. This is documented in
  the UI and model card.
- Shape categories are everyday objects/primitives; no humans, faces, or
  identity-bearing content can be generated, avoiding deepfake-adjacent
  misuse entirely.

## 4. Misuse assessment

- **Dual-use risk: low.** The system generates coarse, untextured geometry
  of simple objects. It cannot produce realistic humans, weapons-grade
  detail, or deceptive content.
- **3D-printing safety:** STL exports are user responsibility; generated
  meshes are not validated for structural or safety-critical use, and the
  README states they are for prototyping, not engineering parts.
- **Attribution:** generated assets are original synthetic geometry; users
  may export and use them freely (MIT license).

## 5. Environmental cost

Training is deliberately small-scale: ~2.5 hours on a 16-thread CPU
(no GPU cluster), roughly comparable to a few hours of laptop use. Inference
is serverless (client-side WASM), so serving cost and energy use are
effectively zero beyond static file hosting.

## 6. Transparency & reproducibility

- Full training code, tests, and evaluation are open source (MIT).
- Every training run writes its eval metrics into the shipped config, so
  deployed quality claims are auditable.
- Generation is deterministic per seed in both Python and the browser,
  enabling exact reproduction of any output.
- Limitations (vocabulary bounds, 32³ resolution, visual-hull constraints)
  are stated prominently in the README and MODEL_CARD rather than hidden.

## 7. Human oversight

Murti is a creative prototyping tool. Generated assets are expected to be
reviewed and refined by humans before any production use, and the system
makes no autonomous decisions affecting people.
