// Murti in-browser inference: ONNX Runtime Web sessions + DDIM sampling
// with classifier-free guidance. Mirrors murti.models.LatentDiffusion.sample
// step for step so browser outputs match the Python pipeline for a seed.
import { tokenize } from "./tokenizer.js";

// Deterministic PRNG so seeds reproduce (mulberry32 + Box-Muller).
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export class MurtiInference {
  constructor(modelsBase = "models") {
    this.modelsBase = modelsBase;
    this.ready = false;
    this.config = null;
    this.sessions = {};
  }

  async load(onProgress) {
    const cfg = await fetch(`${this.modelsBase}/config.json`).then((r) => r.json());
    this.config = cfg;

    // DDPM schedule (mirrors Python: linspace betas, cumprod alpha_bar)
    const { steps, beta_start, beta_end } = cfg;
    this.alphaBar = new Float64Array(steps);
    let cum = 1.0;
    for (let i = 0; i < steps; i++) {
      const beta = beta_start + ((beta_end - beta_start) * i) / (steps - 1);
      cum *= 1.0 - beta;
      this.alphaBar[i] = cum;
    }

    const opts = { executionProviders: ["wasm"] };
    const names = ["decoder", "diffusion"];
    for (let i = 0; i < names.length; i++) {
      const name = names[i];
      onProgress?.(`loading ${name}.onnx (${i + 1}/${names.length})`);
      this.sessions[name] = await ort.InferenceSession.create(
        `${this.modelsBase}/${name}.onnx`, opts);
    }
    this.ready = true;
    return cfg;
  }

  // One UNet noise prediction.
  async _predictNoise(z, tCur, tokens) {
    const { latent_ch: C } = this.config;
    const zT = new ort.Tensor("float32", z, [1, C, 8, 8, 8]);
    const tT = new ort.Tensor("int64", BigInt64Array.from([BigInt(tCur)]), [1]);
    const tokT = new ort.Tensor("int64", BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]);
    const out = await this.sessions.diffusion.run({ z_noisy: zT, t: tT, tokens: tokT });
    return out.noise.data; // Float32Array
  }

  // DDIM sampling with CFG — mirrors LatentDiffusion.sample (eta=0).
  async sample(prompt, { seed = 0, cfgScale = 3.0, ddimSteps = 25, onStep } = {}) {
    if (!this.ready) throw new Error("models not loaded");
    const { latent_ch: C, steps, max_len: MAX_LEN } = this.config;
    const N = C * 8 * 8 * 8;

    const tokens = tokenize(prompt);
    const uncond = new Array(MAX_LEN).fill(0);

    const rng = mulberry32(seed);
    const z = new Float32Array(N);
    for (let i = 0; i < N; i++) z[i] = randn(rng);

    // step_ids = linspace(steps-1, 0, ddimSteps+1).long()
    const stepIds = [];
    for (let i = 0; i <= ddimSteps; i++) {
      stepIds.push(Math.trunc((steps - 1) + (0 - (steps - 1)) * (i / ddimSteps)));
    }

    for (let i = 0; i < ddimSteps; i++) {
      const tCur = stepIds[i];
      const tPrev = stepIds[i + 1];
      const epsC = await this._predictNoise(z, tCur, tokens);
      const epsU = await this._predictNoise(z, tCur, uncond);
      const aT = this.alphaBar[tCur];
      const aPrev = this.alphaBar[tPrev];

      for (let k = 0; k < N; k++) {
        const eps = epsU[k] + cfgScale * (epsC[k] - epsU[k]);
        let z0 = (z[k] - Math.sqrt(1 - aT) * eps) / Math.sqrt(aT);
        z0 = Math.max(-4, Math.min(4, z0));
        const dir = Math.sqrt(Math.max(0, 1 - aPrev)) * eps;
        z[k] = Math.sqrt(aPrev) * z0 + dir;
      }
      onStep?.(i + 1, ddimSteps);
    }
    return z;
  }

  // Decode latent -> occupancy volume (sigmoid applied, matches Python).
  async decode(z) {
    const { latent_ch: C, grid } = this.config;
    const zT = new ort.Tensor("float32", z, [1, C, 8, 8, 8]);
    const out = await this.sessions.decoder.run({ latent: zT });
    return out.volume.data; // Float32Array grid^3, already sigmoided
  }

  async generate(prompt, opts = {}) {
    const z = await this.sample(prompt, opts);
    const volume = await this.decode(z);
    return { volume, grid: this.config.grid, prompt };
  }
}

// ---------------------------------------------------------------------------
// Image-to-3D: silhouette visual hull (mirrors murti.pipeline)
// ---------------------------------------------------------------------------

export function imageToSilhouette(imgElement, grid = 32, threshold = 200) {
  const canvas = document.createElement("canvas");
  canvas.width = grid; canvas.height = grid;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(imgElement, 0, 0, grid, grid);
  const data = ctx.getImageData(0, 0, grid, grid).data;
  const gray = new Float32Array(grid * grid);
  for (let i = 0; i < grid * grid; i++) {
    // luminance
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  // polarity: object = smaller foreground
  let dark = 0, light = 0;
  for (let i = 0; i < gray.length; i++) {
    if (gray[i] < threshold) dark++; else light++;
  }
  const sil = new Uint8Array(grid * grid);
  const useDark = dark <= light;
  for (let i = 0; i < gray.length; i++) {
    sil[i] = useDark ? (gray[i] < threshold ? 1 : 0) : (gray[i] >= threshold ? 1 : 0);
  }
  return sil;
}

function rot90(s, k, grid) {
  // numpy rot90 semantics on a square matrix, k times CCW
  let out = s;
  for (let i = 0; i < ((k % 4) + 4) % 4; i++) {
    const next = new Uint8Array(grid * grid);
    for (let r = 0; r < grid; r++)
      for (let c = 0; c < grid; c++)
        next[(grid - 1 - c) * grid + r] = out[r * grid + c];
    out = next;
  }
  return out;
}

function extrude(sil, grid) {
  // vol[x][y][z] = sil[y][z] (extrusion along x), flat x + y*grid + z*grid*grid
  const vol = new Float32Array(grid * grid * grid);
  for (let x = 0; x < grid; x++)
    for (let y = 0; y < grid; y++)
      for (let z = 0; z < grid; z++)
        vol[x + y * grid + z * grid * grid] = sil[y * grid + z];
  return vol;
}

function gaussian3d(vol, grid, sigma = 0.8) {
  const radius = 2;
  const kernel = [];
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const w = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(w); sum += w;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

  const blurAxis = (src, axis) => {
    const dst = new Float32Array(src.length);
    const stride = [1, grid, grid * grid][axis];
    const dims = [grid, grid, grid];
    for (let idx = 0; idx < src.length; idx++) {
      const coord = Math.floor(idx / stride) % grid;
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let c2 = coord + k;
        c2 = Math.max(0, Math.min(grid - 1, c2)); // clamp (nearest)
        acc += src[idx + (c2 - coord) * stride] * kernel[k + radius];
      }
      dst[idx] = acc;
    }
    return dst;
  };
  let v = vol;
  for (let axis = 0; axis < 3; axis++) v = blurAxis(v, axis);
  return v;
}

export function silhouetteToVolume(sil, grid = 32, carveRotations = true) {
  let hull = extrude(sil, grid);
  if (carveRotations) {
    for (let k = 1; k <= 3; k++) {
      const rot = rot90(sil, k, grid);
      const e = extrude(rot, grid);
      for (let i = 0; i < hull.length; i++) hull[i] = Math.min(hull[i], e[i]);
    }
  }
  return gaussian3d(hull, grid, 0.8);
}

export function imageToVolume(imgElement, grid = 32) {
  const sil = imageToSilhouette(imgElement, grid);
  return silhouetteToVolume(sil, grid, true);
}
