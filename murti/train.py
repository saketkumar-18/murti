"""Murti training pipeline: VAE -> latent diffusion -> classifier -> eval.

Usage:
    python -m murti.train --quick          # smoke test (~2 min)
    python -m murti.train                  # full run
    python -m murti.train --export-only    # re-export ONNX from checkpoints

Writes checkpoints to checkpoints/ and ONNX models to web/models/.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ShapeDataset
from .export import export_glb, export_obj, export_stl, marching_cubes
from .metrics import chamfer_distance, iou, surface_points
from .models import LatentDiffusion, ShapeClassifier, VAE3D
from .pipeline import Pipeline
from .shapes import CLASS_NAMES, GRID, build_scene
from .text import tokenize

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
WEB_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "models")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str):
    print(f"[murti] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage 1: VAE
# ---------------------------------------------------------------------------

def train_vae(epochs: int, batch: int, lr: float, n_train: int, dev) -> VAE3D:
    vae = VAE3D().to(dev)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    ds = ShapeDataset(n_train, seed=1234)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    val = ShapeDataset(256, seed=99991)
    vdl = DataLoader(val, batch_size=batch, shuffle=False)

    best = float("inf")
    for ep in range(epochs):
        vae.train()
        tot = 0.0
        for occ, _, _, _ in dl:
            occ = occ.to(dev)
            loss, parts = vae.loss(occ)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            tot += parts["total"]
        # validation reconstruction IoU
        vae.eval()
        ious = []
        with torch.no_grad():
            for occ, _, _, _ in vdl:
                occ = occ.to(dev)
                recon = torch.sigmoid(vae.decode(vae.encode(occ)[0]))
                for i in range(occ.shape[0]):
                    ious.append(iou(recon[i, 0].cpu().numpy(), occ[i, 0].cpu().numpy()))
        val_iou = float(np.mean(ious))
        avg = tot / len(dl)
        log(f"VAE ep {ep+1}/{epochs} loss={avg:.4f} val_IoU={val_iou:.4f}")
        if avg < best:
            best = avg
            torch.save(vae.state_dict(), os.path.join(CKPT, "vae.pt"))
    vae.load_state_dict(torch.load(os.path.join(CKPT, "vae.pt"), map_location=dev))
    return vae


# ---------------------------------------------------------------------------
# Stage 2: latent diffusion
# ---------------------------------------------------------------------------

def train_diffusion(vae: VAE3D, epochs: int, batch: int, lr: float,
                    n_train: int, dev) -> LatentDiffusion:
    diff = LatentDiffusion().to(dev)
    opt = torch.optim.AdamW(diff.parameters(), lr=lr)
    ds = ShapeDataset(n_train, seed=4321)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

    vae.eval()
    # pre-encode the whole dataset once (latents are tiny)
    log("pre-encoding latents...")
    latents, tokens_l, labels_l = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            occ, toks, lab, _ = ds[i]
            mu, _ = vae.encode(occ.unsqueeze(0).to(dev))
            latents.append(mu.squeeze(0).cpu())
            tokens_l.append(toks)
            labels_l.append(lab)
    latents = torch.stack(latents)
    tokens_l = torch.stack(tokens_l)
    tds = torch.utils.data.TensorDataset(latents, tokens_l)
    tdl = DataLoader(tds, batch_size=batch, shuffle=True, drop_last=True)

    for ep in range(epochs):
        diff.train()
        tot = 0.0
        for z0, toks in tdl:
            z0, toks = z0.to(dev), toks.to(dev)
            loss, parts = diff.training_loss(z0, toks)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
            opt.step()
            tot += parts["mse"]
        log(f"DIFF ep {ep+1}/{epochs} mse={tot/len(tdl):.5f}")
    torch.save(diff.state_dict(), os.path.join(CKPT, "diffusion.pt"))
    return diff


# ---------------------------------------------------------------------------
# Stage 3: classifier
# ---------------------------------------------------------------------------

def train_classifier(epochs: int, batch: int, lr: float, n_train: int, dev) -> ShapeClassifier:
    clf = ShapeClassifier(len(CLASS_NAMES)).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    ds = ShapeDataset(n_train, seed=7777)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    val = ShapeDataset(512, seed=88881)
    vdl = DataLoader(val, batch_size=batch, shuffle=False)

    for ep in range(epochs):
        clf.train()
        tot, correct, seen = 0.0, 0, 0
        for occ, _, lab, _ in dl:
            occ, lab = occ.to(dev), torch.tensor(lab, device=dev) if isinstance(lab, torch.Tensor) else lab.to(dev)
            logits = clf(occ)
            loss = torch.nn.functional.cross_entropy(logits, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            correct += (logits.argmax(1) == lab).sum().item()
            seen += occ.shape[0]
        # val accuracy
        clf.eval()
        vc, vs = 0, 0
        with torch.no_grad():
            for occ, _, lab, _ in vdl:
                occ = occ.to(dev)
                lab = lab.to(dev)
                logits = clf(occ)
                vc += (logits.argmax(1) == lab).sum().item()
                vs += occ.shape[0]
        log(f"CLF ep {ep+1}/{epochs} loss={tot/len(dl):.4f} "
            f"train_acc={correct/seen:.3f} val_acc={vc/vs:.3f}")
    torch.save(clf.state_dict(), os.path.join(CKPT, "classifier.pt"))
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(pipeline: Pipeline, dev, n_samples: int = 24, seed: int = 2024) -> dict:
    """Reconstruction IoU + generation quality on held-out prompts."""
    rng = np.random.default_rng(seed)
    val = ShapeDataset(n_samples, seed=55555)

    rec_ious = []
    for i in range(n_samples):
        occ, toks, lab, cap = val[i]
        recon = pipeline.vae.reconstruct(occ.unsqueeze(0).to(dev)).squeeze().cpu().numpy()
        rec_ious.append(iou(recon, occ.numpy()))

    # generation: sample from class-name prompts, compare to fresh same-class refs
    prompts = [f"a {name}" for name in
               ["sphere", "cube", "cylinder", "cone", "torus", "pyramid",
                "star", "arch", "cross", "rocket", "snowman", "tree"]]
    gen_ious, chamfers = [], []
    for j, prompt in enumerate(prompts):
        res = pipeline.generate(prompt, seed=seed + j, cfg_scale=3.0, ddim_steps=25)
        ref_rng = np.random.default_rng(9000 + j)
        ref = build_scene(ref_rng, GRID)
        # compare against a same-shape reference built deterministically
        from .shapes import build_shape
        name = prompt.split()[-1]
        ref_occ, _ = build_shape(name, np.random.default_rng(9100 + j), GRID)
        gen_ious.append(iou(res.volume, ref_occ))
        pa = surface_points(res.volume)
        pb = surface_points(ref_occ)
        chamfers.append(chamfer_distance(pa, pb))

    report = {
        "reconstruction_iou_mean": float(np.mean(rec_ious)),
        "reconstruction_iou_min": float(np.min(rec_ious)),
        "generation_iou_vs_reference_mean": float(np.mean(gen_ious)),
        "chamfer_distance_mean": float(np.mean(chamfers)),
        "n_recon": n_samples,
        "n_generated": len(prompts),
    }
    return report


def export_samples(pipeline: Pipeline, dev, out_dir: str):
    """Export a gallery of generated assets in all three formats."""
    os.makedirs(out_dir, exist_ok=True)
    prompts = ["a sphere", "a cube", "a torus", "a rocket", "a star",
               "a snowman", "a tree", "a chair", "a pyramid", "an arch",
               "a sphere on top of a cube", "a cone next to a cylinder"]
    manifest = []
    for i, prompt in enumerate(prompts):
        res = pipeline.generate(prompt, seed=100 + i, cfg_scale=3.0, ddim_steps=25)
        slug = prompt.replace(" ", "_")[:40]
        base = os.path.join(out_dir, f"{i:02d}_{slug}")
        export_glb(res.mesh, base + ".glb", name=slug)
        export_obj(res.mesh, base + ".obj", name=slug)
        export_stl(res.mesh, base + ".stl")
        manifest.append({"prompt": prompt, "stats": res.stats,
                         "files": [base + f".{e}" for e in ("glb", "obj", "stl")]})
        log(f"exported '{prompt}' -> {res.stats['faces']} faces")
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--skip-export-samples", action="store_true")
    args = ap.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    dev = device()
    log(f"device: {dev}")
    t0 = time.time()

    if args.quick:
        cfg = dict(vae_epochs=2, diff_epochs=2, clf_epochs=2,
                   vae_n=800, diff_n=800, clf_n=800, batch=16,
                   lr=2e-4)
    else:
        cfg = dict(vae_epochs=8, diff_epochs=16, clf_epochs=8,
                   vae_n=3000, diff_n=3000, clf_n=3000, batch=16,
                   lr=2e-4)

    if not args.export_only:
        vae = train_vae(cfg["vae_epochs"], cfg["batch"], cfg["lr"], cfg["vae_n"], dev)
        diff = train_diffusion(vae, cfg["diff_epochs"], cfg["batch"], cfg["lr"], cfg["diff_n"], dev)
        clf = train_classifier(cfg["clf_epochs"], cfg["batch"], cfg["lr"], cfg["clf_n"], dev)
    else:
        vae, diff, clf = VAE3D(), LatentDiffusion(), ShapeClassifier(len(CLASS_NAMES))
        vae.load_state_dict(torch.load(os.path.join(CKPT, "vae.pt"), map_location=dev))
        diff.load_state_dict(torch.load(os.path.join(CKPT, "diffusion.pt"), map_location=dev))
        clf.load_state_dict(torch.load(os.path.join(CKPT, "classifier.pt"), map_location=dev))

    pipeline = Pipeline(vae, diff, clf, dev)

    log("evaluating...")
    report = evaluate(pipeline, dev)
    log(f"eval: {json.dumps(report, indent=2)}")

    if not args.skip_export_samples:
        export_samples(pipeline, dev, os.path.join(CKPT, "samples"))

    log("exporting ONNX for the browser...")
    from .onnx_export import export_onnx
    paths = export_onnx(vae, diff, WEB_MODELS)
    for k, v in paths.items():
        log(f"  {k}: {v} ({os.path.getsize(v)/1e6:.1f} MB)")

    # write a small config the JS runtime reads
    web_cfg = {
        "steps": diff.steps,
        "beta_start": 1e-4, "beta_end": 0.02,
        "latent_ch": vae.latent_ch, "latent_res": 8, "grid": GRID,
        "max_len": 12, "vocab_size": 40,
        "default_cfg": 3.0, "default_ddim_steps": 25,
        "eval": report,
    }
    with open(os.path.join(WEB_MODELS, "config.json"), "w") as fh:
        json.dump(web_cfg, fh, indent=2)

    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
