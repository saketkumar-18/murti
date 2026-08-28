"""Tests for model forward passes, losses, and sampling shapes."""
import torch

from murti.models import (LatentDiffusion, ShapeClassifier, TextEncoder,
                          VAE3D)
from murti.shapes import CLASS_NAMES
from murti.text import MAX_LEN, VOCAB_SIZE


def test_text_encoder_shape():
    enc = TextEncoder()
    toks = torch.randint(0, VOCAB_SIZE, (4, MAX_LEN))
    out = enc(toks)
    assert out.shape == (4, 64)


def test_vae_forward_and_loss():
    vae = VAE3D()
    x = (torch.rand(2, 1, 32, 32, 32) > 0.8).float()
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == (2, 4, 8, 8, 8)
    loss, parts = vae.loss(x)
    assert loss.requires_grad
    assert all(k in parts for k in ("bce", "kl", "total"))


def test_vae_reconstruct_deterministic():
    vae = VAE3D().eval()
    x = (torch.rand(1, 1, 32, 32, 32) > 0.8).float()
    r = vae.reconstruct(x)
    assert r.shape == x.shape
    assert ((r >= 0) & (r <= 1)).all()


def test_diffusion_q_sample_and_loss():
    diff = LatentDiffusion()
    z0 = torch.randn(2, 4, 8, 8, 8)
    toks = torch.randint(0, VOCAB_SIZE, (2, MAX_LEN))
    diff.train()
    loss, parts = diff.training_loss(z0, toks)
    assert loss.requires_grad and parts["mse"] > 0
    # q_sample at t=0 ~ z0, at t=T-1 ~ noise
    t0 = torch.zeros(2, dtype=torch.long)
    noisy0, _ = diff.q_sample(z0, t0)
    assert torch.allclose(noisy0, z0, atol=0.05)


def test_diffusion_ddim_sample_shape():
    diff = LatentDiffusion().eval()
    toks = torch.randint(0, VOCAB_SIZE, (1, MAX_LEN))
    z = diff.sample(toks, (4, 8, 8, 8), ddim_steps=3, cfg_scale=2.0, seed=0)
    assert z.shape == (1, 4, 8, 8, 8)
    assert torch.isfinite(z).all()


def test_diffusion_seed_reproducible():
    diff = LatentDiffusion().eval()
    toks = torch.randint(0, VOCAB_SIZE, (1, MAX_LEN))
    z1 = diff.sample(toks, (4, 8, 8, 8), ddim_steps=3, seed=123)
    z2 = diff.sample(toks, (4, 8, 8, 8), ddim_steps=3, seed=123)
    assert torch.allclose(z1, z2)


def test_classifier_shape():
    clf = ShapeClassifier(len(CLASS_NAMES))
    x = torch.rand(2, 1, 32, 32, 32)
    logits = clf(x)
    assert logits.shape == (2, len(CLASS_NAMES))
