"""Neural cores: 3D-conv VAE, conditional latent-diffusion UNet, classifier.

All modules are plain PyTorch so they export cleanly to ONNX for the
in-browser inference path (web/).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text import MAX_LEN, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Embedding + 2-layer GRU -> pooled context vector."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, dim: int = 64,
                 max_len: int = MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, num_layers=2, batch_first=True)
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L) long
        x = self.embed(tokens)
        out, _ = self.gru(x)
        mask = (tokens != 0).float().unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled


# ---------------------------------------------------------------------------
# 3D VAE: voxel grid <-> compact latent
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class Encoder3D(nn.Module):
    """(B,1,32,32,32) -> (B, 2*latent_ch, 8,8,8). CPU-light channels."""

    def __init__(self, latent_ch: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.SiLU(),   # 16
            ResBlock3D(32),
            nn.Conv3d(32, 48, 3, stride=2, padding=1), nn.SiLU(),   # 8
            ResBlock3D(48),
            nn.Conv3d(48, 2 * latent_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder3D(nn.Module):
    """(B, latent_ch, 8,8,8) -> (B,1,32,32,32) logits. CPU-light channels."""

    def __init__(self, latent_ch: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(latent_ch, 48, 3, padding=1), nn.SiLU(),
            ResBlock3D(48),
            nn.ConvTranspose3d(48, 32, 4, stride=2, padding=1), nn.SiLU(),  # 16
            ResBlock3D(32),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1), nn.SiLU(),  # 32
            nn.Conv3d(16, 1, 3, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class VAE3D(nn.Module):
    """Beta-VAE over occupancy grids. Latent: (B, latent_ch, 8,8,8)."""

    def __init__(self, latent_ch: int = 4, beta: float = 1e-4):
        super().__init__()
        self.encoder = Encoder3D(latent_ch)
        self.decoder = Decoder3D(latent_ch)
        self.latent_ch = latent_ch
        self.beta = beta

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(self, x) -> Tuple[torch.Tensor, dict]:
        recon, mu, logvar = self.forward(x)
        bce = F.binary_cross_entropy_with_logits(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = bce + self.beta * kl
        return total, {"bce": bce.item(), "kl": kl.item(), "total": total.item()}

    @torch.no_grad()
    def reconstruct(self, x, threshold: float = 0.5) -> torch.Tensor:
        mu, _ = self.encode(x)
        return torch.sigmoid(self.decode(mu))


# ---------------------------------------------------------------------------
# Latent diffusion UNet (operates on 8^3 latents, conditioned on text)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CondResBlock3D(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.cond = nn.Linear(cond_dim, ch)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.cond(cond)[:, :, None, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + x


class LatentUNet3D(nn.Module):
    """Small 3D UNet: (B, latent_ch, 8,8,8) noise prediction, text-conditioned."""

    def __init__(self, latent_ch: int = 4, text_dim: int = 64, base: int = 32):
        super().__init__()
        cond_dim = base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.text = TextEncoder(dim=text_dim)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, cond_dim), nn.SiLU())

        c1, c2, c3 = base, base * 2, base * 2
        self.in_conv = nn.Conv3d(latent_ch, c1, 3, padding=1)
        self.d1 = CondResBlock3D(c1, cond_dim)
        self.down1 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)      # 4
        self.d2 = CondResBlock3D(c2, cond_dim)
        self.down2 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)      # 2
        self.mid1 = CondResBlock3D(c3, cond_dim)
        self.mid2 = CondResBlock3D(c3, cond_dim)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)          # 4
        self.u2 = CondResBlock3D(c2 * 2, cond_dim)
        self.up1 = nn.ConvTranspose3d(c2 * 2, c1, 2, stride=2)      # 8
        self.u1 = CondResBlock3D(c1 * 2, cond_dim)
        self.out_conv = nn.Conv3d(c1 * 2, latent_ch, 3, padding=1)

    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(t) + self.text_proj(self.text(tokens))
        h = self.in_conv(z_noisy)
        s1 = self.d1(h, cond)
        h = self.down1(s1)
        s2 = self.d2(h, cond)
        h = self.down2(s2)
        h = self.mid1(h, cond)
        h = self.mid2(h, cond)
        h = self.up2(h)
        h = self.u2(torch.cat([h, s2], dim=1), cond)
        h = self.up1(h)
        h = self.u1(torch.cat([h, s1], dim=1), cond)
        return self.out_conv(h)


class LatentDiffusion(nn.Module):
    """DDPM in the VAE latent space with classifier-free guidance training."""

    def __init__(self, latent_ch: int = 4, text_dim: int = 64, base: int = 32,
                 steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 uncond_p: float = 0.15):
        super().__init__()
        self.unet = LatentUNet3D(latent_ch, text_dim, base)
        self.steps = steps
        self.uncond_p = uncond_p
        betas = torch.linspace(beta_start, beta_end, steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(z0)
        a = self.sqrt_alpha_bar[t][:, None, None, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None, None, None]
        return a * z0 + b * noise, noise

    def training_loss(self, z0: torch.Tensor, tokens: torch.Tensor,
                      rng: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, dict]:
        B = z0.shape[0]
        device = z0.device
        t = torch.randint(0, self.steps, (B,), device=device)
        noise = torch.randn_like(z0)
        z_noisy, _ = self.q_sample(z0, t, noise)
        # classifier-free guidance: drop the condition with prob uncond_p
        if self.training and self.uncond_p > 0:
            drop = torch.rand(B, device=device) < self.uncond_p
            tokens = tokens.clone()
            tokens[drop] = 0
        pred = self.unet(z_noisy, t, tokens)
        loss = F.mse_loss(pred, noise)
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def sample(self, tokens: torch.Tensor, shape: Tuple[int, ...],
               ddim_steps: int = 25, cfg_scale: float = 3.0,
               eta: float = 0.0, seed: Optional[int] = None) -> torch.Tensor:
        """DDIM sampling with classifier-free guidance."""
        device = next(self.parameters()).device
        B = tokens.shape[0]
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        z = torch.randn((B,) + tuple(shape), generator=gen).to(device)

        step_ids = torch.linspace(self.steps - 1, 0, ddim_steps + 1, dtype=torch.long)
        uncond_tokens = torch.zeros_like(tokens)
        for i in range(ddim_steps):
            t_cur = int(step_ids[i])
            t_prev = int(step_ids[i + 1])
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps_c = self.unet(z, t_batch, tokens)
            eps_u = self.unet(z, t_batch, uncond_tokens)
            eps = eps_u + cfg_scale * (eps_c - eps_u)

            a_t = self.alpha_bar[t_cur]
            a_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            z0_pred = (z - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            z0_pred = torch.clamp(z0_pred, -4.0, 4.0)
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
            dir_term = torch.sqrt(torch.clamp(1 - a_prev - sigma ** 2, min=0.0)) * eps
            z = torch.sqrt(a_prev) * z0_pred + dir_term
            if sigma > 0 and i < ddim_steps - 1:
                z = z + sigma * torch.randn_like(z)
        return z


# ---------------------------------------------------------------------------
# Shape classifier (for image-to-3D conditioning + evaluation)
# ---------------------------------------------------------------------------

class ShapeClassifier(nn.Module):
    def __init__(self, n_classes: int, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, base, 3, stride=2, padding=1), nn.SiLU(),      # 16
            nn.Conv3d(base, base * 2, 3, stride=2, padding=1), nn.SiLU(),  # 8
            nn.Conv3d(base * 2, base * 4, 3, stride=2, padding=1), nn.SiLU(),  # 4
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(base * 4, n_classes),
        )

    def forward(self, x):
        return self.net(x)
