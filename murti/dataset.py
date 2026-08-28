"""PyTorch datasets over the procedural shape engine.

Samples are generated once per (n, seed) and cached in memory, so epochs
are free and the dataset is deterministic per index.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .shapes import GRID, ShapeSample, build_scene
from .text import tokenize

_GEN_CACHE: Dict[Tuple[int, int, int], List[ShapeSample]] = {}


def _generate_all(n: int, seed: int, grid: int) -> List[ShapeSample]:
    key = (n, seed, grid)
    if key not in _GEN_CACHE:
        _GEN_CACHE[key] = [
            build_scene(np.random.default_rng(seed * 100003 + i), grid)
            for i in range(n)
        ]
    return _GEN_CACHE[key]


class ShapeDataset(Dataset):
    """(occupancy[1,g,g,g], token_ids[MAX_LEN], label, caption) tuples."""

    def __init__(self, n_samples: int, seed: int = 1234, grid: int = GRID):
        self.n = n_samples
        self.seed = seed
        self.grid = grid
        self.samples = _generate_all(n_samples, seed, grid)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        sample = self.samples[i]
        occ = torch.from_numpy(sample.occupancy).float().unsqueeze(0)
        toks = torch.tensor(tokenize(sample.caption), dtype=torch.long)
        return occ, toks, sample.label, sample.caption


def make_loaders(n_train: int, n_val: int, batch_size: int, seed: int = 1234,
                 num_workers: int = 0):
    train = ShapeDataset(n_train, seed=seed)
    val = ShapeDataset(n_val, seed=seed + 777)
    tl = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True)
    vl = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tl, vl


def single_prompt_batch(captions, device: Optional[torch.device] = None):
    """Tokenize a list of caption strings into a batch tensor."""
    toks = torch.tensor([tokenize(c) for c in captions], dtype=torch.long)
    if device is not None:
        toks = toks.to(device)
    return toks
