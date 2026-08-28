"""Tests for the procedural shape engine and captions."""
import numpy as np
import pytest

from murti.shapes import (ADJECTIVES, BUILDERS, CLASS_NAMES, GRID, LABEL,
                          build_scene, build_shape)
from murti.text import MAX_LEN, VOCAB, tokenize, decode


def test_all_builders_produce_solid_shapes():
    rng = np.random.default_rng(0)
    for name in BUILDERS:
        occ, cap = build_shape(name, np.random.default_rng(1), GRID)
        assert occ.shape == (GRID, GRID, GRID)
        frac = occ.mean()
        assert 0.005 < frac < 0.6, f"{name} occupancy {frac} out of range"
        assert name in cap


def test_shapes_are_deterministic_per_seed():
    a, _ = build_shape("torus", np.random.default_rng(42), GRID)
    b, _ = build_shape("torus", np.random.default_rng(42), GRID)
    assert np.allclose(a, b)


def test_adjectives_change_volume():
    base, _ = build_shape("cube", np.random.default_rng(7), GRID)
    small, _ = build_shape("cube", np.random.default_rng(7), GRID, adj="small")
    large, _ = build_shape("cube", np.random.default_rng(7), GRID, adj="large")
    assert (small > 0.5).sum() < (base > 0.5).sum() < (large > 0.5).sum()


def test_scene_kinds_and_labels():
    kinds = set()
    for i in range(60):
        s = build_scene(np.random.default_rng(1000 + i), GRID)
        assert s.occupancy.shape == (GRID, GRID, GRID)
        assert 0 <= s.label < len(CLASS_NAMES)
        assert isinstance(s.caption, str) and len(s.caption) > 2
        kinds.add(s.kind)
    assert kinds == {"single", "stack", "side", "pair"}


def test_captions_tokenize_in_vocab():
    for i in range(40):
        s = build_scene(np.random.default_rng(500 + i), GRID)
        toks = tokenize(s.caption)
        assert len(toks) == MAX_LEN
        assert all(0 <= t < len(VOCAB) for t in toks)
        # no <unk> for generated captions
        assert 1 not in toks, f"unk in caption: {s.caption}"


def test_label_map_covers_class_names():
    assert set(LABEL) == set(CLASS_NAMES)
    assert LABEL["compound"] == len(CLASS_NAMES) - 1
