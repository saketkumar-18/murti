"""Tiny text tokenizer + vocabulary for Murti prompts.

The vocabulary covers shape nouns, size/aspect adjectives and the
compositional glue words used by the procedural caption generator.
Unknown words map to <unk>; naive plural stripping ("spheres" -> "sphere")
keeps generated captions in-vocabulary.
"""
from __future__ import annotations

from typing import List

SPECIAL = ["<pad>", "<unk>"]
STRUCT = ["a", "on", "top", "of", "next", "to", "and", "two", "three", "with"]
SHAPES = [
    "sphere", "cube", "cylinder", "cone", "torus", "capsule",
    "pyramid", "prism", "star", "arch", "cross", "table",
    "chair", "snowman", "rocket", "tree", "dumbbell", "block",
]
ADJECTIVES = [
    "tall", "wide", "flat", "small", "large", "thin",
    "chunky", "slender", "squat", "stretched",
]

WORDS = SPECIAL + STRUCT + SHAPES + ADJECTIVES
VOCAB = {w: i for i, w in enumerate(WORDS)}
INV_VOCAB = {i: w for w, i in VOCAB.items()}
VOCAB_SIZE = len(WORDS)
PAD_ID = VOCAB["<pad>"]
UNK_ID = VOCAB["<unk>"]
MAX_LEN = 12


def tokenize(caption: str) -> List[int]:
    """Lowercase, split on whitespace, map to ids, pad/truncate to MAX_LEN."""
    ids: List[int] = []
    for word in caption.lower().replace(",", " ").split():
        wid = VOCAB.get(word)
        if wid is None and word.endswith("s"):
            wid = VOCAB.get(word[:-1])  # naive plural strip
        ids.append(UNK_ID if wid is None else wid)
    ids = ids[:MAX_LEN]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids


def decode(tokens) -> str:
    """Map ids back to words, dropping padding."""
    return " ".join(
        INV_VOCAB.get(int(t), "<unk>") for t in tokens if int(t) != PAD_ID
    )
