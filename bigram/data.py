"""Tiny helper functions shared by the bigram demos."""

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

def load_words(data_path: Optional[Path] = None) -> List[str]:
    """Read the names file (defaults to ../data/names.txt)."""
    if data_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / "data" / "names.txt"
    return data_path.read_text().splitlines()


def build_vocab(words: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """String<->index lookups; '.' is both start and end token."""
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_dataset(words: Sequence[str], block_size: int, stoi: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn words into (context, next_char) pairs."""
    X, Y = [], []
    for word in words:
        context = [0] * block_size  # leading dots
        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # slide window
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def split_words(words: Sequence[str], train: float = 0.8, val: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle and split the list into train/val/test."""
    words = list(words)
    rng = random.Random(seed)
    rng.shuffle(words)
    n = len(words)
    n_train = int(train * n)
    n_val = int(val * n)
    train_words = words[:n_train]
    val_words = words[n_train:n_train + n_val]
    test_words = words[n_train + n_val:]
    return train_words, val_words, test_words
