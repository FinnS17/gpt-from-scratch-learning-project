"""Bare-bones statistical bigram name generator."""

from typing import Dict, Iterable, List, Tuple
import torch
from bigram.data import build_vocab, load_words


def count_bigrams(words: Iterable[str], stoi: Dict[str, int]) -> torch.Tensor:
    """Count how often every char follows every other char."""
    vocab_size = len(stoi)
    counts = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            counts[stoi[ch1], stoi[ch2]] += 1
    return counts


def make_prob_matrix(counts: torch.Tensor) -> torch.Tensor:
    """Counts -> probabilities with tiny smoothing so nothing is impossible."""
    probs = (counts + 1).float()  # add-one smoothing
    probs /= probs.sum(1, keepdim=True)
    return probs


def sample_names(probs: torch.Tensor, itos: Dict[int, str], num_samples: int = 10, seed: int = 1337) -> List[str]:
    """Walk the Markov chain until we hit '.'."""
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(num_samples):
        ix = 0  # start token '.'
        out = []
        while True:
            p_dist = probs[ix]
            ix = torch.multinomial(p_dist, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        names.append("".join(out[:-1]))  # drop the final '.'
    return names


def dataset_nll(words: Iterable[str], probs: torch.Tensor, stoi: Dict[str, int]) -> Tuple[float, float]:
    """Negative log-likelihood of the training set."""
    total = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            prob = probs[stoi[ch1], stoi[ch2]]
            total -= torch.log(prob).item()
            n += 1
    return total, total / n


def main() -> None:
    words = load_words()
    stoi, itos = build_vocab(words)

    counts = count_bigrams(words, stoi)
    probs = make_prob_matrix(counts)

    print(f"vocab size (including '.'): {len(stoi)}")
    print(f"total training words: {len(words)}")

    print("\nSample names from the statistical bigram model:")
    for name in sample_names(probs, itos, num_samples=10):
        print(f"- {name}")

    _, avg_nll = dataset_nll(words, probs, stoi)
    print("\nModel fit (lower is better):")
    print(f"- average per bigram: {avg_nll:.4f}")


if __name__ == "__main__":
    main()
