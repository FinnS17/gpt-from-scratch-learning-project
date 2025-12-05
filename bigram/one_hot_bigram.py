"""One-hot bigram neural net (logistic regression on characters)."""

from typing import Dict, List
import torch
import torch.nn.functional as F
from bigram.data import build_dataset, build_vocab, load_words, split_words


class OneHotBigram:
    """Single-layer bigram model; rows are logits for the next char."""

    def __init__(self, vocab_size: int, generator=None) -> None:
        self.W = torch.randn((vocab_size, vocab_size), generator=generator, requires_grad=True) * 0.01

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch, 1) indices -> one-hot -> matmul with W
        xenc = F.one_hot(X, num_classes=self.W.shape[0]).float()  # turn indices into one-hot
        xenc = xenc.view(X.shape[0], -1)
        return xenc @ self.W

    def parameters(self) -> List[torch.Tensor]:
        return [self.W]


def evaluate(model: OneHotBigram, X: torch.Tensor, Y: torch.Tensor) -> float:
    """Return cross-entropy loss for a split."""
    with torch.no_grad():
        logits = model.forward(X)
        loss = F.cross_entropy(logits, Y)
    return loss.item()


def sample(model: OneHotBigram, itos: Dict[int, str], num_samples: int = 10, seed: int = 2147483647) -> List[str]:
    """Generate names by sampling from the learned probability table."""
    g = torch.Generator().manual_seed(seed)
    probs = F.softmax(model.W, dim=1)
    names = []
    for _ in range(num_samples):
        ix = 0
        out = []
        while True:
            p_dist = probs[ix]
            ix = torch.multinomial(p_dist, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        names.append("".join(out[:-1]))
    return names


def main() -> None:
    words = load_words()
    train_words, val_words, test_words = split_words(words)
    stoi, itos = build_vocab(words)

    block_size = 1
    X_train, Y_train = build_dataset(train_words, block_size, stoi)
    X_val, Y_val = build_dataset(val_words, block_size, stoi)
    X_test, Y_test = build_dataset(test_words, block_size, stoi)

    g = torch.Generator().manual_seed(1337)
    model = OneHotBigram(len(stoi), generator=g)

    max_steps = 2000
    batch_size = 64
    for step in range(max_steps):
        ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
        logits = model.forward(X_train[ix])
        loss = F.cross_entropy(logits, Y_train[ix])

        for p in model.parameters():
            p.grad = None
        loss.backward()

        lr = 50.0  # same aggressive learning rate as the notebook
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 400 == 0:
            val_loss = evaluate(model, X_val, Y_val)
            print(f"{step:4d}/{max_steps}: train_loss={loss.item():.4f} | val_loss={val_loss:.4f}")

    print("\nSample names after training:")
    for name in sample(model, itos, num_samples=10, seed=123):
        print(f"- {name}")

    print("\nFinal losses:")
    print(f"- train: {evaluate(model, X_train, Y_train):.4f}")
    print(f"- val:   {evaluate(model, X_val, Y_val):.4f}")
    print(f"- test:  {evaluate(model, X_test, Y_test):.4f}")


if __name__ == "__main__":
    main()
