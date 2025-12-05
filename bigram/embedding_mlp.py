"""Embedding + MLP bigram model (Makemore part 2/3 style)."""

from typing import Dict, List

import torch

from bigram.data import build_dataset, build_vocab, load_words, split_words


class EmbeddingMLP:
    """Small MLP with a learnable embedding table."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 3,
        n_embd: int = 10,
        n_hidden: int = 200,
        generator=None,
    ) -> None:
        g = generator
        self.block_size = block_size
        self.C = torch.randn((vocab_size, n_embd), generator=g) * 0.1
        self.W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.1
        self.b1 = torch.randn(n_hidden, generator=g) * 0.1
        self.W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
        self.b2 = torch.randn(vocab_size, generator=g) * 0.1
        self.parameters_list = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters_list:
            p.requires_grad = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch, block_size) of indices
        emb = self.C[X]  # (batch, block_size, n_embd)
        h = torch.tanh(emb.view(emb.shape[0], -1) @ self.W1 + self.b1)  # simple one-hidden-layer MLP
        return h @ self.W2 + self.b2

    def parameters(self) -> List[torch.Tensor]:
        return self.parameters_list


def evaluate(model: EmbeddingMLP, X: torch.Tensor, Y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model.forward(X)
        loss = F.cross_entropy(logits, Y)
    return loss.item()


def sample(model: EmbeddingMLP, itos: Dict[int, str], num_samples: int = 10, seed: int = 2147483647) -> List[str]:
    """Generate names by rolling the MLP one character at a time."""
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(num_samples):
        context = [0] * model.block_size
        out = []
        while True:
            logits = model.forward(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        names.append("".join(itos[i] for i in out[:-1]))
    return names


def main() -> None:
    words = load_words()
    train_words, val_words, test_words = split_words(words)
    stoi, itos = build_vocab(words)

    block_size = 3  # same as makemore part 3 default
    X_train, Y_train = build_dataset(train_words, block_size, stoi)
    X_val, Y_val = build_dataset(val_words, block_size, stoi)
    X_test, Y_test = build_dataset(test_words, block_size, stoi)

    g = torch.Generator().manual_seed(42)
    model = EmbeddingMLP(len(stoi), block_size=block_size, n_embd=10, n_hidden=200, generator=g)

    max_steps = 60000
    batch_size = 32
    for step in range(max_steps):
        ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
        logits = model.forward(X_train[ix])
        loss = F.cross_entropy(logits, Y_train[ix])

        for p in model.parameters():
            p.grad = None
        loss.backward()

        lr = 0.1 if step < 40000 else 0.01  # follow the notebook schedule
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 10000 == 0:
            train_loss = evaluate(model, X_train, Y_train)
            val_loss = evaluate(model, X_val, Y_val)
            print(f"{step:6d}/{max_steps}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    print("\nSample names after training:")
    for name in sample(model, itos, num_samples=10, seed=123):
        print(f"- {name}")

    print("\nFinal losses:")
    print(f"- train: {evaluate(model, X_train, Y_train):.4f}")
    print(f"- val:   {evaluate(model, X_val, Y_val):.4f}")
    print(f"- test:  {evaluate(model, X_test, Y_test):.4f}")


if __name__ == "__main__":
    main()
