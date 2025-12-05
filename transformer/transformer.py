"""Minimal decoder-only Transformer for character-level modeling."""

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters (tweak freely)
batch_size = 64
block_size = 64  # max context length
max_iters = 5000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
n_embd = 128
n_layer = 6
n_head = 4
dropout = 0.2
device = "mps" if torch.mps.is_available() else "cpu"


# -------------- data prep --------------
def load_text() -> str:
    """Read the two tiny training text files and glue them together."""
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "..", "data", "input_llm.txt"), "r") as f:
        text1 = f.read()
    with open(os.path.join(here, "..", "data", "input_llm2.txt"), "r") as f:
        text2 = f.read()
    return text1 + text2


def build_tokenizer(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Every unique char gets an id; also build reverse map."""
    chars = sorted(list(set(text)))
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(ids, itos: Dict[int, str]) -> str:
    return "".join(itos[i] for i in ids)


def split_data(data: torch.Tensor, train_frac: float = 0.9):
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data, split: str):
    """Grab a random batch from train/val split."""
    src = data["train"] if split == "train" else data["val"]
    ix = torch.randint(0, len(src) - block_size, (batch_size,))  # random start positions
    x = torch.stack([src[i : i + block_size] for i in ix])
    y = torch.stack([src[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data):
    """Average loss over a few batches for train/val."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# -------------- model parts --------------
class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, head_size: int):
        super().__init__()
        # project token embeddings into key/query/value spaces
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower-triangular mask to block attention to the future
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        head_dim = k.shape[-1]
        # scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # block tokens ahead of us
        wei = F.softmax(wei, dim=-1)  # turn scores into probabilities
        wei = self.dropout(wei)  # regularize a bit
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Bundle several heads then project back."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # mix the heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple 2-layer MLP applied at each time step."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # project back down
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block = attention + MLP with residuals."""

    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramTransformer(nn.Module):
    """Decoder-only Transformer that predicts the next char."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)  # (B, T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok + pos  # add position info to token embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        """Autoregressively sample new tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)  # forward pass on current context
            logits = logits[:, -1, :]  # only keep last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample next char id
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# -------------- training loop --------------
def main():
    torch.manual_seed(1337)
    print(f"Using device: {device}")  # MPS on Macs if available

    raw_text = load_text()
    stoi, itos = build_tokenizer(raw_text)
    data_full = encode(raw_text, stoi)
    train_data, val_data = split_data(data_full)
    data = {"train": train_data, "val": val_data}

    model = BigramTransformer(vocab_size=len(stoi)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(model, data)
            print(f"step {step:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        xb, yb = get_batch(data, "train")  # context tokens + targets
        # forward + loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nSampled text:\n")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    print(decode(generated, itos))


if __name__ == "__main__":
    main()
