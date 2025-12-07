# GPT From Scratch – Learning Project

This repository documents my personal learning journey implementing neural networks and a miniature GPT-style Transformer from scratch, inspired by Andrej Karpathy’s “Neural Networks: Zero to Hero” series. My goal was to re-derive, rewrite, and comment the pieces myself to build intuition for how modern LLM components actually work.

⸻

📜 Journey Through the Repo
- micrograd/ — started by writing a tiny autodiff engine and MLP from scratch
- bigram/statistical — first baseline: pure counts of character transitions, no learning loop, just probabilities from data.
- bigram/one_hot_bigram.py — upgraded to a learnable bigram table (one-hot → linear layer); context size 1, effectively logistic regression on characters.
- bigram/embedding_mlp.py — replaced one-hot with embeddings and a small MLP over a longer context (block_size=3); learned shared representations and nonlinear patterns.
- transformer/transformer.py — assembled embeddings, positional encodings, causal masking, multi-head self-attention, residuals, and layer norm into a decoder-only Transformer; trained a mini GPT on toy text to see generation emerge.
- wavenet/ + notebooks — extra experiments mirroring the series for curiosity and sanity checks.

⸻

🎯 What I Learned
- Fundamentals of backpropagation and gradient flow by implementing and debugging autodiff.
- How batching, optimization steps, and training loops fit together in small MLPs.
- Character-level tokenization, vocab mapping, and sequence encoding for tiny datasets.
- Why embeddings help share statistics and how context length changes modeling power.
- Self-attention mechanics, causal masking, and how heads attend to relevant tokens.
- Multi-head attention, positional encodings, and residual connections for stable training.
- How a decoder-only Transformer stitches these pieces into a working mini GPT.

⸻

📁 Repository Overview
- micrograd/ – tiny autodiff engine with a small MLP demo.
- bigram/ – statistical bigram, one-hot bigram, and embedding+MLP models (shared helpers in data.py).
- transformer/ – minimal decoder-only Transformer; transformer/transformer.py is the runnable GPT-style model.
- wavenet/ – follow-along experiments from the series.
- notebooks/ – exploratory notebooks used during learning.
- data/ – toy datasets (names and small text snippets).

⸻

▶️ Running the Code
- Install dependencies: pip install -r requirements.txt
- Run a demo (e.g., Transformer): python transformer/transformer.py

⸻
