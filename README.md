# GPT From Scratch – Learning Project

This repo is my personal learning journey through building neural networks from scratch, strongly inspired by Andrej Karpathy’s “Neural Networks: Zero to Hero” series. I rewrote the core pieces myself by following along, pausing, re-deriving, commenting, and debugging. I wanted to demystify how modern LLMs work internally by rebuilding the core parts on my own. 

What I tried to understand:
- backpropagation and tiny autodiff engines
- simple tokenization, MLPs, and embeddings
- batch norm and training loops
- attention, transformers, and small GPT-style models

## Repo at a glance
- `micrograd/` — tiny autograd + mini MLP demo notebook.
- `bigram/` — statistical bigram, one-hot bigram, and embedding+MLP name generators (`data.py` has shared helpers).
- `transformer/transformer.py` — minimal decoder-only Transformer on the small text files in `data/`.
- `wavenet/` and notebooks — more experiments following the series.
- `data/` — toy datasets (names, small text snippets).

## Running stuff

1. Install deps: `pip install -r requirements.txt`
2. Run a demo, e.g: `python transformer/transformer.py`
