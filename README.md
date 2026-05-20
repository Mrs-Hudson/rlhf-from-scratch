# rlhf-from-scratch

A from-scratch walk through modern post-training: SFT → DPO → PPO → GRPO at the 0.5B–1.5B scale. Same base model, same evals, same compute budget. Four blog posts on what works, what breaks, and what the practitioner literature gets wrong about small-scale RLHF.

## Status

Week 1: RLHF end-to-end with reward modeling — in progress.

## Setup

```bash
uv sync
uv run python scripts/sanity.py
```
