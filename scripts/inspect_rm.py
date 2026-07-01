"""Inspect a trained reward model qualitatively.

Loads the RM, runs it on held-out HH-RLHF pairs, prints:
  - Score distributions (chosen vs rejected)
  - Sample agreements and disagreements
  - The most confident correct and incorrect predictions

The goal is intuition for what the RM has actually learned, not just
aggregate accuracy metrics. Useful before kicking off PPO, and great
material for the blog post.

Usage:
    uv run python scripts/inspect_rm.py
"""
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm-path", default="./rm-final")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Pairs to score for distribution stats.")
    parser.add_argument("--n-display", type=int, default=3,
                        help="Examples to print in each category.")
    parser.add_argument("--max-length", type=int, default=768)
    return parser.parse_args()


def score_pair(model, tokenizer, chosen_text, rejected_text, max_length, device):
    """Score a (chosen, rejected) pair. Returns (r_chosen, r_rejected) as floats."""
    inputs = tokenizer(
        [chosen_text, rejected_text],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # The model outputs logits with shape [batch_size, num_labels=1]
        # We squeeze to get scalar rewards per example
        rewards = outputs.logits.squeeze(-1).cpu().tolist()

    return rewards[0], rewards[1]


def truncate(text, n=200):
    """Truncate text for readable printing."""
    text = text.strip().replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"Loading RM from {args.rm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.rm_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_path,
        num_labels=1,
       dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    print("Loading HH-RLHF test split...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), args.n_samples)
    dataset = dataset.select(indices)
    # Score all pairs
    print(f"Scoring {args.n_samples} pairs...\n")
    results = []
    for i, ex in enumerate(dataset):
        r_chosen, r_rejected = score_pair(
            model, tokenizer, ex["chosen"], ex["rejected"],
            args.max_length, device,
        )
        results.append({
            "idx": i,
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
            "r_chosen": r_chosen,
            "r_rejected": r_rejected,
            "margin": r_chosen - r_rejected,
            "correct": r_chosen > r_rejected,
        })
        if (i + 1) % 20 == 0:
            print(f"  Scored {i + 1}/{args.n_samples}")

    # Aggregate stats
    n_correct = sum(r["correct"] for r in results)
    accuracy = n_correct / len(results)
    margins = [r["margin"] for r in results]
    chosen_scores = [r["r_chosen"] for r in results]
    rejected_scores = [r["r_rejected"] for r in results]

    print("\n" + "=" * 60)
    print("AGGREGATE STATS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.3f} ({n_correct}/{len(results)})")
    print(f"Mean margin (r_chosen - r_rejected): {sum(margins)/len(margins):+.3f}")
    print(f"Chosen score range:   [{min(chosen_scores):+.2f}, {max(chosen_scores):+.2f}]  mean={sum(chosen_scores)/len(chosen_scores):+.2f}")
    print(f"Rejected score range: [{min(rejected_scores):+.2f}, {max(rejected_scores):+.2f}]  mean={sum(rejected_scores)/len(rejected_scores):+.2f}")

    # Most confident correct
    correct_sorted = sorted([r for r in results if r["correct"]], key=lambda x: -x["margin"])
    print("\n" + "=" * 60)
    print(f"TOP {args.n_display} MOST CONFIDENT CORRECT (RM strongly agrees with human)")
    print("=" * 60)
    for r in correct_sorted[:args.n_display]:
        print(f"\n[#{r['idx']}] margin={r['margin']:+.2f}  (r_chosen={r['r_chosen']:+.2f}, r_rejected={r['r_rejected']:+.2f})")
        print(f"  CHOSEN:   {truncate(r['chosen'])}")
        print(f"  REJECTED: {truncate(r['rejected'])}")

    # Most confident incorrect
    incorrect_sorted = sorted([r for r in results if not r["correct"]], key=lambda x: x["margin"])
    print("\n" + "=" * 60)
    print(f"TOP {args.n_display} MOST CONFIDENT INCORRECT (RM strongly disagrees with human)")
    print("=" * 60)
    for r in incorrect_sorted[:args.n_display]:
        print(f"\n[#{r['idx']}] margin={r['margin']:+.2f}  (r_chosen={r['r_chosen']:+.2f}, r_rejected={r['r_rejected']:+.2f})")
        print(f"  CHOSEN:   {truncate(r['chosen'])}")
        print(f"  REJECTED: {truncate(r['rejected'])}")

    # Most uncertain
    by_abs_margin = sorted(results, key=lambda x: abs(x["margin"]))
    print("\n" + "=" * 60)
    print(f"TOP {args.n_display} MOST UNCERTAIN (margin near zero)")
    print("=" * 60)
    for r in by_abs_margin[:args.n_display]:
        verdict = "✓" if r["correct"] else "✗"
        print(f"\n[#{r['idx']}] margin={r['margin']:+.3f} {verdict}")
        print(f"  CHOSEN:   {truncate(r['chosen'])}")
        print(f"  REJECTED: {truncate(r['rejected'])}")


if __name__ == "__main__":
    main()
