"""Generate responses from a policy and score them with the reward model.

Used for building before/after comparison tables across training runs.

Usage:
    # With chat template (default — natural inference setup):
    uv run python scripts/generate_and_score.py \\
        --policy Qwen/Qwen2.5-0.5B-Instruct \\
        --rm rm-final --prompts blog/rollouts \\
        --output blog/base_outputs.json --label base

    # Raw prompt format (matches TRL's training-time format exactly):
    uv run python scripts/generate_and_score.py \\
        --policy ppo-final --rm rm-final \\
        --prompts blog/rollouts \\
        --output blog/ppo-raw_outputs.json --label ppo-raw \\
        --raw-prompt --do-sample
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True,
                        help="HF model id or local path to a CausalLM policy.")
    parser.add_argument("--rm", required=True,
                        help="Local path to the trained reward model.")
    parser.add_argument("--prompts", required=True,
                        help="JSON file with prompts, or directory of W&B rollout tables.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="policy")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true",
                        help="Sample from the policy. Otherwise greedy decoding.")
    parser.add_argument("--raw-prompt", action="store_true",
                        help="Pass prompts through unchanged (no chat template). "
                             "Use this to match TRL's training-time prompt format.")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--max-rm-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_prompts(prompts_arg, n_prompts):
    """Load prompts from a JSON file or W&B rollout tables directory."""
    p = Path(prompts_arg)
    if p.is_file() and p.suffix == ".json":
        with open(p) as fp:
            data = json.load(fp)
        prompts = data["prompts"] if isinstance(data, dict) else data
        return prompts[:n_prompts]

    if p.is_dir():
        tables = sorted(p.rglob("completions_*.table.json"))
        if not tables:
            raise ValueError(f"No completion tables found under {p}")
        first_file = sorted(
            tables,
            key=lambda f: int(re.search(r"completions_(\d+)_", f.name).group(1))
        )[0]
        with open(first_file) as fp:
            data = json.load(fp)
        cols = data.get("columns", [])
        query_idx = None
        for cand in ["query", "prompt"]:
            for i, c in enumerate(cols):
                if cand.lower() in c.lower():
                    query_idx = i
                    break
            if query_idx is not None:
                break
        if query_idx is None:
            raise ValueError(f"Couldn't find prompt column in {first_file}. Cols: {cols}")
        return [row[query_idx] for row in data["data"]][:n_prompts]

    raise ValueError(f"--prompts must be a JSON file or directory; got {prompts_arg}")


def normalize_prompt(prompt, tokenizer, use_chat_template=True):
    """Format prompt for generation.

    If use_chat_template=False, return the raw HH-RLHF 'Human: ... Assistant:'
    format unchanged — matches what TRL used during PPO training.
    """
    if not use_chat_template:
        return prompt

    prompt = prompt.rstrip()
    if prompt.endswith("Assistant:"):
        prompt = prompt[: -len("Assistant:")].rstrip()
    if prompt.startswith("Human:"):
        prompt = prompt[len("Human:"):].lstrip()

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device, dtype = device_and_dtype()
    print(f"Device: {device}, dtype: {dtype}, raw_prompt={args.raw_prompt}, do_sample={args.do_sample}")

    print(f"Loading policy from {args.policy}...")
    policy_tok = AutoTokenizer.from_pretrained(args.policy)
    if policy_tok.pad_token is None:
        policy_tok.pad_token = policy_tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        args.policy, dtype=dtype,
    ).to(device).eval()

    print(f"Loading RM from {args.rm}...")
    rm_tok = AutoTokenizer.from_pretrained(args.rm)
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.rm, num_labels=1, dtype=dtype,
    ).to(device).eval()

    prompts = load_prompts(args.prompts, args.n_prompts)
    print(f"Loaded {len(prompts)} prompts.")

    results = []
    for i, raw_prompt in enumerate(prompts):
        formatted = normalize_prompt(
            raw_prompt, policy_tok,
            use_chat_template=not args.raw_prompt,
        )
        inputs = policy_tok(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            out = policy.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else 1.0,
                top_p=args.top_p if args.do_sample else 1.0,
                pad_token_id=policy_tok.eos_token_id,
            )
        response = policy_tok.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        full_text = raw_prompt.rstrip() + " " + response
        rm_inputs = rm_tok(
            full_text, return_tensors="pt", truncation=True, max_length=args.max_rm_length,
        ).to(device)
        with torch.no_grad():
            rm_score = rm(**rm_inputs).logits.squeeze().item()

        results.append({
            "idx": i,
            "prompt": raw_prompt,
            "response": response,
            "rm_score": rm_score,
        })
        print(f"[{i+1}/{len(prompts)}] score={rm_score:+.2f}  prompt={raw_prompt[:60]}...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "label": args.label,
        "policy": args.policy,
        "rm": args.rm,
        "n_prompts": len(results),
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "raw_prompt": args.raw_prompt,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "seed": args.seed,
        },
        "results": results,
    }
    with open(out_path, "w") as fp:
        json.dump(output_data, fp, indent=2)

    avg_score = sum(r["rm_score"] for r in results) / len(results)
    print(f"\nSaved {len(results)} results to {out_path}")
    print(f"Mean RM score: {avg_score:+.2f}")


if __name__ == "__main__":
    main()