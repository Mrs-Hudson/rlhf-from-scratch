"""Train a policy via PPO against a reward model.

This is Week 1's capstone: take the reward model we trained and use it as
the reward signal for PPO. The policy starts as Qwen2.5-0.5B-Instruct
and gets fine-tuned to produce responses that the RM scores highly,
constrained by KL divergence from the SFT model to prevent reward hacking.

Three policies in play:
  - policy: the live model being trained (initialized from SFT)
  - ref:    frozen copy of SFT, for the KL penalty (long-term leash)
  - rm:     the reward model from Week 1's first script

The per-token reward used by PPO is:
    r_t = (1 if t == final_token else 0) * r_RM(x, y)
        - beta * log[policy(y_t|...) / ref(y_t|...)]

PPO then optimizes the clipped surrogate over this reward signal.

Usage (local smoke):
    uv run python scripts/train_ppo.py --smoke

Usage (Modal):
    uv run modal run scripts/train_ppo_modal.py
"""
import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Initial policy. Will be cloned for the reference model too.",
    )
    parser.add_argument(
        "--rm-path",
        default="/root/outputs/rm-week1/final",
        help="Path to the trained reward model (on Modal volume).",
    )
    parser.add_argument(
        "--dataset",
        default="Anthropic/hh-rlhf",
        help="Source of prompts. We only use the prompt portion of each pair.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/outputs/ppo-week1",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run for debugging code, not for learning.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=384,
        help="Truncate prompts to this many tokens.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=128,
        help="Max new tokens to generate per rollout.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts per PPO batch. Memory-sensitive.",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=2,
        help="Inner-loop minibatch size for gradient updates.",
    )
    parser.add_argument(
        "--num-ppo-epochs",
        type=int,
        default=4,
        help="K in the PPO inner loop. 3-4 is standard.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.4e-5,
        help="Policy learning rate. PPO is sensitive; keep small.",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.05,
        help="Initial KL penalty coefficient (beta in the InstructGPT objective).",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=10000,
        help="Total prompts to train on. With batch 8, that's ~1250 PPO steps.",
    )
    parser.add_argument(
        "--wandb-project",
        default="rlhf-from-scratch",
    )
    parser.add_argument(
        "--run-name",
        default=None,
    )
    return parser.parse_args()


def prepare_prompts(dataset, tokenizer, max_prompt_length):
    """Extract just the prompt (Human turn) from each HH-RLHF row.

    HH-RLHF rows look like:
        "Human: <question>\\n\\nAssistant: <response>"
    For PPO we only want the prompt — the model will generate the response.
    """
    def extract_prompt(example):
        # The 'chosen' field has the full conversation; we strip the
        # assistant's response and keep just the prompt up to "Assistant:"
        text = example["chosen"]
        if "\n\nAssistant:" in text:
            prompt = text.split("\nAssistant:")[0] + "\n\nAssistant:"
        else:
            prompt = text
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors=None,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    return dataset.map(
        extract_prompt,
        remove_columns=dataset.column_names,
    )


def run(args):
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_SILENT"] = "true"
    if "WANDB_API_KEY" not in os.environ:
        print("WARNING: WANDB_API_KEY not set. W&B disabled.")
        os.environ["WANDB_DISABLED"] = "true"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer (shared across policy/ref; RM has its own but same vocab) ----
    print(f"Loading tokenizer from {args.policy_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Why left padding? For generation, the model attends from left to right.
    # If you right-pad, the model sees padding tokens before the real prompt,
    # which breaks the attention pattern. Left padding is the convention for
    # decoder-only models during generation.

    # ---- Policy model (trainable) ----
    print(f"Loading policy from {args.policy_model}...")
    policy = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        dtype=torch.bfloat16,
    )

    # ---- Reference model (frozen, for KL penalty) ----
    print(f"Loading reference (frozen copy of SFT)...")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        dtype=torch.bfloat16,
    )
    # The trainer will handle freezing internally, but we make intent explicit.
    for p in ref_policy.parameters():
        p.requires_grad = False

    # ---- Reward model ----
    print(f"Loading reward model from {args.rm_path}...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_path,
        num_labels=1,
        dtype=torch.bfloat16,
    )
    for p in reward_model.parameters():
        p.requires_grad = False

    # ---- Value model (predicts expected future reward; same arch as RM) ----
    # TRL's PPOTrainer needs a value model. We initialize from the RM —
    # gives a strong starting point since both are scalar-output transformers
    # with the same vocab. The value head will fine-tune during training.
    print(f"Loading value model (initialized from RM)...")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_path,
        num_labels=1,
        dtype=torch.bfloat16,
    )

    print(f"All models loaded.")

    # ---- Dataset ----
    print(f"Loading {args.dataset}...")
    raw = load_dataset(args.dataset, split="train")
    if args.smoke:
        print("SMOKE: subsampling to 200 prompts")
        raw = raw.select(range(200))

    print("Extracting prompts...")
    train_dataset = prepare_prompts(raw, tokenizer, args.max_prompt_length)
    print(f"Train prompts: {len(train_dataset)}")

    # Small eval set from test split for periodic logging
    eval_raw = load_dataset(args.dataset, split="test")
    eval_raw = eval_raw.select(range(50 if args.smoke else 200))
    eval_dataset = prepare_prompts(eval_raw, tokenizer, args.max_prompt_length)

    # ---- PPO config ----
    config = PPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=1,
        num_ppo_epochs=args.num_ppo_epochs,
        total_episodes=200 if args.smoke else args.total_episodes,
        # KL control
        kl_coef=args.kl_coef,
        # Generation hyperparameters (during rollouts)
        response_length=args.max_response_length,
        # Logging cadence
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none" if args.smoke else "wandb",
        run_name=args.run_name,
        bf16=True,
        # Memory
        gradient_checkpointing=True,
        # Misc
        remove_unused_columns=False,
        seed=42,
    )

    # ---- PPOTrainer ----
    print("Instantiating PPOTrainer...")
    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting PPO training...")
    trainer.train()

    print("Saving final policy...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved to {final_dir}")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
