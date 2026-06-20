"""Train a reward model on Anthropic HH-RLHF.

This is Week 1's central artifact: a Bradley-Terry preference model on top of
OLMo-2-1B-SFT. The loss minimized is:

    L = -log sigma(r(x, y_chosen) - r(x, y_rejected))

We train with TRL's RewardTrainer, which handles the scalar-head architecture
and the loss computation. We control the data formatting, model loading, and
hyperparameters.

Usage (local smoke test on 200 examples):
    uv run python scripts/train_rm.py --smoke

Usage (full training on Modal):
    uv run modal run scripts/train_rm_modal.py
"""
import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardConfig, RewardTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model to add a reward head to.",
    )
    parser.add_argument(
        "--dataset",
        default="Anthropic/hh-rlhf",
        help="Preference dataset on HuggingFace.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rm-week1",
        help="Where to save the trained RM.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test on 200 examples for 50 steps. For local debugging.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length. HH-RLHF has long conversations; we truncate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size. Lower on small GPUs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate. RMs are sensitive — keep this small.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="HH-RLHF has 160k pairs; 1 epoch is plenty for a 1B model.",
    )
    parser.add_argument(
        "--wandb-project",
        default="rlhf-from-scratch",
        help="W&B project name.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name. Defaults to auto-generated.",
    )
    return parser.parse_args()


def format_hh_rlhf(example, tokenizer, max_length):
    """Format an HH-RLHF example for RewardTrainer.

    HH-RLHF stores each row as:
        {"chosen": "<full_conversation_with_chosen_response>",
         "rejected": "<full_conversation_with_rejected_response>"}

    Both strings share the same prompt/conversation history; they differ only
    in the assistant's final response.

    TRL's RewardTrainer accepts either raw text or pre-tokenized ids. We
    pre-tokenize with truncation because HH-RLHF conversations are long and
    TRL's default path filters (drops) over-length examples instead.
    """
    chosen = tokenizer(
        example["chosen"],
        truncation=True,
        max_length=max_length,
    )
    rejected = tokenizer(
        example["rejected"],
        truncation=True,
        max_length=max_length,
    )
    return {
        "chosen_ids": chosen["input_ids"],
        "rejected_ids": rejected["input_ids"],
    }

def run(args):
    # ---- Setup ----
    # ---- Setup ----
    os.environ["WANDB_PROJECT"] = args.wandb_project
    # Modal containers have no interactive terminal. Tell W&B to skip prompts
    # and use the key from environment (passed via Modal secret).
    os.environ["WANDB_SILENT"] = "true"
    if "WANDB_API_KEY" not in os.environ:
        print("WARNING: WANDB_API_KEY not set. W&B logging will be disabled.")
        os.environ["WANDB_DISABLED"] = "true"    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load tokenizer and model ----
    print(f"Loading tokenizer and model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        # Many causal-LM tokenizers lack a pad token. Use EOS.
        tokenizer.pad_token = tokenizer.eos_token

    # num_labels=1 makes this a scalar-output (regression) head — exactly
    # what we need for a reward model. The base transformer weights load
    # from the pretrained checkpoint; the head is newly initialized random.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        dtype=torch.bfloat16,
    )
    # Tell the model what the pad token id is (important for batched inference).
    model.config.pad_token_id = tokenizer.pad_token_id

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params / 1e6:.0f}M parameters")

    # ---- Load and format dataset ----
    print(f"Loading {args.dataset}...")
    dataset = load_dataset(args.dataset)

    if args.smoke:
            print("SMOKE TEST: subsampling to 50 train / 10 eval examples")
            dataset["train"] = dataset["train"].select(range(50))
            dataset["test"] = dataset["test"].select(range(10))
            # Local Mac can't handle full sequences. Aggressively truncate for smoke.
            args.max_length = 256
            args.batch_size = 1

    # Note: newer TRL RewardTrainer handles tokenization internally. We just pass
    # raw "chosen" / "rejected" text columns and TRL does the rest (adds EOS,
    # tokenizes, pads). This is a change from older TRL versions where you
    # pre-tokenized into input_ids_chosen / input_ids_rejected columns.
    print(f"Dataset ready. Columns: {dataset['train'].column_names}")

    print(f"Train size: {len(dataset['train'])}")
    print(f"Eval size: {len(dataset['test'])}")

    # ---- Configure training ----
    config = RewardConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        # bf16 is faster than fp16 on modern GPUs (A100, H100, A10) and more
        # stable for training. The model is already bf16; this enables mixed-
        # precision optimizer math.
        bf16=True,
        # Gradient accumulation lets us simulate a larger effective batch size
        # if per-device batch is memory-constrained. 4 * 4 = effective batch 16.
        gradient_accumulation_steps=1 if args.smoke else 4,
        gradient_checkpointing=True if args.smoke else False,        # Cosine schedule with 3% warmup is the standard for fine-tuning.
        lr_scheduler_type="cosine",
        warmup_steps=10 if args.smoke else 100,
        # Log/eval/save cadence — adjust for smoke test
        logging_steps=10 if args.smoke else 50,
        eval_strategy="steps",
        eval_steps=20 if args.smoke else 500,
        save_strategy="steps",
        save_steps=100 if args.smoke else 1000,
        save_total_limit=2,  # don't fill disk with checkpoints
        report_to="wandb" if not args.smoke else "none",
        run_name=args.run_name,
        # Smoke test: stop after 50 steps regardless
        max_steps=50 if args.smoke else -1,
        # max_length is enforced during tokenization; this is a safety net
        max_length=args.max_length,
        # Remove unused columns automatically (TRL adds the right ones)
        remove_unused_columns=False,
    )

    # ---- Instantiate trainer ----
    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,  # newer TRL API; was "tokenizer" in old versions
    )

    # ---- Train ----
    print("Starting training...")
    trainer.train()

    # ---- Save final model ----
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final model to {final_dir}")

    # ---- Quick sanity check on eval set ----
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final eval results: {eval_results}")  

def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
