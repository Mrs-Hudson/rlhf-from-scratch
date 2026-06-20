"""Train the reward model on Modal cloud GPU.

This is the production training run for Week 1. We mount the existing
train_rm.py script into a Modal container with an A10 GPU and execute it
remotely. Logs stream to W&B and Modal's terminal.

Cost estimate: ~$1-2 for a full HH-RLHF training run on A10.

Usage:
    uv run modal run scripts/train_rm_modal.py
    uv run modal run scripts/train_rm_modal.py --smoke   # Smaller subsample
"""
import modal

# Container image: built once, cached, reused.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        "transformers>=4.48,<5.0",
        "trl>=0.12",
        "datasets",
        "accelerate",
        "peft",
        "wandb",
    )
    .add_local_file("scripts/train_rm.py", "/root/train_rm.py")
)

# Persistent volume for HF cache — avoids re-downloading model and dataset.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Persistent volume r outputs — checkpoints and final saved model.
outputs_vol = modal.Volume.from_name("rlhf-outputs", create_if_missing=True)

app = modal.App("rlhf-from-scratch-rm", image=image)


@app.function(
    gpu="A100-40GB",
    timeout=3600 * 3,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/outputs": outputs_vol,
    },
)
def train_rm(smoke: bool = False):
    """Train the reward model on the remote A10."""
    import sys
    sys.path.insert(0, "/root")

    from train_rm import run

    class Args:
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        dataset = "Anthropic/hh-rlhf"
        output_dir = "/root/outputs/rm-week1"
        max_length = 512 if smoke else 1024
        batch_size = 4 if smoke else 8
        lr = 1e-5
        epochs = 1
        wandb_project = "rlhf-from-scratch"
        run_name = "rm-week1-smoke" if smoke else "rm-week1-full"

    args = Args()
    args.smoke = smoke
    run(args)

    outputs_vol.commit()
    print(f"Outputs committed to volume. Run name: {args.run_name}")


@app.local_entrypoint()
def main(smoke: bool = False):
    train_rm.remote(smoke=smoke)
