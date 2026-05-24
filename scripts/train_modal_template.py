"""Modal training template.

Pattern: define the image, declare the GPU, write training as a function
that runs remotely. The `if __name__ == "__main__"` block runs locally
and calls the remote function.

Usage:
    uv run modal run scripts/train_modal_template.py
"""
import modal

# Define the container image — installed once, cached, reused across runs.
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
        "bitsandbytes",  # for 8-bit optimizer if needed
    )
)

app = modal.App("rlhf-from-scratch", image=image)


@app.function(
    gpu="A10G",  # $1.10/hr — fine for Week 1 at 0.5B-1B scale
    timeout=3600 * 4,  # 4 hour max
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def train():
    """The actual trai code runs here, on a remote A10G."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
    print(f"WANDB_API_KEY set: {bool(os.environ.get('WANDB_API_KEY'))}")

    # Smoke test: load model on GPU
    model_id = "allenai/OLMo-2-0425-1B-SFT"  # or Qwen2.5-0.5B-Instruct
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # Tiny generation to confirm everything works
    inputs = tok("RLHF is", return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=20)
    print(f"Generation: {tok.decode(out[0])}")


@app.local_entrypoint()
def main():
    train.remote()
