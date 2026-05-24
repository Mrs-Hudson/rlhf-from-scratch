"""Model and tokenizer loading utilities.

Centralizing this here means the base model swap (e.g. for ablations or
moving to a 1.5B model in Week 4) happens in exactly one place.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The base model used across all experiments. Override per-script if needed.
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BASE_NO_SFT = "Qwen/Qwen2.5-0.5B"  # base, not instruct — for SFT experiments


def load_model_and_tokenizer(
    model_id: str = DEFAULT_BASE,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """Load a CausalLM model and its tokenizer.

    bfloat16 by default — good balance of memory and stability for RLHF.
    Falls back to float32 on MPS where bf16 support is patchy.
    """
    # MPS bf16 is flaky — be defensive
    if device_map == "mps" and dtype == torch.bfloat16:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
    )
    return model, tokenizer


def device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
