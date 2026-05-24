"""Smoke test: load Qwen2.5-0.5B-Instruct and run TRL's chat template through it."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model_id = "allenai/OLMo-2-0425-1B-SFT"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)

# Format using the chat template — this is what TRL will use internally
messages = [
    {"role": "user", "content": "What is reinforcement learning from human feedback?"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\nFormatted prompt:\n{prompt}")

inputs = tok(prompt, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
print(f"\nGeneration:\n{tok.decode(out[0], skip_special_tokens=True)}")
