import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

model_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

inputs = tok("The capital of France is", return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=10)
print(tok.decode(out[0]))
