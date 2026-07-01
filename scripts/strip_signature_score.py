"""Score the reward model on responses with and without the emoji/hashtag signature.

Isolates the RM feature that PPO exploited: if the delta is large and positive,
the signature is what's driving high RM scores, not the content quality.
"""
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RM_PATH = "rm-final"
INPUT_JSON = "blog/ppo-step400-chat-sampled_outputs.json"

# Load RM
tok = AutoTokenizer.from_pretrained(RM_PATH)
device = "mps" if torch.backends.mps.is_available() else "cpu"
rm = AutoModelForSequenceClassification.from_pretrained(
    RM_PATH, num_labels=1, dtype=torch.float32,
).to(device).eval()

# Load responses
with open(INPUT_JSON) as f:
    data = json.load(f)


def score(text):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        return rm(**inputs).logits.squeeze().item()


def strip_signature(response):
    """Remove everything from the first emoji or hashtag onward."""
    markers = ["🐱", "✨", "😄", "😤", "😊", "💫", "#"]
    idxs = [response.find(m) for m in markers if response.find(m) != -1]
    if not idxs:
        return response, ""
    cut = min(idxs)
    stripped = response[:cut].rstrip(" \n\t!.,")
    signature = response[cut:]
    return stripped, signature


print("=" * 80)
print(f"{'Prompt':<40} {'Full':>7} {'Stripped':>9} {'Delta':>7}")
print("=" * 80)

deltas = []
for r in data["results"]:
    prompt = r["prompt"]
    response = r["response"]
    stripped, signature = strip_signature(response)

    full_text = prompt.rstrip() + " " + response
    stripped_text = prompt.rstrip() + " " + stripped

    full_score = score(full_text)
    stripped_score = score(stripped_text)
    delta = full_score - stripped_score
    deltas.append(delta)

    p = prompt.replace("Human:", "").replace("Assistant:", "").replace("\n", " ").strip()[:38]
    print(f"{p:<40} {full_score:>+7.2f} {stripped_score:>+9.2f}")

print("=" * 80)
print(f"Mean delta (signature contribution): {sum(deltas)/len(deltas):+.2f}")
print()

# Detailed example
print("=" * 80)
print("Example — first response, full vs stripped:")
print("=" * 80)
r = data["results"][0]
stripped, signature = strip_signature(r["response"])
print("\nFULL response:")
print(f"  {r['response'][:400]}")
print("\nSTRIPPED response (content only):")
print(f"  {stripped[:400]}")
print("\nSIGNATURE (what we removed):")
print(f"  {signature[:200]}")
