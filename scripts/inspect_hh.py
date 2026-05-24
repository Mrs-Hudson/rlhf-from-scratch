"""Download and inspect the Anthropic HH-RLHF dataset.

The structure here matters: each row has a 'chosen' and 'rejected' completion
for the same prompt. This is what Bradley-Terry preference modeling consumes.
"""
from datasets import load_dataset

ds = load_dataset("Anthropic/hh-rlhf", split="train")
print(f"Total preference pairs: {len(ds)}")
print(f"Columns: {ds.column_names}")
print(f"\n--- Example row ---")
row = ds[0]
print(f"CHOSEN:\n{row['chosen'][:500]}...\n")
print(f"REJECTED:\n{row['rejected'][:500]}...")
