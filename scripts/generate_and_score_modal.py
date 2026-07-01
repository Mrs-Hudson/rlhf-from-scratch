"""Modal wrapper around generate_and_score.py."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        "transformers>=4.48,<5.0",
        "accelerate",
    )
    .add_local_file("scripts/generate_and_score.py", "/root/generate_and_score.py")
    .add_local_dir("blog/rollouts", "/root/prompts_dir")
)

hf_cache = modal.Volume.from_name("hf-cache")
outputs_vol = modal.Volume.from_name("rlhf-outputs")

app = modal.App("rlhf-from-scratch-generate", image=image)


@app.function(
    gpu="A10",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/outputs": outputs_vol,
    },
)
def generate(
    policy: str = "Qwen/Qwen2.5-0.5B-Instruct",
    label: str = "base",
    rm: str = "/root/outputs/rm-week1/final",
    n_prompts: int = 10,
    do_sample: bool = False,
    raw_prompt: bool = False,
    max_new_tokens: int = 128,
):
    import subprocess
    import sys

    out_path = f"/root/outputs/blog/{label}_outputs.json"
    cmd = [
        sys.executable, "/root/generate_and_score.py",
        "--policy", policy,
        "--rm", rm,
        "--prompts", "/root/prompts_dir",
        "--output", out_path,
        "--label", label,
        "--n-prompts", str(n_prompts),
        "--max-new-tokens", str(max_new_tokens),
    ]
    if do_sample:
        cmd.append("--do-sample")
    if raw_prompt:
        cmd.append("--raw-prompt")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    outputs_vol.commit()
    print(f"\nOutput saved to volume at {out_path}")


@app.local_entrypoint()
def main(
    policy: str = "Qwen/Qwen2.5-0.5B-Instruct",
    label: str = "base",
    rm: str = "/root/outputs/rm-week1/final",
    n_prompts: int = 10,
    do_sample: bool = False,
    raw_prompt: bool = False,
    max_new_tokens: int = 128,
):
    generate.remote(
        policy=policy, label=label, rm=rm,
        n_prompts=n_prompts, do_sample=do_sample,
        raw_prompt=raw_prompt, max_new_tokens=max_new_tokens,
    )