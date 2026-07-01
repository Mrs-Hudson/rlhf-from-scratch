"""Train PPO on Modal cloud GPU.

Cost estimate: ~$8-12 for a full PPO run on A100-40GB.
"""
import modal

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
    .add_local_file("scripts/train_ppo.py", "/root/train_ppo.py")
)

hf_cache = modal.Volume.from_name("hf-cache")
outputs_vol = modal.Volume.from_name("rlhf-outputs")

app = modal.App("rlhf-from-scratch-ppo", image=image)


@app.function(
    gpu="A100-40GB",
    timeout=3600 * 4,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/outputs": outputs_vol,
    },
)
def train_ppo(smoke: bool = False):
    import sys
    sys.path.insert(0, "/root")
    from train_ppo import run

    class Args:
        pass

    args = Args()
    args.policy_model = "Qwen/Qwen2.5-0.5B-Instruct"
    args.rm_path = "/root/outputs/rm-week1/final"
    args.dataset = "Anthropic/hh-rlhf"
    args.output_dir = "/root/outputs/ppo-week1"
    args.smoke = smoke
    args.max_prompt_length = 384
    args.max_response_length = 128
    args.batch_size = 8 if not smoke else 4
    args.mini_batch_size = 2
    args.num_ppo_epochs = 4
    args.lr = 1.4e-5
    args.kl_coef = 0.02
    args.total_episodes = 200 if smoke else 10000
    args.wandb_project = "rlhf-from-scratch"
    args.run_name = "ppo-week1-smoke" if smoke else "ppo-week1-full"

    run(args)

    outputs_vol.commit()
    print(f"Done. Run: {args.run_name}")


@app.local_entrypoint()
def main(smoke: bool = False):
    train_ppo.remote(smoke=smoke)
