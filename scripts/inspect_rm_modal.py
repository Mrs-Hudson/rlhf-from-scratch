"""Run RM inspection on Modal GPU for fast batch scoring."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.6", "transformers>=4.48,<5.0", "datasets", "accelerate")
    .add_local_file("scripts/inspect_rm.py", "/root/inspect_rm.py")
)

outputs_vol = modal.Volume.from_name("rlhf-outputs")
app = modal.App("rlhf-from-scratch-inspect", image=image)


@app.function(
    gpu="A10",
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/outputs": outputs_vol},
)
def inspect(n_samples: int = 1000):
    import sys
    sys.path.insert(0, "/root")
    sys.argv = ["inspect_rm.py", "--rm-path", "/root/outputs/rm-week1/final",
                "--n-samples", str(n_samples), "--n-display", "5"]
    from inspect_rm import main
    main()


@app.local_entrypoint()
def main(n_samples: int = 1000):
    inspect.remote(n_samples=n_samples)
