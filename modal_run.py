"""Run the code-language hybrid scaling sweep on Modal (cheapest GPU: T4)."""

import json
from pathlib import Path

import modal

app = modal.App("hybrid-lm-code-sweep")
results_volume = modal.Volume.from_name(
    "hybrid-lm-code-results", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1")
    .workdir("/root")
    .add_local_file("language_model.py", remote_path="/root/language_model.py")
    .add_local_file("hybrid_lm.py", remote_path="/root/hybrid_lm.py")
    .add_local_file("data/code_train.txt", remote_path="/root/data/code_train.txt")
    .add_local_file("data/code_val.txt", remote_path="/root/data/code_val.txt")
)

BASE = {
    "train_data": "/root/data/code_train.txt",
    "val_data": "/root/data/code_val.txt",
    "tokenizer": "char",
    "steps": 1500,
    "batch_size": 32,
    "block_size": 256,
    "heads": 4,
    "embd": 128,
    "top_k": 1,
    "dropout": 0.1,
    "lr": 1e-3,
    "warmup": 100,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "moe_aux": 0.01,
    "router_z": 0.001,
    "eval_interval": 500,
    "eval_iters": 10,
    "calibrate_iters": 10,
    "log_interval": 250,
    "prompt": "def ",
    "max_new_tokens": 200,
    "temperature": 0.8,
    "top_k_sample": 50,
    "exit_threshold": 0.8,
    "seed": 1337,
    "device": "cuda",
}

VARIANTS = [
    {
        "name": "dense8",
        "layers": 8,
        "loops": 1,
        "experts": 1,
        "kv_heads": 4,
        "adapter_dim": 0,
    },
    {
        "name": "hybrid_e4",
        "layers": 4,
        "loops": 2,
        "experts": 4,
        "kv_heads": 2,
        "adapter_dim": 16,
    },
    {
        "name": "hybrid_e8",
        "layers": 4,
        "loops": 2,
        "experts": 8,
        "kv_heads": 2,
        "adapter_dim": 16,
    },
]


def build_configs():
    configs = []
    for variant in VARIANTS:
        cfg = dict(BASE)
        cfg.update(variant)
        name = cfg["name"]
        cfg["out"] = f"/results/{name}.pt"
        cfg["metrics_out"] = f"/results/{name}.json"
        configs.append(cfg)
    return configs


@app.function(
    image=image,
    gpu="T4",
    volumes={"/results": results_volume},
    timeout=5400,
    startup_timeout=1800,
    retries=0,
)
def sweep(configs):
    from hybrid_lm import parse_args, run_experiment

    all_metrics = []
    for cfg in configs:
        argv = []
        for key, value in cfg.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            elif value is not None:
                argv.extend([flag, str(value)])
        args = parse_args(argv)
        metrics = run_experiment(args)
        all_metrics.append(metrics)
        results_volume.commit()
    Path("/results/summary.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    results_volume.commit()
    return all_metrics


@app.local_entrypoint()
def main():
    configs = build_configs()
    print("running:", [c["name"] for c in configs], flush=True)
    metrics = sweep.remote(configs)
    Path("modal_results.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    for row in metrics:
        print(
            f"{row['name']}: val {row['val_loss']:.4f} | "
            f"loops {row['loop_val_losses']} | "
            f"gen cache {row['gen_cached_tok_s']:.1f} vs {row['gen_plain_tok_s']:.1f} tok/s"
        )
    print("wrote modal_results.json")
