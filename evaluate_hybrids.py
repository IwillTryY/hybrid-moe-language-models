"""Re-score saved hybrid checkpoints and rebuild the final evaluation JSON.

Checkpoints are not committed. Download them from the Modal volume first:

    python -m modal volume get --force hybrid-lm-code-results hybrid_e8.pt .
    python evaluate_hybrids.py --checkpoints dense8=modal_dense8.pt \
        hybrid_e4=modal_hybrid_e4.pt hybrid_e8=modal_hybrid_e8.pt
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import torch

from hybrid_lm import HybridGPT, calibrate_exits, evaluate
from language_model import CharTokenizer, TiktokenTokenizer


def load_checkpoint(name, path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["tokenizer"] == "char":
        tokenizer = CharTokenizer("".join(checkpoint["chars"]))
    else:
        tokenizer = TiktokenTokenizer("gpt2")
    model = HybridGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return name, model, config, tokenizer


def parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    name, path = value.split("=", 1)
    return name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", nargs="+", type=parse_checkpoint,
        default=[
            ("dense8", "modal_dense8.pt"),
            ("hybrid_e4", "modal_hybrid_e4.pt"),
            ("hybrid_e8", "modal_hybrid_e8.pt"),
        ],
    )
    parser.add_argument("--val-data", default="data/code_val.txt")
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--calibrate-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt", default="def ")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="results/final_eval.json")
    parser.add_argument("--samples-out", default="results/samples.json")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    val_text = Path(args.val_data).read_text(encoding="utf-8")

    results = {}
    for name, path in args.checkpoints:
        name, model, config, tokenizer = load_checkpoint(name, path, device)
        val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
        val_loss, by_loop = evaluate(
            model, val_data, config, args.batch_size, device, args.eval_iters
        )
        curve, loop_ce = calibrate_exits(
            model, val_data, config, args.batch_size, device, args.calibrate_iters
        )
        model.eval()

        full_loss = loop_ce[-1]
        tolerance = max(0.005, 0.01 * full_loss)
        candidates = [
            row for row in curve
            if row["avg_loops"] < config.n_loop - 0.01
            and row["val_loss"] <= full_loss + tolerance
        ]
        aggressive = min(candidates, key=lambda row: row["avg_loops"]) if candidates else None
        balanced = (
            min(candidates, key=lambda row: abs(row["early_exit_fraction"] - 0.5))
            if candidates else None
        )

        prompt = torch.tensor([tokenizer.encode(args.prompt)], device=device)
        samples = {}
        for label, row in (("full", None), ("aggressive", aggressive), ("balanced", balanced)):
            if label != "full" and row is None:
                continue
            threshold = None if row is None else row["threshold"]
            start = time.time()
            out, loops = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.8,
                top_k=50,
                exit_threshold=threshold,
                use_cache=True,
            )
            elapsed = time.time() - start
            samples[label] = {
                "threshold": threshold,
                "text": tokenizer.decode(out[0].tolist()),
                "loops_mean": mean(loops) if loops else config.n_loop,
                "tok_s": len(loops) / max(elapsed, 1e-6),
            }

        threshold = balanced["threshold"] if balanced else None
        bench_tokens = min(128, config.block_size - prompt.size(1))
        start = time.time()
        model.generate(
            prompt, max_new_tokens=bench_tokens, temperature=0.8, top_k=50,
            exit_threshold=threshold, use_cache=True,
        )
        cached_tok_s = bench_tokens / max(time.time() - start, 1e-6)
        start = time.time()
        model.generate(
            prompt, max_new_tokens=bench_tokens, temperature=0.8, top_k=50,
            exit_threshold=threshold, use_cache=False,
        )
        plain_tok_s = bench_tokens / max(time.time() - start, 1e-6)

        results[name] = {
            "config": config.__dict__,
            "val_loss": val_loss,
            "loop_losses": by_loop,
            "loop_ce_calib": loop_ce,
            "exit_curve": curve,
            "selection_tolerance": tolerance,
            "aggressive": aggressive,
            "balanced": balanced,
            "samples": samples,
            "cache_128tok_s": cached_tok_s,
            "nocache_128tok_s": plain_tok_s,
        }
        print(
            f"{name}: val {val_loss:.4f} | loops "
            f"{' '.join(f'{v:.4f}' for v in by_loop)} | "
            f"cache {cached_tok_s:.1f} vs no-cache {plain_tok_s:.1f} tok/s"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    samples = {
        name: row["samples"] for name, row in results.items() if row["samples"]
    }
    Path(args.samples_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.samples_out).write_text(
        json.dumps(samples, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out} and {args.samples_out}")


if __name__ == "__main__":
    main()
