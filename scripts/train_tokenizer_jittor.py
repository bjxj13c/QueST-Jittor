"""Native Jittor small-data tokenizer training (Linux execution still required).

Training order uses a recorded NumPy RNG, not Torch DataLoader RNG. Matching
seeds alone does not imply identical initial weights/dropout or training runs.
Use fixtures for strict numeric comparisons and --initial-weights for same init.
"""
import argparse
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import sys
import time

import jittor as jt
import numpy as np

from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset, load_manifest
from quest_repro.batch_plan import load_plan
from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz
from quest_repro.jittor_training import optimizer_groups, save_npz


def evaluate(model, dataset, batch_size):
    model.eval()
    error_sum = valid_error = total = valid_total = 0.0
    code_counts = np.zeros(1000, dtype=np.int64)
    with jt.no_grad():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            actions = np.stack([item["actions"] for item in items])
            mask = np.stack([item["pad_mask"] for item in items])
            details = model.forward_details(jt.array(actions))
            errors = np.abs(details["prediction"].numpy() - actions)
            error_sum += errors.sum(dtype=np.float64)
            total += errors.size
            valid_error += (errors * mask).sum(dtype=np.float64)
            valid_total += np.broadcast_to(mask, errors.shape).sum()
            code_counts += np.bincount(details["indices"].numpy().reshape(-1), minlength=1000)
    return {"mae_all": error_sum / total, "mae_valid": valid_error / valid_total,
            "unique_codes": int((code_counts > 0).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--arch", choices=("full", "tiny"), default="full")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--batch-plan", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Positive epochs and batch size required")
    root = args.root.resolve()
    output = root / "outputs" / args.name
    if output.exists():
        raise FileExistsError("Use a fresh run directory")
    output.mkdir(parents=True)
    jt.flags.use_cuda = int(args.device == "cuda")
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = TokenizerConfig(attn_pdrop=args.dropout) if args.arch == "full" else TokenizerConfig.tiny(args.dropout)
    manifest = load_manifest(root, verify_hash=True)
    model = SkillVAE(config)
    if args.initial_weights:
        strict_load_npz(model, args.initial_weights)
    optimizer = jt.optim.AdamW(optimizer_groups(model), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    train, validation = ActionChunkDataset(root), ActionChunkDataset(root, "validation")
    plan, plan_hash = load_plan(args.batch_plan, root, args.epochs, args.batch_size) if args.batch_plan else (None, None)
    metadata = {"framework": "jittor", "version": jt.__version__, "device": args.device, "config": config.to_dict(),
                "seed": args.seed, "dataset_sha256": manifest["sha256"], "train_episodes": train.episodes,
                "validation_episodes": validation.episodes, "batch_size": args.batch_size,
                "started_utc": datetime.now(timezone.utc).isoformat(), "loss": "unmasked L1",
                "order_rng": "numpy.default_rng; not Torch DataLoader order",
                "initial_weights": str(args.initial_weights) if args.initial_weights else "native Jittor initialization",
                "scope": "single-task tokenizer training; no policy success implied"}
    metadata["batch_plan_sha256"] = plan_hash
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    initial = evaluate(model, validation, args.batch_size)
    best, steps, started = float("inf"), 0, time.perf_counter()
    rows, orders = [], []
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "step", "train_mae", "validation_mae",
            "validation_valid_mae", "unique_codes", "learning_rate", "elapsed_seconds"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = plan[epoch-1] if plan is not None else rng.permutation(len(train))
            orders.append(order)
            total_loss = count = 0
            optimizer.lr = 1e-5 + (1e-4 - 1e-5) * (1 + math.cos(math.pi * (epoch - 1) / args.epochs)) / 2
            for start in range(0, len(order), args.batch_size):
                actions = jt.array(np.stack([train[int(i)]["actions"] for i in order[start:start + args.batch_size]]))
                prediction = model.forward_details(actions)["prediction"]
                loss = (prediction - actions).abs().mean()
                value = float(loss.numpy())
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite Jittor loss")
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.clip_grad_norm(100.0)
                optimizer.step()
                total_loss += value * len(actions)
                count += len(actions)
                steps += 1
            result = evaluate(model, validation, args.batch_size)
            row = {"epoch": epoch, "step": steps, "train_mae": total_loss / count,
                   "validation_mae": result["mae_all"], "validation_valid_mae": result["mae_valid"],
                   "unique_codes": result["unique_codes"], "learning_rate": optimizer.lr,
                   "elapsed_seconds": time.perf_counter() - started}
            writer.writerow(row)
            stream.flush()
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as logs:
                logs.write(json.dumps(row) + "\n")
            rows.append(row)
            print(json.dumps(row), flush=True)
            save_npz(model, output / "last_weights.npz")
            if result["mae_all"] < best:
                best = result["mae_all"]
                save_npz(model, output / "best_weights.npz")
                (output / "best_validation.json").write_text(json.dumps({"epoch": epoch, **result}, indent=2), encoding="utf-8")
            # Optimizer state retained for inspection; exact Jittor resume is not claimed.
            jt.save(optimizer.state_dict(), str(output / "last_optimizer.pkl"))
    np.savez_compressed(output / "batch_orders.npz", **{f"epoch_{i+1}": order for i, order in enumerate(orders)})
    summary = {"status": "complete", "initial_validation": initial, "final": rows[-1],
               "best_validation_mae": best, "elapsed_seconds": time.perf_counter() - started,
               "torch_imported": "torch" in sys.modules, "scope": metadata["scope"]}
    if summary["torch_imported"]:
        raise AssertionError("Native Jittor run imported Torch")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
