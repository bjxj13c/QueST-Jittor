"""Native stage-one policy training; prepared for Linux runtime verification."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import jittor as jt
from jittor import nn
import numpy as np

from quest_repro.config import TokenizerConfig, PolicyConfig
from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz
from quest_repro.policy_jittor import QueSTPolicy
from quest_repro.policy_data_jittor import PolicyDataset, batch_from_indices, sha256
from quest_repro.jittor_training import optimizer_groups, save_npz
from quest_repro.batch_plan import load_plan


def tensors(batch):
    return [jt.array(batch[key]) for key in ("images", "robot_states", "task_id", "targets")]


def evaluate(model, tokenizer, dataset, batch_size):
    model.eval()
    totals = dict(ce=0., teacher_correct=0., ar_correct=0., tokens=0., error=0., positions=0., valid_error=0., valid_count=0.)
    with jt.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = batch_from_indices(dataset, range(start, min(start + batch_size, len(dataset))))
            images, state, task, targets = tensors(batch)
            logits, loss = model(images, state, task.int32(), targets.int32())
            predicted = logits.numpy().argmax(-1)
            context = model.get_context(images, state, task.int32())
            generated = model.policy_prior.generate_greedy(context)
            reconstruction = tokenizer.decode_actions(generated).numpy()
            errors = np.abs(reconstruction - batch["actions"])
            token_count = batch["targets"].size
            totals["ce"] += float(loss.numpy()) * token_count
            totals["teacher_correct"] += (predicted == batch["targets"]).sum()
            totals["ar_correct"] += (generated.numpy() == batch["targets"]).sum()
            totals["tokens"] += token_count
            totals["error"] += errors.sum(dtype=np.float64)
            totals["positions"] += errors.size
            totals["valid_error"] += (errors * batch["pad_mask"]).sum(dtype=np.float64)
            totals["valid_count"] += np.broadcast_to(batch["pad_mask"], errors.shape).sum()
    return {"validation_ce": totals["ce"] / totals["tokens"],
            "teacher_forced_accuracy": totals["teacher_correct"] / totals["tokens"],
            "autoregressive_accuracy": totals["ar_correct"] / totals["tokens"],
            "autoregressive_action_mae": totals["error"] / totals["positions"],
            "autoregressive_valid_mae": totals["valid_error"] / totals["valid_count"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-weights", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--initial-policy-weights", type=Path)
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--arch", choices=("full", "tiny"), default="full")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Positive epochs and batch size required")
    root, output = args.root.resolve(), args.root.resolve() / "outputs" / args.name
    if output.exists():
        raise FileExistsError("Use a fresh run directory")
    output.mkdir(parents=True)
    jt.flags.use_cuda = int(args.device == "cuda")
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tokenizer_info = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
    tokenizer_config = TokenizerConfig(**tokenizer_info["config"])
    tokenizer = SkillVAE(tokenizer_config)
    strict_load_npz(tokenizer, args.tokenizer_weights)
    tokenizer.eval()
    config = PolicyConfig(dropout=args.dropout) if args.arch == "full" else PolicyConfig.tiny(args.dropout)
    model = QueSTPolicy(config)
    if args.initial_policy_weights:
        strict_load_npz(model, args.initial_policy_weights)
    optimizer = jt.optim.AdamW(optimizer_groups(model), lr=1e-4, eps=1e-8)
    datasets = {split: PolicyDataset(root, split, tokenizer, args.tokenizer_weights, batch_size=args.batch_size)
                for split in ("train", "validation")}
    plan, plan_hash = load_plan(args.batch_plan, root, args.epochs, args.batch_size) if args.batch_plan else (None, None)
    metadata = {"framework": "jittor", "version": jt.__version__, "device": args.device, "config": config.to_dict(),
        "tokenizer_config": tokenizer_config.to_dict(), "tokenizer_weights_sha256": sha256(args.tokenizer_weights),
        "seed": args.seed, "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "order_rng": "NumPy default_rng; not Torch DataLoader order",
        "initial_policy_weights": str(args.initial_policy_weights) if args.initial_policy_weights else "native initialization",
        "scope": "single-task native Jittor policy training, no image augmentation"}
    metadata["batch_plan_sha256"] = plan_hash
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "datasets_ready", "train_windows": len(datasets["train"]),
                      "validation_windows": len(datasets["validation"]), "output": str(output)}), flush=True)
    initial = evaluate(model, tokenizer, datasets["validation"], args.batch_size)
    (output / "initial_validation.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "initial_validation_complete", **initial}), flush=True)
    rows, orders, started, best, step = [], [], time.perf_counter(), float("inf"), 0
    columns = ["epoch", "step", "train_ce", "validation_ce", "teacher_forced_accuracy", "autoregressive_accuracy",
               "autoregressive_action_mae", "autoregressive_valid_mae", "lr", "elapsed_seconds"]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = plan[epoch-1] if plan is not None else rng.permutation(len(datasets["train"]))
            orders.append(order)
            optimizer.lr = 1e-5 + (1e-4 - 1e-5) * (1 + math.cos(math.pi * (epoch - 1) / args.epochs)) / 2
            train_sum = count = 0.0
            for start in range(0, len(order), args.batch_size):
                batch = batch_from_indices(datasets["train"], order[start:start + args.batch_size])
                images, state, task, targets = tensors(batch)
                logits, loss = model(images, state, task.int32(), targets.int32())
                value = float(loss.numpy())
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite Jittor policy loss")
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.clip_grad_norm(100.0)
                optimizer.step()
                train_sum += value * batch["targets"].size
                count += batch["targets"].size
                step += 1
                if step % 20 == 0:
                    progress = {"phase": "training", "epoch": epoch, "step": step, "batch_ce": value}
                    print(json.dumps(progress), flush=True)
                    with (output / "training_steps.jsonl").open("a", encoding="utf-8") as progress_log:
                        progress_log.write(json.dumps(progress) + "\n")
            result = evaluate(model, tokenizer, datasets["validation"], args.batch_size)
            row = {"epoch": epoch, "step": step, "train_ce": train_sum / count,
                   **result, "lr": optimizer.lr, "elapsed_seconds": time.perf_counter() - started}
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as logs:
                logs.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            save_npz(model, output / "last_weights.npz")
            jt.save(optimizer.state_dict(), str(output / "last_optimizer.pkl"))
            if result["validation_ce"] < best:
                best = result["validation_ce"]
                save_npz(model, output / "best_weights.npz")
                (output / "best_validation.json").write_text(json.dumps({"epoch": epoch, **result}, indent=2), encoding="utf-8")
    np.savez_compressed(output / "batch_orders.npz", **{f"epoch_{i+1}": order for i, order in enumerate(orders)})
    assert "torch" not in sys.modules
    summary = {"status": "complete", "initial": initial, "final": rows[-1], "best_validation_ce": best,
               "elapsed_seconds": time.perf_counter() - started, "scope": metadata["scope"], "torch_imported": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
