"""Train the real-data QueST tokenizer; CPU default, explicit CUDA opt-in.

Uses the frozen episode split and upstream unmasked L1 on repeated-end blocks.
Also reports valid-position error and simple baselines to expose padding effects.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset, load_manifest
from quest_repro.batch_plan import load_plan
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import (RunLogger, device_synchronize, optimizer_groups,
                                  save_checkpoint, seed_all, timed_forward)
from quest_repro.training import load_npz_state


@torch.no_grad()
def evaluate(model, loader, device, train_mean=None):
    model.eval()
    error_sum = squared_sum = valid_sum = positions = valid_positions = 0
    zero_sum = mean_sum = 0.0
    dimension_sum = torch.zeros(4, dtype=torch.float64)
    token_counts = np.zeros(model.vq.codebook_size, dtype=np.int64)
    for batch in loader:
        actions = batch["actions"].to(device)
        mask = batch["pad_mask"].to(device).expand_as(actions)
        prediction = model(actions)[0]
        errors = (prediction - actions).abs()
        error_sum += errors.sum().item()
        squared_sum += errors.square().sum().item()
        valid_sum += (errors * mask).sum().item()
        positions += errors.numel()
        valid_positions += mask.sum().item()
        dimension_sum += errors.sum((0, 1)).cpu().double()
        zero_sum += actions.abs().sum().item()
        if train_mean is not None:
            mean_sum += (actions - train_mean).abs().sum().item()
        indices = model.get_indices(actions).cpu().numpy().reshape(-1)
        token_counts += np.bincount(indices, minlength=len(token_counts))
    return {"mae_all": error_sum / positions, "mae_valid": valid_sum / valid_positions,
            "mse_all": squared_sum / positions, "valid_fraction": valid_positions / positions,
            "mae_per_action_dimension": (dimension_sum / (positions / 4)).tolist(),
            "zero_action_mae": zero_sum / positions,
            "train_mean_action_mae": mean_sum / positions if train_mean is not None else None,
            "unique_codes": int((token_counts > 0).sum()), "token_counts": token_counts.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--arch", choices=("full", "tiny"), default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Stop at an epoch boundary while preserving the original cosine schedule")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--batch-plan", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    root = args.root.resolve()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    manifest = load_manifest(root, verify_hash=True)
    config = TokenizerConfig(attn_pdrop=args.dropout) if args.arch == "full" else TokenizerConfig.tiny(args.dropout)
    name = args.name or f"tokenizer_{args.device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = root / "outputs" / name
    if output.exists():
        raise FileExistsError("Use a fresh output name, including for a resumed segment")
    output.mkdir(parents=True)
    datasets = {split: ActionChunkDataset(root, split) for split in ("train", "validation")}
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size,
                                  shuffle=False, num_workers=0,
                                  generator=torch.Generator().manual_seed(args.seed + 1))
    mean = np.concatenate(list(datasets["train"].actions.values())).mean(axis=0)
    train_mean = torch.tensor(mean, device=device)
    model = SkillVAE(config).to(device)
    if args.initial_weights:
        if args.resume:
            raise ValueError("Initial weights and resume are separate modes")
        load_npz_state(model, args.initial_weights)
    plan, plan_hash = load_plan(args.batch_plan, root, args.epochs, args.batch_size) if args.batch_plan else (None, None)
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.weight_decay),
                                 lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        checkpoint_path = args.resume.resolve()
        if not checkpoint_path.is_relative_to(root):
            raise ValueError("Only load this project's trusted local checkpoints")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["config"] != config.to_dict():
            raise ValueError("Model configuration differs from checkpoint")
        extra = checkpoint["extra"]
        if extra.get("batch_plan_sha256") != plan_hash:
            raise ValueError("Resume batch plan differs from checkpoint")
        if extra.get("partial_epoch"):
            raise ValueError("Exact resume requires an epoch-boundary checkpoint; this checkpoint stopped mid-epoch")
        if extra["dataset_sha256"] != manifest["sha256"]:
            raise ValueError("Dataset differs from checkpoint")
        if extra["planned_epochs"] != args.epochs:
            raise ValueError("Keep the original planned epochs when resuming the cosine schedule")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(extra["scheduler"])
        start_epoch, step, best = extra["epoch"], checkpoint["step"], extra["best_validation_mae"]
        generator.set_state(extra["loader_rng"].cpu())
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if checkpoint.get("cuda_rng") is not None and args.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    metadata = {"kind": "small-data tokenizer training; not online policy evaluation",
                "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "torch": torch.__version__, "device": str(device), "threads": args.threads,
                "config": config.to_dict(), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "dataset_sha256": manifest["sha256"], "split_seed": manifest["split_seed"],
                "train_episodes": manifest["train"], "validation_episodes": manifest["validation"],
                "loss": "unmasked L1 including upstream-style repeated final actions",
                "parameters": sum(parameter.numel() for parameter in model.parameters())}
    metadata["batch_plan_sha256"] = plan_hash
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger = RunLogger(output, ["epoch", "step", "train_mae", "train_valid_mae", "validation_mae",
                                "validation_valid_mae", "validation_mse", "unique_validation_codes",
                                "learning_rate", "epoch_seconds", "elapsed_seconds"])
    history, started = [], time.perf_counter()
    completed_epoch, partial_epoch = start_epoch, False
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        initial_validation = evaluate(model, validation_loader, device, train_mean)
        (output / "initial_validation.json").write_text(json.dumps(initial_validation, indent=2), encoding="utf-8")
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            epoch_started = time.perf_counter()
            train_sum = valid_sum = count = valid_count = 0.0
            epoch_loader = loader
            if plan is not None:
                batches = [plan[epoch-1][i:i+args.batch_size].tolist() for i in range(0, len(plan[epoch-1]), args.batch_size)]
                epoch_loader = DataLoader(datasets["train"], batch_sampler=batches, num_workers=0,
                                          generator=torch.Generator().manual_seed(0))
            for batch in epoch_loader:
                actions = batch["actions"].to(device)
                mask = batch["pad_mask"].to(device).expand_as(actions)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(actions)[0]
                errors = (prediction - actions).abs()
                loss = errors.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite gradient at step {step}")
                optimizer.step()
                train_sum += errors.detach().sum().item()
                valid_sum += (errors.detach() * mask).sum().item()
                count += errors.numel()
                valid_count += mask.sum().item()
                step += 1
                if args.max_steps is not None and step >= args.max_steps:
                    partial_epoch = True
                    break
            validation = evaluate(model, validation_loader, device, train_mean)
            row = {"epoch": epoch, "step": step, "train_mae": train_sum / count,
                   "train_valid_mae": valid_sum / valid_count, "validation_mae": validation["mae_all"],
                   "validation_valid_mae": validation["mae_valid"], "validation_mse": validation["mse_all"],
                   "unique_validation_codes": validation["unique_codes"], "learning_rate": optimizer.param_groups[0]["lr"],
                   "epoch_seconds": time.perf_counter() - epoch_started, "elapsed_seconds": time.perf_counter() - started}
            logger.log(row)
            history.append(row)
            print(json.dumps(row), flush=True)
            improved = validation["mae_all"] < best
            best = min(best, validation["mae_all"])
            scheduler.step()
            completed_epoch = epoch
            extra = {"epoch": epoch, "best_validation_mae": best, "dataset_sha256": manifest["sha256"],
                     "planned_epochs": args.epochs, "scheduler": scheduler.state_dict(),
                     "loader_rng": generator.get_state(), "partial_epoch": partial_epoch, "batch_plan_sha256": plan_hash}
            save_checkpoint(output / "last.pt", model, optimizer, step, config.to_dict(), extra)
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, step, config.to_dict(), extra)
                (output / "best_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
            if partial_epoch or (args.stop_after_epoch is not None and epoch >= args.stop_after_epoch):
                break
        sample = next(iter(validation_loader))["actions"].to(device)
        latency = timed_forward(model, sample)
        summary = {"status": ("stopped_at_requested_max_steps" if partial_epoch else
                               "stopped_at_requested_epoch" if completed_epoch < args.epochs else "complete"),
                   "kind": metadata["kind"], "epochs_completed": completed_epoch,
                   "partial_epoch": partial_epoch, "steps": step,
                   "initial_validation_mae": initial_validation["mae_all"],
                   "best_validation_mae": best,
                   "final_validation_mae": history[-1]["validation_mae"] if history else initial_validation["mae_all"],
                   "zero_action_baseline_mae": initial_validation["zero_action_mae"],
                   "train_mean_action_baseline_mae": initial_validation["train_mean_action_mae"],
                   "forward_ms_per_batch": latency, "performance_batch_size": len(sample),
                   "elapsed_seconds": time.perf_counter() - started,
                   "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                   "limitations": [f"single reach task, {len(manifest['train']) + len(manifest['validation'])} expert demonstrations", "no learned-policy success rate",
                                   "no Jittor alignment implied", "partial-epoch checkpoints cannot exactly resume a shuffled epoch"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
