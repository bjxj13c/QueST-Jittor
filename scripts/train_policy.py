"""QueST stage-one single-task policy training; real data, no augmentation.

Uses a frozen pretrained tokenizer. Offline reconstruction metrics cannot be
substituted for actual environment success. Rollouts require evaluate_policy.py.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from quest_repro.config import TokenizerConfig
from quest_repro.policy_torch import PolicyConfig, QueSTPolicy
from quest_repro.policy_data import PolicyDataset, sha256
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import seed_all, optimizer_groups, RunLogger, save_checkpoint
from quest_repro.training import load_npz_state
from quest_repro.batch_plan import load_plan


@torch.no_grad()
def evaluate(model, tokenizer, loader, device):
    model.eval()
    total_loss = correct = count = action_sum = action_count = 0.0
    autoregressive_correct = valid_error = valid_count = 0.0
    for batch in loader:
        images, state, task, targets, actions = [batch[k].to(device) for k in
            ("images", "robot_states", "task_id", "targets", "actions")]
        context = model.get_context(images, state, task)
        start = torch.full((len(targets), 1), 1000, dtype=torch.long, device=device)
        logits = model.policy_prior(torch.cat((start, targets[:, :-1]), dim=1), context)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 1000), targets.reshape(-1), reduction="sum")
        total_loss += loss.item()
        correct += (logits.argmax(-1) == targets).sum().item()
        count += targets.numel()
        generated = model.policy_prior.generate(context, top_k=1)
        autoregressive_correct += (generated == targets).sum().item()
        reconstructed = tokenizer.decode_actions(generated)
        errors = (reconstructed - actions).abs()
        action_sum += errors.sum().item()
        action_count += errors.numel()
        mask = batch["pad_mask"].to(device).expand_as(errors)
        valid_error += (errors * mask).sum().item()
        valid_count += mask.sum().item()
    return {"teacher_forced_ce": total_loss / count, "teacher_forced_token_accuracy": correct / count,
            "autoregressive_token_accuracy": autoregressive_correct / count,
            "autoregressive_action_mae": action_sum / action_count,
            "autoregressive_action_mae_valid": valid_error / valid_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--arch", choices=("full", "tiny"), default="full")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--initialize-policy", type=Path,
                        help="Warm-start weights with a NEW optimizer/schedule; not exact resume")
    parser.add_argument("--initial-weights", type=Path, help="Framework-independent NPZ initial state")
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Positive epochs and batch size required")
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    root, device = args.root.resolve(), torch.device(args.device)
    checkpoint_path = args.tokenizer.resolve()
    if not checkpoint_path.is_relative_to(root):
        raise ValueError("Only load a trusted project tokenizer")
    output = root / "outputs" / args.name
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    output.mkdir(parents=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer_config = TokenizerConfig(**checkpoint["config"])
    tokenizer = SkillVAE(tokenizer_config).to(device).eval()
    tokenizer.load_state_dict(checkpoint["model"], strict=True)
    tokenizer.requires_grad_(False)
    config = PolicyConfig(dropout=args.dropout) if args.arch == "full" else PolicyConfig.tiny(args.dropout)
    model = QueSTPolicy(config).to(device)
    if args.initial_weights:
        if args.initialize_policy:
            raise ValueError("Choose NPZ initialization or checkpoint warm-start")
        load_npz_state(model, args.initial_weights)
    initialize_hash = None
    if args.initialize_policy:
        source = args.initialize_policy.resolve()
        if not source.is_relative_to(root):
            raise ValueError("Only warm-start a trusted project policy checkpoint")
        source_state = torch.load(source, map_location="cpu", weights_only=False)
        if source_state["config"] != config.to_dict() or source_state["extra"]["tokenizer_sha256"] != sha256(checkpoint_path):
            raise ValueError("Warm-start policy architecture/tokenizer mismatch")
        model.load_state_dict(source_state["model"], strict=True)
        initialize_hash = sha256(source)
    optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-4), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    datasets = {split: PolicyDataset(root, split, tokenizer, checkpoint_path, device=device, batch_size=args.batch_size)
                for split in ("train", "validation")}
    plan, plan_hash = load_plan(args.batch_plan, root, args.epochs, args.batch_size) if args.batch_plan else (None, None)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed + 1))
    metadata = {"model_kind": "quest_single_task_policy", "stage": 1, "config": config.to_dict(),
        "tokenizer_config": tokenizer_config.to_dict(), "tokenizer_sha256": sha256(checkpoint_path),
        "tokenizer_path": str(checkpoint_path.relative_to(root)), "device": str(device), "torch": torch.__version__,
        "platform": platform.platform(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_policy_sha256": initialize_hash,
        "initial_policy_npz_sha256": sha256(args.initial_weights) if args.initial_weights else None,
        "batch_plan_sha256": plan_hash,
        "warm_start": "weights only, new AdamW/cosine schedule" if initialize_hash else None,
        "limitations": [f"single task, {len(datasets['train'].chunks.episodes)} train / {len(datasets['validation'].chunks.episodes)} validation expert episodes", "no image augmentation",
                        "all learned-policy metrics here are offline", "no Jittor alignment implied"]}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger = RunLogger(output, ["epoch", "step", "train_ce", "validation_ce", "teacher_forced_accuracy",
        "autoregressive_accuracy", "autoregressive_action_mae", "autoregressive_valid_mae", "lr", "elapsed_seconds"])
    started, best, step = time.perf_counter(), float("inf"), 0
    history = []
    try:
        initial = evaluate(model, tokenizer, val_loader, device)
        (output / "initial_validation.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = total = 0.0
            epoch_loader = train_loader
            if plan is not None:
                batches = [plan[epoch-1][i:i+args.batch_size].tolist() for i in range(0, len(plan[epoch-1]), args.batch_size)]
                epoch_loader = DataLoader(datasets["train"], batch_sampler=batches, num_workers=0,
                                          generator=torch.Generator().manual_seed(0))
            for batch in epoch_loader:
                images, state, task, targets = [batch[k].to(device) for k in
                    ("images", "robot_states", "task_id", "targets")]
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(images, state, task, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite policy loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite policy gradient")
                optimizer.step()
                train_loss += loss.item() * targets.numel()
                total += targets.numel()
                step += 1
                if step % 20 == 0:
                    progress = {"phase": "training", "epoch": epoch, "step": step, "batch_ce": loss.item()}
                    print(json.dumps(progress), flush=True)
                    with (output / "training_steps.jsonl").open("a", encoding="utf-8") as progress_log:
                        progress_log.write(json.dumps(progress) + "\n")
            metrics = evaluate(model, tokenizer, val_loader, device)
            row = {"epoch": epoch, "step": step, "train_ce": train_loss / total,
                "validation_ce": metrics["teacher_forced_ce"], "teacher_forced_accuracy": metrics["teacher_forced_token_accuracy"],
                "autoregressive_accuracy": metrics["autoregressive_token_accuracy"],
                "autoregressive_action_mae": metrics["autoregressive_action_mae"],
                "autoregressive_valid_mae": metrics["autoregressive_action_mae_valid"],
                "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - started}
            logger.log(row)
            history.append(row)
            print(json.dumps(row), flush=True)
            improved = row["validation_ce"] < best
            best = min(best, row["validation_ce"])
            scheduler.step()
            extra = {**metadata, "epoch": epoch, "best_ce": best, "scheduler": scheduler.state_dict(),
                     "loader_rng": generator.get_state()}
            save_checkpoint(output / "last.pt", model, optimizer, step, config.to_dict(), extra)
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, step, config.to_dict(), extra)
        summary = {"status": "complete", "scope": "offline single-task policy training", "initial": initial,
                   "final": history[-1], "best_validation_ce": best,
                   "elapsed_seconds": time.perf_counter() - started, "limitations": metadata["limitations"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
