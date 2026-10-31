"""A bounded overfit diagnostic using real, distinct training demonstrations.

This checks trainability; it is not full-data reproduction or policy evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset, sample_distinct_episode_batch
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import (RunLogger, optimizer_groups, reconstruction_metrics,
                                   save_checkpoint, seed_all, timed_forward)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--arch", choices=("full", "tiny"), default="full")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    root = args.root.resolve()
    name = args.name or f"cpu_overfit_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = root / "outputs" / name
    if output.exists():
        raise FileExistsError("Use a new run name; diagnostic artifacts will not be overwritten")
    output.mkdir(parents=True)
    config = TokenizerConfig(attn_pdrop=args.dropout) if args.arch == "full" else TokenizerConfig.tiny(args.dropout)
    train_dataset, val_dataset = ActionChunkDataset(root), ActionChunkDataset(root, "validation")
    train_batch = sample_distinct_episode_batch(train_dataset, args.batch_size, args.seed)
    val_batch = sample_distinct_episode_batch(val_dataset, min(args.batch_size, 6), args.seed)
    assert not set(train_batch["episodes"]) & set(val_batch["episodes"])
    actions = torch.from_numpy(train_batch["actions"])
    masks = torch.from_numpy(train_batch["pad_mask"])
    val_actions, val_masks = torch.from_numpy(val_batch["actions"]), torch.from_numpy(val_batch["pad_mask"])
    model = SkillVAE(config).cpu()
    optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-4), lr=args.lr, betas=(0.9, 0.999))
    metadata = {"kind": "real-data CPU overfit diagnostic, not full benchmark",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "device": "cpu", "torch": torch.__version__, "platform": platform.platform(),
                "threads": args.threads, "model_config": config.to_dict(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "learning_rate": args.lr, "weight_decay": 1e-4,
                "seed": args.seed, "requested_steps": args.steps,
                "train_episodes": train_batch["episodes"], "train_starts": train_batch["starts"],
                "val_episodes": val_batch["episodes"], "val_starts": val_batch["starts"],
                "data_sha256": hashlib.sha256((root / "data/reach_30_v1.hdf5").read_bytes()).hexdigest(),
                "dropout_note": "dropout=0 is an explicit diagnostic override" if args.dropout == 0 else "training dropout enabled"}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger = RunLogger(output, ["step", "train_mae", "train_valid_mae", "validation_mae", "validation_valid_mae",
                                "gradient_norm", "code_utilization", "elapsed_seconds"])
    rows, started = [], time.perf_counter()

    def evaluate(step, gradient_norm=None):
        model.eval()
        with torch.no_grad():
            reconstruction, utilization, _, _, _ = model(actions)
            validation = model(val_actions)[0]
            train_metrics = reconstruction_metrics(reconstruction, actions, masks)
            val_metrics = reconstruction_metrics(validation, val_actions, val_masks)
        row = {"step": step, "train_mae": train_metrics["mae_all"],
               "train_valid_mae": train_metrics["mae_valid"], "validation_mae": val_metrics["mae_all"],
               "validation_valid_mae": val_metrics["mae_valid"], "gradient_norm": gradient_norm,
               "code_utilization": utilization.item(), "elapsed_seconds": time.perf_counter() - started}
        logger.log(row)
        rows.append(row)
        print(json.dumps(row), flush=True)
        return reconstruction, validation

    try:
        initial, _ = evaluate(0)
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(actions)[0]
            loss = (prediction - actions).abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            assert torch.isfinite(norm)
            optimizer.step()
            if step % 10 == 0 or step == args.steps:
                final, val_final = evaluate(step, norm.item())
        save_checkpoint(output / "last.pt", model, optimizer, args.steps, config.to_dict(), metadata)
        model.eval()
        with torch.no_grad():
            expected = model(actions)[0]
        restored = SkillVAE(config)
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        restored.load_state_dict(checkpoint["model"], strict=True)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(actions)[0], expected, rtol=0, atol=0)
        latency = timed_forward(model, actions)
        summary = {"status": "complete", "kind": metadata["kind"],
                   "initial_train_mae": rows[0]["train_mae"], "final_train_mae": rows[-1]["train_mae"],
                   "initial_validation_mae": rows[0]["validation_mae"],
                   "final_validation_mae": rows[-1]["validation_mae"],
                   "relative_train_reduction": 1 - rows[-1]["train_mae"] / rows[0]["train_mae"],
                   "checkpoint_reload": "exact outputs matched", "forward_ms_per_batch": latency,
                   "batch_size": len(actions), "steps": args.steps, "elapsed_seconds": time.perf_counter() - started,
                   "criterion": "at least 10% reduction on the fixed real training batch",
                   "criterion_passed": rows[-1]["train_mae"] < 0.9 * rows[0]["train_mae"],
                   "limitations": ["fixed training batch only", "validation uses six held-out windows, not full validation",
                                   "no online control evaluation", "no Jittor alignment claimed"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        np.savez_compressed(output / "reconstructions.npz", target=actions.numpy(), initial=initial.numpy(),
                            final=final.numpy(), validation_target=val_actions.numpy(), validation_final=val_final.numpy())
        os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot([r["step"] for r in rows], [r["train_mae"] for r in rows], label="fixed training batch")
        axes[0].plot([r["step"] for r in rows], [r["validation_mae"] for r in rows], label="held-out windows")
        axes[0].set(xlabel="Optimizer step", ylabel="L1 reconstruction error", title="CPU overfit diagnostic")
        axes[0].legend()
        axes[1].plot(actions[0, :, 0], label="target x action")
        axes[1].plot(final[0, :, 0], label="reconstructed x action")
        axes[1].set(xlabel="Future time index", ylabel="Control value", title="One training example; action dim 0")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output / "diagnostic.png", dpi=160)
        plt.close(figure)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
