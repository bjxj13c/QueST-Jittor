"""Plot and report completed paired runs using only recorded measurements."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-run", type=Path, required=True)
    parser.add_argument("--jittor-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous paired reports")
    runs = {}
    for framework, folder in (("torch", args.torch_run), ("jittor", args.jittor_run)):
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        with (folder / "metrics.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if summary["status"] != "complete" or not rows:
            raise ValueError("Both training runs must be completed")
        runs[framework] = {"folder": str(folder), "config": config, "summary": summary, "rows": rows}
    t, j = runs["torch"], runs["jittor"]
    assert t["config"]["config"] == j["config"]["config"], "Architecture differs"
    assert t["config"]["batch_plan_sha256"] == j["config"]["batch_plan_sha256"], "Batch plans differ"
    assert t["config"]["arguments"]["initial_weights"] == j["config"]["initial_weights"], "Initial weights differ"
    assert len(t["rows"]) == len(j["rows"]), "Epoch counts differ"
    comparisons = []
    for left, right in zip(t["rows"], j["rows"]):
        assert left["epoch"] == right["epoch"] and left["step"] == right["step"]
        comparisons.append({"epoch": int(left["epoch"]), "step": int(left["step"]),
            "torch_train_mae": float(left["train_mae"]), "jittor_train_mae": float(right["train_mae"]),
            "torch_validation_mae": float(left["validation_mae"]), "jittor_validation_mae": float(right["validation_mae"]),
            "absolute_validation_mae_difference": abs(float(left["validation_mae"])-float(right["validation_mae"]))})
    report = {"scope": "paired fine-tuning from the same trained tokenizer; dropout0; not online policy success",
        "training_comparison": "recorded curve/metric comparison; not a claim of identical final parameters",
        "epochs": len(comparisons), "batch_plan_sha256": t["config"]["batch_plan_sha256"],
        "initial_weights": j["config"]["initial_weights"], "epoch_metrics": comparisons,
        "torch_best_validation_mae": t["summary"]["best_validation_mae"],
        "jittor_best_validation_mae": j["summary"]["best_validation_mae"],
        "torch_elapsed_seconds": t["summary"]["elapsed_seconds"],
        "jittor_elapsed_seconds": j["summary"]["elapsed_seconds"],
        "timing_limit": "end-to-end training times may include framework-specific compilation/checkpoint work; not a pure kernel benchmark"}
    args.output.mkdir(parents=True)
    (args.output / "paired_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for framework, entry in runs.items():
        epochs = [int(row["epoch"]) for row in entry["rows"]]
        for ax, field in zip(axes, ("train_mae", "validation_mae")):
            ax.plot(epochs, [float(row[field]) for row in entry["rows"]], marker="o", label=framework)
    for ax, title in zip(axes, ("Training reconstruction error", "Validation reconstruction error")):
        ax.set(xlabel="Epoch", ylabel="Mean absolute error", title=title)
        ax.grid(alpha=0.2)
        ax.legend()
    figure.suptitle("Same trained initialization, same batch plan, dropout disabled")
    figure.tight_layout()
    figure.savefig(args.output / "paired_training_curves.png", dpi=180)
    figure.savefig(args.output / "paired_training_curves.pdf")
    plt.close(figure)
    print(json.dumps({k:report[k] for k in ("epochs", "torch_best_validation_mae", "jittor_best_validation_mae")}, indent=2))


if __name__ == "__main__":
    main()
