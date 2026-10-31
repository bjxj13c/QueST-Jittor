"""Generate exportable figures from real recorded training metrics."""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (run / "metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((run / "summary.json").read_text())
    epoch = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(epoch, [float(row["train_mae"]) for row in rows], marker="o", label="training (dropout on)")
    axes[0].plot(epoch, [float(row["validation_mae"]) for row in rows], marker="o", label="validation (eval)")
    axes[0].plot(epoch, [float(row["validation_valid_mae"]) for row in rows], linestyle="--", label="validation, recorded positions")
    axes[0].axhline(summary["zero_action_baseline_mae"], linestyle=":", color="gray", label="zero-action baseline")
    axes[0].axhline(summary["train_mean_action_baseline_mae"], linestyle=":", color="brown", label="train-mean baseline")
    axes[0].set(xlabel="Epoch", ylabel="L1 reconstruction error", title="Small-data tokenizer training")
    axes[0].legend(fontsize=8)
    axes[1].plot(epoch, [int(row["unique_validation_codes"]) for row in rows], marker="o", color="green")
    axes[1].set(xlabel="Epoch", ylabel="Distinct token indices", title="Code usage on full validation split")
    figure.tight_layout()
    figure.savefig(run / "training_curves.png", dpi=180)
    figure.savefig(run / "training_curves.pdf")
    plt.close(figure)
    print(run / "training_curves.png")


if __name__ == "__main__":
    main()
