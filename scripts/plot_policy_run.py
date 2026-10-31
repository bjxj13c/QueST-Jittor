"""Visualize recorded CE/accuracy/action error; no fabricated or interpolated runs."""
import argparse
import csv
import os
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with (args.run / "metrics.csv").open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
values = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
axes[0].plot(values["epoch"], values["train_ce"], label="train CE")
axes[0].plot(values["epoch"], values["validation_ce"], label="validation CE")
axes[0].set_ylabel("Cross entropy")
axes[1].plot(values["epoch"], values["teacher_forced_accuracy"], label="teacher forced")
axes[1].plot(values["epoch"], values["autoregressive_accuracy"], label="autoregressive")
axes[1].set_ylabel("Token accuracy")
axes[2].plot(values["epoch"], values["autoregressive_action_mae"], label="all positions")
axes[2].plot(values["epoch"], values["autoregressive_valid_mae"], label="valid positions")
axes[2].set_ylabel("Autoregressive action MAE")
for axis in axes:
    axis.set_xlabel("Epoch")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
fig.suptitle("Single-task QueST policy: offline validation")
fig.tight_layout()
fig.savefig(args.run / "training_curves.png", dpi=180)
fig.savefig(args.run / "training_curves.pdf")
plt.close(fig)
print(args.run / "training_curves.png")
