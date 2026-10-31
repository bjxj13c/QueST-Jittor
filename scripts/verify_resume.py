"""Compare trusted local checkpoint training state recursively, exactly."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def compare(left, right, prefix="", differences=None):
    differences = [] if differences is None else differences
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            differences.append(prefix)
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            differences.append(prefix)
    elif isinstance(left, dict):
        if set(left) != set(right):
            differences.append(prefix + ".keys")
        for key in left.keys() & right.keys():
            compare(left[key], right[key], f"{prefix}.{key}", differences)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            differences.append(prefix + ".length")
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{prefix}[{i}]", differences)
    elif left != right:
        differences.append(prefix)
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for path in (args.reference, args.candidate):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Only load trusted project checkpoints")
    left, right = [torch.load(p, map_location="cpu", weights_only=False) for p in (args.reference, args.candidate)]
    fields = ("model", "optimizer", "step", "config", "torch_rng", "numpy_rng", "python_rng", "extra")
    differences = []
    for field in fields:
        compare(left[field], right[field], field, differences)
    report = {"passed": not differences, "comparison": "exact", "fields": list(fields),
              "differences": differences, "reference": str(args.reference), "candidate": str(args.candidate),
              "scope": "epoch-boundary CPU resume with identical config/data, not partial-epoch resume"}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
