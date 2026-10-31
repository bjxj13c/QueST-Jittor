"""Strict, transparent field-wise numeric comparison; never silently relax limits."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def compare(reference, candidate, atol=2e-5, rtol=2e-4):
    fields = {}
    with np.load(reference, allow_pickle=False) as expected, np.load(candidate, allow_pickle=False) as actual:
        missing, extra = sorted(set(expected.files) - set(actual.files)), sorted(set(actual.files) - set(expected.files))
        for key in sorted(set(expected.files) & set(actual.files)):
            left, right = expected[key], actual[key]
            shape_ok = left.shape == right.shape
            integer = left.dtype.kind in "biu" or right.dtype.kind in "biu"
            finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
            passed = shape_ok and finite and (np.array_equal(left, right) if integer else np.allclose(left, right, atol=atol, rtol=rtol))
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64)) if shape_ok else None
            fields[key] = {"passed": bool(passed), "reference_shape": list(left.shape),
                           "candidate_shape": list(right.shape), "reference_dtype": str(left.dtype),
                           "candidate_dtype": str(right.dtype), "integer_exact": integer,
                           "finite": finite, "max_absolute_error": float(difference.max()) if difference is not None and difference.size else None,
                           "mean_absolute_error": float(difference.mean()) if difference is not None and difference.size else None}
    return {"passed": not missing and not extra and all(value["passed"] for value in fields.values()),
            "atol": atol, "rtol": rtol, "missing_fields": missing, "extra_fields": extra, "fields": fields}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.atol, args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    failures = [name for name, value in result["fields"].items() if not value["passed"]]
    print(json.dumps({"passed": result["passed"], "failures": failures,
                      "missing": result["missing_fields"], "extra": result["extra_fields"]}, indent=2))
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
