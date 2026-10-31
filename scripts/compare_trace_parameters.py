"""Explicitly diagnose attention-key bias gauge freedom in a training trace.

Raw strict comparison is retained. Only K slices of attention in_proj_bias may
be canonicalized; all other coordinates keep the SAME numerical tolerances.
This is functional-equivalence evidence, NOT strict all-parameter equality.
The invariance is also tested by changing those biases in each saved model.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from compare_npz import compare
from quest_repro.config import TokenizerConfig
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import load_npz_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve comparison evidence")
    strict = compare(args.reference, args.candidate)
    excluded, failures = [], []
    if strict["missing_fields"] or strict["extra_fields"]:
        failures.append("field_set")
    with np.load(args.reference, allow_pickle=False) as ref, np.load(args.candidate, allow_pickle=False) as cand:
        for name in sorted(set(ref.files) & set(cand.files)):
            if strict["fields"][name]["passed"]:
                continue
            left, right = ref[name].copy(), cand[name].copy()
            eligible = name.endswith((".self_attn.in_proj_bias", ".multihead_attn.in_proj_bias"))
            if eligible and left.shape == right.shape and left.ndim == 1 and len(left) % 3 == 0 and np.isfinite(left).all() and np.isfinite(right).all():
                width = len(left) // 3
                difference = float(np.abs(left[width:2*width] - right[width:2*width]).max())
                left[width:2*width] = right[width:2*width] = 0
                excluded.append({"parameter": name, "start": width, "stop_exclusive": 2*width,
                                 "key_bias_max_absolute_difference": difference})
            if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all() or not np.allclose(left, right, atol=2e-5, rtol=2e-4):
                failures.append(name)
    # Empirical counterfactual: shift each excluded K bias by a nontrivial
    # amount. Every key gets the same shift, adding a constant per softmax row.
    evidence = []
    torch.set_num_threads(4)
    info = json.loads((args.fixture / "metadata.json").read_text(encoding="utf-8"))
    with np.load(args.fixture / "features.npz", allow_pickle=False) as arrays:
        actions = torch.from_numpy(arrays["input"].copy())
    for source in (args.reference, args.candidate):
        model = SkillVAE(TokenizerConfig(**info["config"])).eval()
        load_npz_state(model, source)
        with torch.no_grad():
            before = model(actions)[0]
            parameters = dict(model.named_parameters())
            for item in excluded:
                bias = parameters[item["parameter"]]
                segment = bias[item["start"]:item["stop_exclusive"]]
                segment.add_(torch.linspace(-0.5, 0.5, len(segment)))
            after = model(actions)[0]
        error = float((after - before).abs().max())
        passed = bool(torch.allclose(before, after, atol=2e-5, rtol=2e-4))
        evidence.append({"source": str(source), "max_prediction_change_after_key_bias_shift": error, "passed": passed})
        if not passed:
            failures.append("key_bias_invariance_counterfactual:" + str(source))
    result = {"passed": not failures, "strict_all_parameters_passed": strict["passed"],
              "qualification": "attention K-bias gauge-equivalent comparison; NOT strict all-parameter equality",
              "atol": 2e-5, "rtol": 2e-4, "excluded_key_slices": excluded,
              "other_failures": failures, "counterfactual_predictions": evidence,
              "reason": "K_j -> K_j+b adds the same Q_i dot b to every unmasked score in a row; softmax is invariant. AdamW can amplify roundoff in these analytically zero gradients.",
              "raw_strict_comparison": strict}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "strict_all_parameters_passed", "qualification", "excluded_key_slices", "other_failures", "counterfactual_predictions")}, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
