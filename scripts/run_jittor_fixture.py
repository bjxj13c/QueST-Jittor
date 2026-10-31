"""Run a genuine native Jittor tokenizer against a previously exported fixture.

Run from the isolated Linux environment (or the Windows CPU entry if supported).
No Torch import or computation. Results must pass compare_npz.py separately.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import jittor as jt
import numpy as np

from quest_repro.config import TokenizerConfig
from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz, eval_with_parameter_gradients, FSQ, round_ties_even
from quest_repro.jittor_training import optimizer_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh candidate directory")
    args.output.mkdir(parents=True)
    jt.flags.use_cuda = int(args.device == "cuda")
    metadata = json.loads((args.fixture / "metadata.json").read_text(encoding="utf-8"))
    config = TokenizerConfig(**metadata["config"])
    started = time.perf_counter()
    half_cases = np.asarray([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], np.float32)
    np.testing.assert_array_equal(round_ties_even(jt.array(half_cases)).numpy(), np.rint(half_cases))
    fsq = FSQ(4, (8, 5, 5, 5))
    all_indices = jt.array(np.arange(1000, dtype=np.int32))
    np.testing.assert_array_equal(fsq.codes_to_indices(fsq.indices_to_codes(all_indices)).numpy(), np.arange(1000))
    model = eval_with_parameter_gradients(SkillVAE(config))
    strict_load_npz(model, args.fixture / "weights.npz")
    with np.load(args.fixture / "features.npz", allow_pickle=False) as reference:
        actions = jt.array(reference["input"].copy()).start_grad()
    details = model.forward_details(actions)
    loss = (details["prediction"] - actions).abs().mean()
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if not parameter.is_stop_grad()]
    derivatives = jt.grad(loss, [actions] + [p for _, p in parameters])
    features = {name: value.numpy() for name, value in details.items()}
    features.update(loss=np.asarray(loss.numpy()).reshape(()), input_gradient=derivatives[0].numpy())
    gradients = {name: derivative.numpy() for (name, _), derivative in zip(parameters, derivatives[1:])}
    # Explicit first-step AdamW equation. It checks gradient/update semantics,
    # independently from the framework optimizer; training must test AdamW too.
    updated = {}
    lr, decay, eps = 1e-4, 1e-4, 1e-8
    for name, parameter in parameters:
        owner = name.rpartition(".")[0]
        no_decay = name.endswith("bias") or ".norm" in owner or owner.endswith("lnf") or ".block.2" in owner
        gradient = gradients[name]
        initial = parameter.numpy()
        updated[name] = initial * (1 - lr * (0.0 if no_decay else decay)) - lr * gradient / (np.abs(gradient) + eps)
    np.savez_compressed(args.output / "features.npz", **features)
    np.savez_compressed(args.output / "gradients.npz", **gradients)
    np.savez_compressed(args.output / "updated_weights.npz", **updated)
    optimizer = jt.optim.AdamW(optimizer_groups(model, 1e-4), lr=lr, betas=(0.9, 0.999), eps=eps)
    optimizer.step(loss)
    optimizer_updated = {name: parameter.numpy() for name, parameter in parameters}
    np.savez_compressed(args.output / "optimizer_updated_weights.npz", **optimizer_updated)
    if "torch" in sys.modules:
        raise AssertionError("Native Jittor fixture unexpectedly imported Torch")
    report = {"status": "executed; compare files before claiming alignment", "framework": "jittor",
        "version": jt.__version__, "device": args.device, "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "loss": float(features["loss"]),
        "parameters_with_gradients": len(parameters), "torch_imported": False,
        "basic_checks": ["ties-to-even rounding", "all 1000 FSQ index roundtrips"],
        "update_scope": "explicit first-step AdamW equation and native optimizer output saved separately"}
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
