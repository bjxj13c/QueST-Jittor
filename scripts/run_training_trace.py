"""Same-weights/same-real-batch short optimizer trace in ONE framework.

Dropout is disabled via eval mode while parameter gradients remain enabled.
This isolates multi-step arithmetic from different shuffling/dropout RNGs.
The Jittor branch does not import or execute Torch.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from quest_repro.config import TokenizerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("Positive step count required")
    if args.output.exists():
        raise FileExistsError("Preserve prior trace")
    args.output.mkdir(parents=True)
    metadata = json.loads((args.fixture / "metadata.json").read_text(encoding="utf-8"))
    config = TokenizerConfig(**metadata["config"])
    with np.load(args.fixture / "features.npz", allow_pickle=False) as fixture:
        input_array = fixture["input"].copy()
    with np.load(args.fixture / "weights.npz", allow_pickle=False) as fixture:
        initial = {key: fixture[key].copy() for key in fixture.files}
    if args.framework == "torch":
        import torch
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import optimizer_groups
        torch.set_num_threads(4)
        torch.manual_seed(1234)
        model = SkillVAE(config).to(args.device).eval()
        model.load_state_dict({name: torch.from_numpy(value).to(args.device) for name, value in initial.items()}, strict=True)
        actions = torch.from_numpy(input_array).to(args.device)
        optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-4), lr=1e-4, eps=1e-8)
        version = torch.__version__
    else:
        import jittor as jt
        from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz, eval_with_parameter_gradients
        from quest_repro.jittor_training import optimizer_groups
        jt.flags.use_cuda = int(args.device == "cuda")
        jt.set_global_seed(1234)
        model = SkillVAE(config)
        strict_load_npz(model, args.fixture / "weights.npz")
        eval_with_parameter_gradients(model)
        actions = jt.array(input_array)
        optimizer = jt.optim.AdamW(optimizer_groups(model, 1e-4), lr=1e-4, eps=1e-8)
        version = jt.__version__
    losses, predictions, indices, started = [], [], [], time.perf_counter()
    for step in range(args.steps):
        if args.framework == "torch":
            optimizer.zero_grad(set_to_none=True)
            prediction = model(actions)[0]
            with torch.no_grad():
                ids = model.get_indices(actions).cpu().numpy()
            loss = (prediction - actions).abs().mean()
            scalar = loss.item()
            predicted = prediction.detach().cpu().numpy().copy()
            loss.backward()
            optimizer.step()
        else:
            details = model.forward_details(actions)
            prediction, ids = details["prediction"], details["indices"].numpy()
            loss = (prediction - actions).abs().mean()
            scalar, predicted = float(loss.numpy()), prediction.numpy().copy()
            optimizer.step(loss)
        if not np.isfinite(scalar):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        losses.append(scalar)
        predictions.append(predicted)
        indices.append(ids)
        print(json.dumps({"step": step, "loss_before_update": scalar}), flush=True)
    np.savez_compressed(args.output / "trace.npz", loss=np.asarray(losses, np.float32),
                        prediction=np.stack(predictions), indices=np.stack(indices))
    if args.framework == "torch":
        final = {name: p.detach().cpu().numpy().copy() for name, p in model.state_dict().items()}
    else:
        final = {name: p.numpy().copy() for name, p in model.state_dict().items()}
        assert "torch" not in sys.modules
    np.savez_compressed(args.output / "final_weights.npz", **final)
    report = {"framework": args.framework, "version": version, "device": args.device,
              "steps": args.steps, "mode": "eval with parameter gradients; dropout disabled",
              "input": "same fixed real batch as fixture", "fixture": str(args.fixture),
              "optimizer": "AdamW lr=1e-4, betas=(0.9,0.999), eps=1e-8, decay=1e-4 excluding norm/bias",
              "first_loss": losses[0], "last_loss_before_update": losses[-1],
              "elapsed_seconds": time.perf_counter() - started,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "numeric optimizer trace, not a full training curve or online evaluation"}
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
