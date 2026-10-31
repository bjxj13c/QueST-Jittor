"""Execute native Jittor policy on the same real RGB/state Torch fixture."""
import argparse
import json
from pathlib import Path
import sys

import jittor as jt
import numpy as np

from quest_repro.config import PolicyConfig
from quest_repro.policy_jittor import QueSTPolicy
from quest_repro.tokenizer_jittor import strict_load_npz, eval_with_parameter_gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    jt.flags.use_cuda = int(args.device == "cuda")
    if args.output.exists():
        raise FileExistsError("Use a fresh candidate directory")
    args.output.mkdir(parents=True)
    metadata = json.loads((args.fixture / "metadata.json").read_text(encoding="utf-8"))
    model = eval_with_parameter_gradients(QueSTPolicy(PolicyConfig(**metadata["config"])))
    strict_load_npz(model, args.fixture / "weights.npz")
    with np.load(args.fixture / "features.npz", allow_pickle=False) as reference:
        data = {key: reference[key].copy() for key in ("images", "robot_states", "task_id", "targets")}
    images, state = jt.array(data["images"]), jt.array(data["robot_states"]).start_grad()
    tasks, targets = jt.array(data["task_id"]).int32(), jt.array(data["targets"]).int32()
    context = model.get_context(images, state, tasks)
    logits, loss = model(images, state, tasks, targets)
    parameters = [(name, p) for name, p in model.named_parameters() if not p.is_stop_grad()]
    derivatives = jt.grad(loss, [state] + [p for _, p in parameters])
    features = {**data, "context": context.numpy(), "logits": logits.numpy(),
                "loss": np.asarray(loss.numpy()).reshape(()), "robot_state_gradient": derivatives[0].numpy()}
    gradients = {name: gradient.numpy() for (name, _), gradient in zip(parameters, derivatives[1:])}
    np.savez_compressed(args.output / "features.npz", **features)
    np.savez_compressed(args.output / "gradients.npz", **gradients)
    assert "torch" not in sys.modules
    report = {"framework": "jittor", "version": jt.__version__, "device": args.device,
              "loss": float(features["loss"]), "torch_imported": False,
              "scope": "candidate outputs; compare_npz.py must pass before claiming parity"}
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
