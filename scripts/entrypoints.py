"""Small command-line front ends for the existing reproduction scripts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def build_command(framework, training, arguments=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=("Train the QueST tokenizer or policy" if training else "Evaluate a native Jittor policy"),
        epilog="Additional arguments are passed to the original script. See docs/TRAINING.md or docs/EVALUATION.md.",
    )
    if training:
        parser.add_argument("--stage", choices=("tokenizer", "policy"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without importing a model or starting a run")
    args, rest = parser.parse_known_args(arguments)
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if any(value == "--framework" or value.startswith("--framework=") for value in rest):
        parser.error("The framework is selected by this entry point; omit --framework")
    if training and args.stage == "policy" and args.device != "cuda":
        parser.error("The controlled policy-training script requires CUDA; tokenizer CPU checks remain available")

    command = [sys.executable]
    if framework == "jittor":
        # Keep the existing compiler/cache setup and the native Jittor execution path.
        command += [str(root / "scripts/run_jittor_linux.py"), "--device", args.device]
    if not training:
        target = "evaluate_policy_jittor.py"
        fixed = ["--device", args.device]
    elif args.stage == "tokenizer":
        target = "train_tokenizer_jittor.py" if framework == "jittor" else "train_tokenizer.py"
        fixed = ["--device", args.device]
    else:
        target = "train_controlled_policy.py"
        fixed = ["--framework", framework]
    command += [str(root / "scripts" / target), *fixed, *rest]
    return args, command


def main(framework, training=True):
    args, command = build_command(framework, training)
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    # Preserve the child script's exit status and its existing output checks.
    raise SystemExit(subprocess.call(command))
