"""Strip trusted training checkpoints to portable inference state; no optimizer."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for path in (args.policy, args.tokenizer):
    if not path.resolve().is_relative_to(root):
        raise ValueError("Only load trusted locally generated project checkpoints")
if args.output.exists():
    raise FileExistsError("Use a fresh export directory")
args.output.mkdir(parents=True)
sizes = {}
for name in ("policy", "tokenizer"):
    source = getattr(args, name)
    state = torch.load(source, map_location="cpu", weights_only=False)
    target = args.output / f"{name}.pt"
    portable = {key: state[key] for key in ("model", "config", "extra")}
    portable["scope"] = "inference export; optimizer/RNG state removed; not exact-resumable"
    torch.save(portable, target)
    sizes[name] = {"source_bytes": source.stat().st_size, "inference_bytes": target.stat().st_size}
# Policy identity uses the original tokenizer .pt byte hash, which changes when
# stripping optimizer state. Preserve that semantic link explicitly for evaluator.
from quest_repro.policy_data import sha256
policy = torch.load(args.output / "policy.pt", map_location="cpu", weights_only=False)
policy["extra"]["original_tokenizer_checkpoint_sha256"] = policy["extra"]["tokenizer_sha256"]
policy["extra"]["tokenizer_sha256"] = sha256(args.output / "tokenizer.pt")
torch.save(policy, args.output / "policy.pt")
sizes["policy"]["inference_bytes"] = (args.output / "policy.pt").stat().st_size
report = {"files": sizes, "scope": "inference only, no training resume", "source_policy": str(args.policy),
          "source_tokenizer": str(args.tokenizer), "policy_sha256": sha256(args.output / "policy.pt"),
          "tokenizer_sha256": sha256(args.output / "tokenizer.pt")}
(args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
