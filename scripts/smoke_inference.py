"""Portable saved-model smoke test with a real validation input and optional exact reference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from quest_repro.config import PolicyConfig, TokenizerConfig
from quest_repro.data import ActionChunkDataset, load_manifest
from quest_repro.policy_torch import QueSTPolicy
from quest_repro.tokenizer_torch import SkillVAE

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--model-dir", type=Path, default=Path("outputs/server_inference_v1"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
root = args.root.resolve()
directory = args.model_dir if args.model_dir.is_absolute() else root / args.model_dir
if not directory.resolve().is_relative_to(root):
    raise ValueError("Only load trusted project inference weights")
if args.output.exists():
    raise FileExistsError("Preserve prior smoke result")
torch.set_num_threads(4)
manifest = load_manifest(root, verify_hash=True)
pstate = torch.load(directory / "policy.pt", map_location="cpu", weights_only=False)
tstate = torch.load(directory / "tokenizer.pt", map_location="cpu", weights_only=False)
policy, tokenizer = QueSTPolicy(PolicyConfig(**pstate["config"])).eval(), SkillVAE(TokenizerConfig(**tstate["config"])).eval()
policy.load_state_dict(pstate["model"], strict=True)
tokenizer.load_state_dict(tstate["model"], strict=True)
batch = next(iter(DataLoader(ActionChunkDataset(root, "validation", include_observations=True), batch_size=2)))
with torch.no_grad():
    actions, tokens = policy.predict_actions(batch["obs"]["corner_rgb"], batch["obs"]["robot_states"], batch["task_id"], tokenizer)
arrays = {"actions": actions.numpy(), "tokens": tokens.numpy()}
if args.reference:
    with np.load(args.reference, allow_pickle=False) as expected:
        for name, value in arrays.items():
            np.testing.assert_array_equal(value, expected[name])
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output, **arrays)
report = {"status": "passed", "dataset_sha256": manifest["sha256"], "episodes": batch["episode"],
          "starts": batch["start"].tolist(), "actions_shape": list(actions.shape), "tokens_shape": list(tokens.shape),
          "exact_reference_match": bool(args.reference), "scope": "portable checkpoint inference, not a policy rollout"}
args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
