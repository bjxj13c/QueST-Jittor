"""Exact original-vs-export check on parameters and a real validation input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from quest_repro.config import PolicyConfig, TokenizerConfig
from quest_repro.data import ActionChunkDataset
from quest_repro.policy_torch import QueSTPolicy
from quest_repro.tokenizer_torch import SkillVAE

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--export", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
for path in (args.policy, args.tokenizer, args.export):
    if not path.resolve().is_relative_to(root):
        raise ValueError("Trusted project files only")
outputs = []
batch = next(iter(DataLoader(ActionChunkDataset(root, "validation", include_observations=True), batch_size=2)))
for index, (policy_path, tokenizer_path) in enumerate(((args.policy, args.tokenizer),
                (args.export / "policy.pt", args.export / "tokenizer.pt"))):
    pstate = torch.load(policy_path, map_location="cpu", weights_only=False)
    tstate = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    if index == 0:
        original_policy, original_tokenizer = pstate["model"], tstate["model"]
    else:
        assert pstate["model"].keys() == original_policy.keys()
        assert tstate["model"].keys() == original_tokenizer.keys()
        for key in original_policy:
            assert torch.equal(pstate["model"][key], original_policy[key]), key
        for key in original_tokenizer:
            assert torch.equal(tstate["model"][key], original_tokenizer[key]), key
    policy, tokenizer = QueSTPolicy(PolicyConfig(**pstate["config"])).eval(), SkillVAE(TokenizerConfig(**tstate["config"])).eval()
    policy.load_state_dict(pstate["model"], strict=True)
    tokenizer.load_state_dict(tstate["model"], strict=True)
    with torch.no_grad():
        actions, tokens = policy.predict_actions(batch["obs"]["corner_rgb"], batch["obs"]["robot_states"],
                                                batch["task_id"], tokenizer, top_k=1)
    outputs.append((actions, tokens))
for original, exported in zip(outputs[0], outputs[1]):
    torch.testing.assert_close(original, exported, rtol=0, atol=0)
report = {"passed": True, "parameters_and_buffers": "exact", "real_validation_predicted_actions": "exact",
          "generated_tokens": "exact", "scope": "inference export preserves outputs, not resumable training"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
