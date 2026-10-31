"""Inspect a trained policy's RGB/state context on the lessons' real sample.

Local CPU, eval forward only. Does not train, generate actions, or run Jittor.
"""
import json
from pathlib import Path

import torch

from quest_repro.config import PolicyConfig
from quest_repro.data import ActionChunkDataset, load_manifest
from quest_repro.policy_torch import QueSTPolicy


def main():
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "outputs/fixtures/tokenizer_full_trained/metadata.json").read_text(encoding="utf-8"))
    manifest = load_manifest(root, verify_hash=True)
    if manifest["sha256"] != metadata["dataset_sha256"]:
        raise ValueError("Dataset and lesson example differ")
    episode, start = metadata["episodes"][0], int(metadata["starts"][0])
    dataset = ActionChunkDataset(root, "train", include_observations=True)
    item = dataset[dataset.index.index((episode, start))]
    images = torch.from_numpy(item["obs"]["corner_rgb"]).unsqueeze(0)
    states = torch.from_numpy(item["obs"]["robot_states"]).unsqueeze(0)
    task = torch.tensor([int(item["task_id"])], dtype=torch.long)

    torch.set_num_threads(2)
    checkpoint = torch.load(root / "outputs/server_inference_v1/policy.pt", map_location="cpu", weights_only=False)
    policy = QueSTPolicy(PolicyConfig(**checkpoint["config"])).eval()
    policy.load_state_dict(checkpoint["model"], strict=True)
    visual = policy.image_encoders["corner_rgb"]
    traces, handles = {}, []
    modules = {
        "ResNet stem": visual.resnet18_base,
        "ResNet block 1": visual.block_1,
        "ResNet block 2": visual.block_2,
        "ResNet block 3": visual.block_3,
        "ResNet block 4": visual.block_4,
        "SpatialSoftmax": visual.projection_layer.spatial_softmax,
        "Image embedding": visual,
        "State embedding": policy.lowdim_encoders["robot_states"],
        "Observation projection": policy.obs_proj,
        "Task embedding": policy.task_encoder,
    }
    for name, module in modules.items():
        def capture(module, inputs, output, name=name):
            traces[name] = list(output.shape)
        handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():
            context = policy.get_context(images, states, task)
    finally:
        for handle in handles:
            handle.remove()

    print("Lesson 06: trained policy observation context, CPU only")
    print(f"Real source: {episode}, start index {start}; same time as lessons 01-05")
    print("Raw image [batch, observation time, RGB, height, width]:", list(images.shape))
    print("Raw image dtype/range:", images.dtype, int(images.min()), int(images.max()))
    print("Policy divides RGB by 255 once, then merges batch/time for the CNN.")
    print("Robot state shape:", list(states.shape))
    print("Robot state values:", states[0, 0].numpy().round(4))
    print("State definition: concatenate obs[:4] and obs[18:22]; not the full 39-D state.")
    for name, shape in traces.items():
        print(name + ":", shape)
    print("Concatenate image/state features: [1, 1, 256] + [1, 1, 128] -> [1, 1, 384]")
    print("Observation projection: learned Linear(384, 384); width stays the same.")
    print("Task ID:", task.tolist(), "(single reach task; no language input)")
    print("Final context [task position, observation position]:", list(context.shape))
    print("All context values finite:", bool(torch.isfinite(context).all()))
    print("Scope: trained policy feature inspection; no new training, token prediction, or rollout.")


if __name__ == "__main__":
    main()
