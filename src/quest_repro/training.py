"""Small, explicit training utilities with portable logs and checkpoint state."""
import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn


def load_npz_state(model, path):
    with np.load(path, allow_pickle=False) as archive:
        current = model.state_dict()
        if set(current) != set(archive.files):
            raise ValueError("Initial weight fields differ from model")
        converted = {}
        for name, target in current.items():
            value = archive[name]
            if value.shape != tuple(target.shape):
                raise ValueError(f"Initial weight shape mismatch: {name}")
            converted[name] = torch.from_numpy(value.copy()).to(device=target.device, dtype=target.dtype)
        model.load_state_dict(converted, strict=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_groups(model, weight_decay):
    no_decay_types = (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d, nn.GroupNorm)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        owner_name, _, leaf = name.rpartition(".")
        owner = model.get_submodule(owner_name) if owner_name else model
        if leaf.endswith("bias") or isinstance(owner, no_decay_types) or not leaf.endswith("weight"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    assert len({id(p) for p in decay + no_decay}) == len(decay + no_decay)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def reconstruction_metrics(prediction, target, mask):
    errors = (prediction - target).abs()
    expanded = mask.expand_as(errors)
    return {"mae_all": errors.mean().item(),
            "mae_valid": (errors * expanded).sum().item() / max(1, expanded.sum().item()),
            "mse_all": errors.square().mean().item()}


def cpu_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {key: cpu_state(value) for key, value in state.items()}
    if isinstance(state, list):
        return [cpu_state(value) for value in state]
    if isinstance(state, tuple):
        return tuple(cpu_state(value) for value in state)
    return state


def save_checkpoint(path, model, optimizer, step, config, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = {"format_version": 1, "model": cpu_state(model.state_dict()),
                "optimizer": cpu_state(optimizer.state_dict()) if optimizer else None,
                "step": step, "config": config, "extra": extra or {},
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(contents, temporary)
    os.replace(temporary, path)


class RunLogger:
    def __init__(self, directory, columns):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.directory / "metrics.csv").open("w", newline="", encoding="utf-8")
        self.json_file = (self.directory / "metrics.jsonl").open("w", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=columns)
        self.writer.writeheader()

    def log(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()
        self.json_file.write(json.dumps(row) + "\n")
        self.json_file.flush()

    def close(self):
        self.csv_file.close()
        self.json_file.close()


def device_synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def timed_forward(model, actions, repeats=10, warmup=2):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(actions)
        device_synchronize(actions.device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(actions)
        device_synchronize(actions.device)
    return (time.perf_counter() - started) * 1000 / repeats
