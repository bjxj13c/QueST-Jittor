"""Native Jittor optimizer grouping, without importing Torch."""
import jittor as jt
from jittor import nn


def owner_module(model, path):
    current = model
    for component in path.split(".") if path else []:
        current = current[component] if isinstance(current, nn.Sequential) else getattr(current, component)
    return current


def optimizer_groups(model, weight_decay=1e-4):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.is_stop_grad():
            continue
        path, _, leaf = name.rpartition(".")
        owner = owner_module(model, path)
        normalizer = type(owner).__name__ in ("LayerNorm", "GroupNorm", "BatchNorm2d", "Embedding")
        (no_decay if leaf.endswith("bias") or normalizer or not leaf.endswith("weight") else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def save_npz(model, path):
    import numpy as np
    values = {name: value.numpy() for name, value in model.state_dict().items()}
    for name in values:
        if name.endswith("num_batches_tracked"):
            values[name] = values[name].reshape(())
    np.savez_compressed(path, **values)
