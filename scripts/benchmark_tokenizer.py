"""Matched-core inference timing with real fixture actions and explicit sync.

Both frameworks time encode -> FSQ -> decode, excluding utilization metrics.
Repeated inputs are for profiling only, not independent evaluation samples.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from quest_repro.config import TokenizerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous benchmark")
    if args.batch_size <= 0 or args.repeats <= 0 or args.warmup < 1:
        raise ValueError("Require positive batch/repeats/warmup")
    metadata = json.loads((args.fixture / "metadata.json").read_text(encoding="utf-8"))
    config = TokenizerConfig(**metadata["config"])
    with np.load(args.fixture / "features.npz", allow_pickle=False) as fixture:
        examples = fixture["input"].copy()
    inputs = np.tile(examples, (int(np.ceil(args.batch_size / len(examples))), 1, 1))[:args.batch_size]
    peak_memory = None
    if args.framework == "torch":
        import torch
        from quest_repro.tokenizer_torch import SkillVAE
        torch.set_num_threads(4)
        model = SkillVAE(config).to(args.device).eval()
        with np.load(args.fixture / "weights.npz", allow_pickle=False) as weights:
            model.load_state_dict({name: torch.from_numpy(weights[name].copy()).to(args.device) for name in weights.files}, strict=True)
        actions = torch.from_numpy(inputs).to(args.device)
        def predict():
            with torch.no_grad():
                codes = model.vq(model.encode(actions))[0]
                return model.decode(codes).cpu().numpy()
        def synchronize():
            if args.device == "cuda":
                torch.cuda.synchronize()
        version = torch.__version__
    else:
        import jittor as jt
        from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz
        jt.flags.use_cuda = int(args.device == "cuda")
        model = SkillVAE(config)
        strict_load_npz(model, args.fixture / "weights.npz")
        model.eval()
        actions = jt.array(inputs)
        def predict():
            with jt.no_grad():
                codes = model.vq(model.encode(actions))[0]
                return model.decode(codes).numpy()
        def synchronize():
            jt.sync_all(True)
        version = jt.__version__
    for _ in range(args.warmup):
        result = predict()
        synchronize()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite benchmark prediction")
    if args.framework == "torch" and args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.repeats):
        synchronize()
        tick = time.perf_counter()
        result = predict()
        synchronize()
        timings.append((time.perf_counter() - tick) * 1000)
    if args.framework == "torch" and args.device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
    if args.framework == "jittor":
        assert "torch" not in sys.modules
    report = {"framework": args.framework, "version": version, "device": args.device,
        "platform": platform.platform(), "batch_size": args.batch_size, "config": config.to_dict(),
        "warmup": args.warmup, "repeats": args.repeats, "milliseconds": timings,
        "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)), "peak_torch_cuda_allocated_bytes": peak_memory,
        "jittor_peak_memory": "not measured by this script", "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "core encode/FSQ/decode, float32, eval, gradients disabled, synchronized, host transfer included",
        "input_scope": "repeated real fixture blocks for profiling; not independent evaluation samples",
        "limitations": ["run both frameworks on the same machine under comparable load",
                        "not a policy/controller decision-frequency benchmark"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"median_ms": report["median_ms"], "p95_ms": report["p95_ms"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
