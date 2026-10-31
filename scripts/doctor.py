"""Read-only project/environment checks, without connecting to a cloud server."""
import argparse
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-data", action="store_true", help="Fail if the frozen dataset is not prepared")
    args = parser.parse_args()
    root = args.root.resolve()
    packages = {}
    for name in ("torch", "torchvision", "numpy", "h5py", "einops", "einx", "vector-quantize-pytorch", "jittor"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "root": str(root),
              "free_gib": round(shutil.disk_usage(root).free / 1024**3, 2),
              "packages": packages, "checks": {}}
    failed = False
    try:
        from quest_repro.data import load_manifest
        if not (root / "records/reach_split_v1.json").exists() and not args.require_data:
            report["checks"]["dataset"] = "not prepared yet; environment-only check"
        else:
            manifest = load_manifest(root, verify_hash=True)
            report["checks"]["dataset_sha256"] = "passed"
            report["checks"]["train_episodes"] = len(manifest["train"])
            report["checks"]["validation_episodes"] = len(manifest["validation"])
    except Exception as error:
        report["checks"]["dataset"] = f"{type(error).__name__}: {error}"
        failed = True
    if packages["torch"]:
        import torch
        report["torch"] = {"cuda_available": torch.cuda.is_available(),
                           "cuda_build": torch.version.cuda,
                           "cpu_tensor_check": bool(torch.isfinite(torch.arange(4.) ** 2).all())}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
