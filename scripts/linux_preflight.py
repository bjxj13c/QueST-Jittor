"""Read-only Linux/CUDA/EGL inspection. Never prints credentials or full env."""
import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def run(command):
    if shutil.which(command[0]) is None:
        return {"available": False}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"available": True, "returncode": result.returncode,
                "stdout": result.stdout.strip()[-3000:], "stderr": result.stderr.strip()[-1000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(root)
    egl = {"found": ctypes.util.find_library("EGL")}
    try:
        ctypes.CDLL("libEGL.so.1")
        egl["load"] = "OK"
    except OSError as error:
        egl["load"] = str(error)
    report = {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
              "workspace": str(root), "disk_free_gib": disk.free / 2**30,
              "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
              "nvcc": run(["nvcc", "--version"]), "g++": run(["g++", "--version"]), "EGL": egl,
              "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
              "scope": "read-only environment inspection, not model/framework validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
