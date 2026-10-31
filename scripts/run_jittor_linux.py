"""Project-scoped Jittor entry: explicit CPU/CUDA, existing compiler only."""
import argparse
import os
from pathlib import Path
import runpy
import shutil
import ssl
import sys
import sysconfig
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("script", type=Path)
parser.add_argument("arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
target = args.script.resolve()
if not target.is_relative_to(root) or not target.is_file():
    raise ValueError("Execute an existing project script")
if shutil.which("g++") is None:
    raise RuntimeError("An existing g++ compiler is required")
nvcc = shutil.which("nvcc") if args.device == "cuda" else ""
if args.device == "cuda" and not nvcc:
    raise RuntimeError("CUDA requested but nvcc absent; automatic installation is disabled")
# A venv created from Conda may not copy python3.x-config into its bin/.
# Keep explicit user selection; otherwise look in the actual base interpreter.
if "python_config_path" not in os.environ:
    helper = Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}-config"
    if helper.is_file():
        os.environ["python_config_path"] = str(helper)
header = Path(sysconfig.get_path("include")) / "Python.h"
if not header.is_file():
    raise RuntimeError(f"Python development header missing: {header}")
for key in ("jittor_home", "tmp", "matplotlib"):
    (root / "cache" / key).mkdir(parents=True, exist_ok=True)
os.environ["JITTOR_HOME"] = str(root / "cache/jittor_home")
# Jittor puts its base cache before the CUDA subdirectory on sys.path.
# A CPU core in that base can shadow the GPU core when switching devices.
# Separate cache roots retain both builds without deleting either artifact.
os.environ["cache_name"] = "quest_repro_" + args.device
os.environ["nvcc_path"] = nvcc
# Enabling CUDA through an environment flag runs its C++ setter during static
# initialization, before the executor exists. Enable it after import instead.
os.environ["use_cuda"] = "0"
# The default C* parallel operator compiler intermittently corrupted memory
# during repeated CUDA fixture runs on this host. Serial compilation leaves
# model arithmetic unchanged and avoids that thread pool.
os.environ["use_parallel_op_compiler"] = "0"
os.environ["TMPDIR"] = str(root / "cache/tmp")
os.environ["MPLCONFIGDIR"] = str(root / "cache/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "4")
https_context = ssl._create_default_https_context
with patch.object(Path, "home", return_value=root / "cache/jittor_home"):
    import jittor_utils
    ssl._create_default_https_context = https_context
    import jittor as jt
ssl._create_default_https_context = https_context
jt.flags.use_cuda = int(args.device == "cuda")
sys.path.insert(0, str(root / "src"))
sys.argv = [str(target)] + args.arguments
runpy.run_path(str(target), run_name="__main__")
