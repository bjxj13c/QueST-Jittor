"""Process-scoped Windows CPU entry with cache confinement and no CUDA installer.

Jittor's own utilities use Path.home for a small configuration file even when
JITTOR_HOME is set. During imports only, redirect that lookup to this workspace.
No HOME/USERPROFILE, registry, driver or existing environment is modified.
"""
import os
from pathlib import Path
import runpy
import ssl
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "cache/jittor_home"
HOME.mkdir(parents=True, exist_ok=True)
os.environ["JITTOR_HOME"] = str(HOME)
os.environ["nvcc_path"] = ""  # Explicitly prevents Jittor's automatic CUDA installation.
os.environ["use_cuda"] = "0"
os.environ["DISABLE_MULTIPROCESSING"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["use_parallel_op_compiler"] = "0"
# Retain Jittor's packaged data.gz: PyPI does not include all core .cc files.
# This installed MSVC currently rejects casts in data.cc; Linux is the verified
# next target, rather than bypassing required core definitions.
os.environ["TEMP"] = str(ROOT / "cache/tmp")
os.environ["TMP"] = str(ROOT / "cache/tmp")

if len(sys.argv) < 2:
    raise SystemExit("Usage: jittor_cpu_entry.py <trusted-project-script.py> [arguments]")
target = Path(sys.argv[1]).resolve()
if not target.is_relative_to(ROOT) or not target.is_file():
    raise ValueError("Only an existing script within this workspace may be executed")

verified_https_context = ssl._create_default_https_context
with patch.object(Path, "home", return_value=HOME):
    import jittor_utils
    # Upstream utilities alter this process default; retain verified HTTPS for downloads.
    ssl._create_default_https_context = verified_https_context
    import jittor as jt
ssl._create_default_https_context = verified_https_context
jt.flags.use_cuda = 0
assert Path(jittor_utils.cache_path).resolve().is_relative_to(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.argv = [str(target)] + sys.argv[2:]
runpy.run_path(str(target), run_name="__main__")
