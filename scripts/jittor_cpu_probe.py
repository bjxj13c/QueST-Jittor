"""Small genuine Jittor CPU forward/backward/save/load probe."""
from pathlib import Path
import argparse
import json
import sys
import time

import jittor as jt
import jittor_utils
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "reports/jittor_cpu_probe.json")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()
jt.flags.use_cuda = int(args.device == "cuda")
started = time.perf_counter()
x = jt.array(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
loss = (x * x).sum()
gradient = jt.grad(loss, x)
np.testing.assert_allclose(loss.numpy(), 30.0, rtol=0, atol=0)
np.testing.assert_allclose(gradient.numpy(), np.array([[2., 4.], [6., 8.]], np.float32), rtol=0, atol=0)
path = ROOT / "outputs/jittor_cpu_probe.pkl"
path.parent.mkdir(parents=True, exist_ok=True)
jt.save({"x": x}, str(path))
restored = jt.load(str(path))
np.testing.assert_array_equal(np.asarray(restored["x"]), x.numpy())
assert jt.flags.use_cuda == int(args.device == "cuda")
assert "torch" not in sys.modules
report = {"status": "passed", "framework": "jittor", "version": jt.__version__,
          "device": args.device, "use_cuda": int(jt.flags.use_cuda),
          "cache_path": jittor_utils.cache_path, "compiler": jittor_utils.cc_path,
          "forward": 30.0, "gradient": gradient.numpy().tolist(),
          "save_load": "exact", "torch_imported": False,
          "elapsed_seconds_after_import": time.perf_counter() - started,
          "scope": "basic tensor probe on requested device; no model, training or performance verification"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
