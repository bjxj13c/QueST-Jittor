import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("compare_npz", Path(__file__).resolve().parents[1] / "scripts/compare_npz.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_comparator_detects_token_mismatch_even_if_float_errors_are_small(tmp_path):
    np.savez(tmp_path / "a.npz", indices=np.array([1, 2], dtype=np.int32), output=np.ones(2))
    np.savez(tmp_path / "b.npz", indices=np.array([1, 3], dtype=np.int32), output=np.ones(2))
    result = module.compare(tmp_path / "a.npz", tmp_path / "b.npz")
    assert not result["passed"] and not result["fields"]["indices"]["passed"]


def test_comparator_detects_missing_field_and_nonfinite(tmp_path):
    np.savez(tmp_path / "a.npz", output=np.ones(2), gradient=np.ones(2))
    np.savez(tmp_path / "b.npz", output=np.array([1.0, np.nan]))
    result = module.compare(tmp_path / "a.npz", tmp_path / "b.npz")
    assert not result["passed"] and result["missing_fields"] == ["gradient"]
    assert not result["fields"]["output"]["finite"]
