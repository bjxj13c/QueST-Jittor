import json
from pathlib import Path
import numpy as np
import pytest
from quest_repro.batch_plan import create_plan, load_plan

ROOT = Path(__file__).resolve().parents[1]


def test_batch_plan_roundtrip_and_reject_changed_protocol(tmp_path):
    path = tmp_path / "plan.npz"
    create_plan(ROOT, path, 2, 16, 42)
    orders, digest = load_plan(path, ROOT, 2, 16)
    assert len(orders) == 2 and len(orders[0]) == 1176 and len(digest) == 64
    assert not np.array_equal(orders[0], orders[1])
    with pytest.raises(ValueError):
        load_plan(path, ROOT, 3, 16)
    with pytest.raises(ValueError):
        load_plan(path, ROOT, 2, 32)


def test_batch_plan_rejects_duplicate_or_other_split(tmp_path):
    path = tmp_path / "plan.npz"
    create_plan(ROOT, path, 1, 16, 42)
    with np.load(path, allow_pickle=False) as data:
        fields = {key: data[key].copy() for key in data.files}
    fields["epoch_1"][0] = fields["epoch_1"][1]
    changed = tmp_path / "duplicate.npz"
    np.savez_compressed(changed, **fields)
    with pytest.raises(ValueError):
        load_plan(changed, ROOT, 1, 16)
    metadata = json.loads(str(fields["metadata"]))
    metadata["split_fingerprint"] = "another-split"
    fields["metadata"] = np.asarray(json.dumps(metadata))
    changed = tmp_path / "other_split.npz"
    np.savez_compressed(changed, **fields)
    with pytest.raises(ValueError):
        load_plan(changed, ROOT, 1, 16)
