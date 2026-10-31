from pathlib import Path
import numpy as np
import h5py

from quest_repro.data import ActionChunkDataset, load_manifest, sample_distinct_episode_batch

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_episode_split_and_hash():
    manifest = load_manifest(ROOT, verify_hash=True)
    assert len(manifest["train"]) == 24
    assert len(manifest["validation"]) == 6
    assert len(set(manifest["train"] + manifest["validation"])) == 30


def test_windows_never_cross_episode_boundaries():
    dataset = ActionChunkDataset(ROOT)
    assert len(dataset) == 1176
    for episode in dataset.episodes:
        length = len(dataset.actions[episode])
        index = dataset.index.index((episode, length - 1))
        item = dataset[index]
        np.testing.assert_array_equal(item["actions"], np.repeat(dataset.actions[episode][-1:], 32, axis=0))
        assert item["pad_mask"].sum() == 1
    assert len(ActionChunkDataset(ROOT, "validation")) == 298


def test_policy_image_contract_does_not_double_normalize():
    dataset = ActionChunkDataset(ROOT, include_observations=True)
    item = dataset[0]
    image = item["obs"]["corner_rgb"]
    assert image.dtype == np.uint8 and image.shape == (1, 3, 128, 128)
    with h5py.File(dataset.path, "r") as file:
        episode, start = dataset.index[0]
        expected = file[episode]["images"][start].transpose(2, 0, 1)
    np.testing.assert_array_equal(image[0], expected)
    normalized = image.astype(np.float32) / 255.0
    assert normalized.max() > 0.5 and normalized.max() <= 1.0
    assert "observations" not in item and "obs_gt" not in item["obs"]


def test_debug_batch_uses_distinct_training_episodes():
    dataset = ActionChunkDataset(ROOT)
    batch = sample_distinct_episode_batch(dataset, size=8, seed=17)
    assert batch["actions"].shape == (8, 32, 4)
    assert len(set(batch["episodes"])) == 8
    assert batch["pad_mask"].all()
    assert not set(batch["episodes"]) & set(load_manifest(ROOT)["validation"])
