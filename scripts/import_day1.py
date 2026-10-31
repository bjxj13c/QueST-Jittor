"""Verify and import the known day-one archive without trusting archive paths."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "backups/quest_day1_20261027.tar.gz"


def main():
    with gzip.open(ARCHIVE, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        manifest = json.load(tar.extractfile("records/reach_split_v1.json"))
        payload = tar.extractfile("data/reach_30_v1.hdf5").read()
        assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
        data_path = ROOT / "data/reach_30_v1.hdf5"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if data_path.exists():
            assert data_path.read_bytes() == payload
        else:
            data_path.write_bytes(payload)
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            assert not Path(member.name).is_absolute() and ".." not in parts
            if parts[0] not in {"scripts", "records", "outputs", "DAY1_NOTES.md"}:
                continue
            destination = ROOT / "provenance/day1" / member.name
            assert destination.resolve().is_relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            contents = tar.extractfile(member).read()
            if destination.exists():
                assert destination.read_bytes() == contents
            else:
                destination.write_bytes(contents)
            if parts[0] == "records":
                record = ROOT / member.name
                record.parent.mkdir(parents=True, exist_ok=True)
                if record.exists():
                    assert record.read_bytes() == contents
                else:
                    record.write_bytes(contents)
    report = {
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "dataset_sha256": manifest["sha256"],
        "dataset_bytes": len(payload),
        "train_episodes": len(manifest["train"]),
        "validation_episodes": len(manifest["validation"]),
        "verification": "gzip CRC, archived dataset SHA256, bounded extraction",
    }
    (ROOT / "reports/day1_import.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
