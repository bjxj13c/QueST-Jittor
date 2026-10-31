"""Verify a self-created manifest bundle, then restore into a NEW folder only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
parser.add_argument("--destination", type=Path, required=True)
args = parser.parse_args()
destination = args.destination.resolve()
if destination.exists():
    raise FileExistsError("Restoration requires a new directory; existing files are preserved")
with zipfile.ZipFile(args.archive) as archive:
    manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
    expected = {entry["path"] for entry in manifest["files"]}
    if set(archive.namelist()) != expected | {"PACKAGE_MANIFEST.json"}:
        raise ValueError("Unexpected or missing archive members")
    if len(archive.namelist()) != len(expected) + 1:
        raise ValueError("Duplicate archive members")
    for entry in manifest["files"]:
        name = PurePosixPath(entry["path"])
        target = (destination / entry["path"]).resolve()
        if name.is_absolute() or ".." in name.parts or "\\" in entry["path"] or not target.is_relative_to(destination):
            raise ValueError("Unsafe archive path")
        payload = archive.read(entry["path"])
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Archive payload verification failed")
    destination.mkdir(parents=True)
    for entry in manifest["files"]:
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(entry["path"]))
    (destination / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"restored": str(destination), "files": len(expected), "verification": "all manifest sizes/SHA256 and path bounds passed"}, indent=2))
