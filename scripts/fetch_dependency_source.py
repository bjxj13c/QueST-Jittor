"""Fetch exact wheel sources from PyPI, verify digest, retain provenance.

No wheel is installed or executed by this script. Only selected text files are
copied for source auditing, and all destinations are bounded by this workspace.
"""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"vector-quantize-pytorch": "1.14.24", "positional-encodings": "6.0.3"}


def main():
    for package, version in PACKAGES.items():
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=40) as response:
            metadata = json.load(response)
        item = next(item for item in metadata["urls"] if item["filename"].endswith(".whl"))
        with urllib.request.urlopen(item["url"], timeout=60) as response:
            wheel = response.read()
        assert hashlib.sha256(wheel).hexdigest() == item["digests"]["sha256"]
        destination = ROOT / "provenance/dependencies" / f"{package}-{version}"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            for name in archive.namelist():
                if not (name.endswith(".py") or name.endswith("METADATA") or "LICENSE" in name.upper()):
                    continue
                path = (destination / name).resolve()
                assert path.is_relative_to(destination)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
        provenance = {"package": package, "version": version, "url": item["url"],
                      "sha256": item["digests"]["sha256"],
                      "requires_dist": metadata["info"]["requires_dist"]}
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
