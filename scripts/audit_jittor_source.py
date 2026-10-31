"""Download Jittor source for portability audit; do not install or import it."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    with urllib.request.urlopen("https://pypi.org/pypi/jittor/json", timeout=30) as response:
        metadata = json.load(response)
    item = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
    with urllib.request.urlopen(item["url"], timeout=60) as response:
        payload = response.read()
    assert hashlib.sha256(payload).hexdigest() == item["digests"]["sha256"]
    destination = ROOT / "provenance/jittor-source" / metadata["info"]["version"]
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.size > 2 * 1024 * 1024:
                continue
            if not member.name.endswith((".py", ".md", "LICENSE.txt", "LICENSE")):
                continue
            parts = Path(member.name).parts[1:]
            target = (destination.joinpath(*parts)).resolve()
            assert target.is_relative_to(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    report = {"version": metadata["info"]["version"], "url": item["url"],
              "sha256": item["digests"]["sha256"], "requires_dist": metadata["info"]["requires_dist"],
              "state": "downloaded for inspection; not installed or imported"}
    (ROOT / "reports/jittor_source_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
