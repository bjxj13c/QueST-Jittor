"""Inspect tracked/staged text for common credential formats; never print values."""
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
patterns = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "url_with_password": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
}
findings = []
count = 0
for relative in paths:
    if not relative:
        continue
    path = root / relative
    if path.suffix.lower() in (".png", ".pdf", ".mp4", ".npz", ".pt", ".hdf5"):
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    count += 1
    for name, pattern in patterns.items():
        if pattern.search(text):
            findings.append({"path": relative, "kind": name})
report = {"text_files_checked": count, "findings": findings,
          "scope": "common-format scan, not a guarantee that arbitrary credentials are absent"}
print(json.dumps(report, indent=2))
if findings:
    raise SystemExit(1)
