"""Build audited local source/server ZIPs, excluding envs/caches/credentials.

No uploads, git publishing, paid resources or destructive file operations.
Every archived payload is hashed and re-read to verify archive integrity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def selected_files(root, kind):
    files = set()
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", ".gitignore", ".gitattributes"):
        if (root / name).is_file():
            files.add(root / name)
    for directory in ("src", "scripts", "tests", "requirements", "docs", "licenses", "reports"):
        for path in (root / directory).rglob("*"):
            if not path.is_file() or any(part == "__pycache__" or part.endswith(".egg-info") for part in path.parts) or path.suffix in (".pyc", ".log"):
                continue
            if path.suffix.lower() in (".py", ".ps1", ".cmd", ".sh", ".toml", ".md", ".txt", ".json", ".jsonl", ".xml", ".csv", ".png", ".pdf", ".mp4"):
                files.add(path)
    for name in ("reach_split_v1.json", "reach_train_statistics_v1.json", "quest_commit.txt",
                 "torch_cpu_freeze.txt", "jittor_environment_freeze.txt"):
        if (root / "records" / name).is_file():
            files.add(root / "records" / name)
    if kind == "server":
        for name in ("AGENTS.md", "PLAN.md", "HANDOFF.md"):
            if (root / name).is_file():
                files.add(root / name)
        files.add(root / "data/reach_30_v1.hdf5")
        for path in (root / "records").glob("*plan*.npz"):
            files.add(path)
        for path in (root / "outputs/fixtures").rglob("*"):
            if path.is_file() and path.suffix in (".npz", ".json"):
                files.add(path)
        for path in (root / "outputs/server_inference_v1").glob("*"):
            if path.is_file() and path.suffix in (".pt", ".json", ".npz"):
                files.add(path)
        for run in ("tokenizer_cpu_full_10epochs_v1", "policy_cpu_full_5epochs_v1",
                    "policy_cpu_warmstart5_plus10_v1", "policy_cpu_dev5_v1", "policy_cpu_dev5_zero_both_v1",
                    "policy_cpu_warmstart_dev5_v1", "policy_cpu_warmstart_dev5_zero_both_v1", "training_trace_torch_v1"):
            for path in (root / "outputs" / run).rglob("*"):
                if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".png", ".pdf", ".npz"):
                    files.add(path)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("source", "server"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    release = root / "releases"
    release.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = release / f"quest_{args.kind}_{timestamp}.zip"
    entries = []
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in selected_files(root, args.kind):
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or path.is_symlink():
                raise ValueError("Refusing a symlink or file outside this workspace")
            relative = path.relative_to(root).as_posix()
            if path.name in (".env", "id_rsa", "id_ed25519") or path.suffix in (".pem", ".key"):
                raise ValueError("Credential-like path excluded")
            payload = path.read_bytes()
            archive.writestr(relative, payload)
            entries.append({"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        manifest = {"kind": args.kind, "created_utc": datetime.now(timezone.utc).isoformat(), "files": entries,
                    "scope": "local preparation bundle, no upload or publication performed"}
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        for entry in entries:
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise RuntimeError("ZIP content SHA256 mismatch")
    report = {"file": str(destination), "bytes": destination.stat().st_size, "entries": len(entries),
              "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "verification": "CRC and every payload SHA256 passed"}
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
