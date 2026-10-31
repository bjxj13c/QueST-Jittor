"""Fetch the public read-only reference only if missing; verify pinned revision."""
from pathlib import Path
import subprocess
from quest_repro.reference import QUEST_COMMIT

root = Path(__file__).resolve().parents[1]
repository = root / "upstream/QueST"
if not repository.exists():
    repository.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", "https://github.com/pairlab/QueST.git", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "checkout", "--detach", QUEST_COMMIT], check=True)
actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
if actual != QUEST_COMMIT:
    raise RuntimeError("Existing reference has another revision; it was not modified")
print("Pinned reference:", actual)
