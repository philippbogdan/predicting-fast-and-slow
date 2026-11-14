"""check.py passes on the repository as it is and fails as soon as a result is not what the numbers were computed from."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ["check.py", "README.md", "admission", "scripts", "results", "docs", "paper/numbers.tex"]


def copy(tmp_path):
    for name in NEEDED:
        src, dst = ROOT / name, tmp_path / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy(src, dst)
    return tmp_path


def run(root):
    return subprocess.run([sys.executable, str(root / "check.py")], capture_output=True, text=True)


def test_passes_as_it_stands(tmp_path):
    done = run(copy(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "every check passed" in done.stdout


def test_fails_on_a_changed_result(tmp_path):
    root = copy(tmp_path)
    path = root / "results" / "heldout" / "detection" / "umass.WebSearch1.asu0.x8.json"
    record = json.loads(path.read_text())
    record["windows"]["1"]["signals_G2_request"]["euclid"]["auroc"] = 0.99
    path.write_text(json.dumps(record, indent=1) + "\n")
    done = run(root)
    assert done.returncode == 1
    assert "FAIL" in done.stdout


def test_fails_on_a_changed_quoted_number(tmp_path):
    root = copy(tmp_path)
    readme = root / "README.md"
    readme.write_text(readme.read_text().replace("50 of the 57", "57 of the 57"))
    done = run(root)
    assert done.returncode == 1
    assert "README.md" in done.stdout
