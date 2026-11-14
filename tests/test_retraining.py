"""check.py fails when a number of METHOD.md's paragraph on retraining, or a file it rests on, is changed."""
import json
import re

from test_check import copy, run


def test_fails_on_a_changed_retraining_number(tmp_path):
    root = copy(tmp_path)
    method = root / "METHOD.md"
    text = method.read_text()
    said = re.search(r"the check fires on (\d+) of the", text)
    assert said, "METHOD.md no longer says on how many traces Heimdall's check fires"
    method.write_text(text.replace(said.group(0), f"the check fires on {int(said.group(1)) + 1} of the"))
    done = run(root)
    assert done.returncode == 1
    assert "FAIL  METHOD.md" in done.stdout


def test_fails_on_a_changed_retraining_result(tmp_path):
    root = copy(tmp_path)
    path = root / "results" / "retraining" / "slide10" / "umass.WebSearch1.asu0.x8.json"
    record = json.loads(path.read_text())
    record["windows"]["2"]["signals"]["tracker"]["pooled"] = 0.51
    path.write_text(json.dumps(record, indent=1) + "\n")
    done = run(root)
    assert done.returncode == 1
    assert "FAIL  1 result files match" not in done.stdout and "results/retraining/slide10" in done.stdout
