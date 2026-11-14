"""The figures of METHOD.md are what their scripts draw from results/, byte for byte, and check.py fails when a
number in their captions is not what the result files say."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from test_check import copy, run

ROOT = Path(__file__).resolve().parents[1]
PLATES = ROOT / "figures" / "plates"


def make():
    spec = importlib.util.spec_from_file_location("make", PLATES / "make.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_plate_redraws_byte_for_byte():
    files = make().drawn()
    assert len(files) == 8
    for name, text in files.items():
        assert (PLATES / name).read_bytes() == text.encode(), f"figures/plates/{name} is not what its script draws"


def test_the_table_figure_redraws_byte_for_byte():
    """Fig. 4 is the paper's Figure 2, figures/table.py's light and dark versions."""
    sys.path.insert(0, str(ROOT / "figures"))
    import matplotlib.pyplot as plt
    import style
    import table
    for theme in ("light", "dark"):
        fig = table.figure(theme)
        text = style.svg(fig)
        plt.close(fig)
        assert (ROOT / "figures" / f"table-{theme}.svg").read_bytes() == text.encode(), f"figures/table-{theme}.svg"


def test_the_claim_is_read_as_numpy_reads_it():
    sys.path.insert(0, str(ROOT / "scripts"))
    import plate_data as D
    _, _, _, scored = D.S.heldout_records()
    claim = D.claim()
    for i, name in enumerate(sorted(scored)):
        z = np.load(ROOT / "results" / "heldout" / "hero" / f"{name}.npz")
        n, k = int(z["reads"]), int(z["runs"])
        size = np.diff(np.linspace(0, n, k + 1).round().astype(int))
        for key in D.SIGNALS:
            count = np.round(z[f"strip_{key}"].astype(np.float64) * size).astype(int)
            t = count.reshape(D.TENTHS, -1).sum(axis=1) / size.reshape(D.TENTHS, -1).sum(axis=1)
            assert list(100 * (t - count.sum() / size.sum())) == list(claim[key][i]), (name, key)


def test_check_fails_on_a_changed_caption_number(tmp_path):
    root = copy(tmp_path)
    page = root / "METHOD.md"
    text = page.read_text()
    assert "it stays within 6 points of the average" in text
    page.write_text(text.replace("it stays within 6 points", "it stays within 5 points"))
    done = run(root)
    assert done.returncode == 1
    assert "METHOD.md" in done.stdout
