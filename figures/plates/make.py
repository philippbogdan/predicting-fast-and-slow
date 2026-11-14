#!/usr/bin/env python3
"""The ink plates of METHOD.md, drawn from results/ into this folder, light and dark. Its Fig. 4 is the paper's
Figure 2, drawn by figures/table.py.

    python figures/plates/make.py            draws them
    python figures/plates/make.py --check    draws them in memory and fails if a committed file differs

Needs numpy and matplotlib (which brings fontTools), and the two faces in fonts/. plate.py is the drawing
style, plate_<name>.py draws <name>-light.svg and <name>-dark.svg, and every number a plate shows comes from
scripts/plate_data.py or scripts/summary.py. tests/test_plates.py redraws every file, and figures/table.py's
two, and compares bytes.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.pop("PLATE_FONTS", None)          # the faces beside this file, so that the bytes are the same
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import plate  # noqa: E402

ORDER = ("claim", "protocol", "bias", "tail")       # Figs. 1, 2, 3 and 5 (Fig. 4 is the paper's, figures/table.py)
PX = 2.4                                                     # pixels to the point on the screen


def drawn():
    """{file name: its text} for every plate, light and dark."""
    import importlib
    out = {}
    for name in ORDER:
        for key, fig in importlib.import_module(f"plate_{name}").plates().items():
            for page in ("light", "dark"):
                s = fig.svg(page, PX)
                plate._verify(s, plate.PAGES[page])
                out[f"{key}-{page}.svg"] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = drawn()
    if a.check:
        bad = [n for n, s in files.items() if not (HERE / n).exists() or (HERE / n).read_text() != s]
        for n in files:
            print(("DIFFERS  " if n in bad else "same     ") + f"figures/plates/{n}")
        return 1 if bad else 0
    for n, s in files.items():
        (HERE / n).write_text(s)
    print(", ".join(f"{n} {len(s.encode()) / 1024:.0f} KB" for n, s in files.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
