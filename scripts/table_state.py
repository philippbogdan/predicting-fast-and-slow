#!/usr/bin/env python3
"""What the table has learnt by the end of one replay: every visited cell's entry and its history.

Replays the table (admission/tracker.py, unchanged) over one baseline replay and writes, for every
cell a read was submitted in: the entry q after the last completion, how many reads were submitted
there, how many of them came back slower than T, and how many Heimdall called slow.

Usage:
  table_state.py NAME --replay FILE.parquet --models FILE.npz --out FILE.json [--duration S]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from admission import tracker as trk  # noqa: E402
from admission.heimdall import FEATURES  # noqa: E402
from admission.protocol import load_trace  # noqa: E402
from admission.train import load_models  # noqa: E402
from direct import Replay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=180.0)
    a = ap.parse_args()
    D = load_trace(Replay(a.name, a.replay, a.duration))
    F = D["F"]
    R = F[F["io_type"].to_numpy() == 1].reset_index(drop=True)
    T = float(D["T_read_w0"])
    ts = R["ts_us"].to_numpy(np.float64)
    lat = R["latency"].to_numpy(np.float64)
    cell = trk.cells(R["size"].to_numpy(np.float64), R["out_bytes"].to_numpy(np.float64),
                     R["oldest_age_us"].to_numpy(np.float64), R["since_done_us"].to_numpy(np.float64), T)
    slow = lat > T
    called_slow = load_models(a.models)[0][0].logit(R[FEATURES].to_numpy(np.float64)) > 0
    tab = np.full(trk.N_CELLS, 0.5)
    for j in np.argsort(ts + lat, kind="mergesort"):          # completions, in order
        tab[cell[j]] += trk.ALPHA * (float(slow[j]) - tab[cell[j]])
    cells = []
    for c in np.unique(cell):
        m = cell == c
        z, rest = int(c % trk.N_QUIET), int(c // trk.N_QUIET)
        age, rest = rest % trk.N_AGE, rest // trk.N_AGE
        load, size = rest % trk.N_LOAD, rest // trk.N_LOAD
        cells.append({"size_class": int(size), "load_class": int(load), "age_class": int(age), "quiet": z,
                      "q": float(tab[c]), "reads": int(m.sum()), "slow": int(slow[m].sum()),
                      "heimdall_called_slow": int(called_slow[m].sum())})
    out = {"trace": a.name, "T_us": T, "reads": int(len(R)), "cells_visited": len(cells),
           "cells_total": trk.N_CELLS, "cells": cells}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.name}: {len(R)} reads, {len(cells)} of {trk.N_CELLS} cells visited, T = {T:.0f} us")


if __name__ == "__main__":
    main()
