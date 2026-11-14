#!/usr/bin/env python3
"""retrain.py scores a refitted segment on its own reads, with its calibration reads in front. This shows that the
shortcut is exact: with the first minute's networks and calibration reads, scoring the later reads as a segment
gives the scores and the logit that scoring the whole trace gives. Prints the largest difference of each.

Usage: check_segment_scoring.py TRACE [--replays DIR] [--work DIR]   (after retrain.py has run on TRACE)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrain as r  # noqa: E402
from admission.heimdall import FEATURES  # noqa: E402
from admission.protocol import CAL_LABELS, load_trace  # noqa: E402
from direct import Replay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--replays", type=Path, default=ROOT / "data" / "replays")
ap.add_argument("--work", type=Path, default=ROOT / "data" / "retraining")
a = ap.parse_args()
n = a.trace
D = load_trace(Replay(n, str(a.replays / f"{n}.parquet"), 180.0))
F = D["F"]
R = F[F["io_type"].to_numpy() == 1].reset_index(drop=True)
R.attrs["T_read_w0"] = D["T_read_w0"]
X = R[FEATURES].to_numpy(np.float64)
win = R["win"].to_numpy()
y_w0 = R["y_w0"].to_numpy().astype(np.int64)
models, bnn = r.load_models(a.work / "models" / n / "w0.npz")
T0 = float(D["labels_w0"].ip_latency)
sc, lg, pred, _ = r.score_all(R, X, models, bnn, D["train_X"], D["train_y"], T0, win, y_w0, CAL_LABELS)
later = np.flatnonzero(win >= 1)
cal = np.flatnonzero((win == 0) & (y_w0 >= 0))[-1000:]
sc2, lg2, _ = r.score_segment(R, X, later, cal, y_w0[cal], models, bnn, D["train_X"], D["train_y"], T0)
print({k: float(np.abs(sc[k][later] - sc2[k]).max()) for k in r.NINE}, float(np.abs(lg[later] - lg2).max()))
