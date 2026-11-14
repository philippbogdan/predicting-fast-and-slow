#!/usr/bin/env python3
"""Per trace, schedule and later window: the paper's quantities, from the per-read files retrain.py wrote.

Writes results/retraining/<arm>/<trace>.json, holding the schedule as run (`spec`, `segments`, the replay's
SHA-256, the repository's revision and the torch and numpy versions) and, for every later window (1 to 4, 30 s
each), under `windows`:
  n, slow (reads whose own latency is above the knee), called_slow (Heimdall's slow calls), wrong (Heimdall's
  call disagrees with the read's own latency class),
  for the table (`tracker`) and each of the nine estimators, the AUROC (admission.metrics.auroc) with which its
  score ranks
    pooled       Heimdall's wrong guesses above its right ones, over every read,
    called_fast  among reads Heimdall calls fast, those that come back slow,
    called_slow  among reads Heimdall calls slow, those that come back fast,
  and forecast: the AUROC of the table's entry q and of Heimdall's logit for "this read comes back slow".
The definitions are those of scripts/classwise.py and scripts/direct.py. scripts/summary.py (retraining) turns
these files into the summary.

Usage: metrics.py [--work DIR] [--arms a,b] [TRACE ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from admission.metrics import auroc  # noqa: E402

NINE = ["euclid", "mahal", "conformal100", "conformal1000", "bnn", "margin", "ensemble", "knn", "trust"]
OUT = ROOT / "results" / "retraining"
KEPT = ("spec", "segments", "later_reads", "replay_sha256", "repo_revision", "torch", "numpy")


def windows(work, arm, name):
    sh = np.load(work / "reads" / "shared" / f"{name}.npz")
    rd = np.load(work / "reads" / arm / f"{name}.npz")
    win, y, q = sh["win"], sh["y_req"].astype(bool), sh["q"]
    pred = rd["pred"].astype(bool)
    logit = rd["logit"]
    wrong = pred != y
    out = {}
    for w in (1, 2, 3, 4):
        m = win == w
        if not m.any():
            continue
        f, s = m & ~pred, m & pred
        row = {"n": int(m.sum()), "slow": int(y[m].sum()), "called_slow": int(pred[m].sum()),
               "wrong": int(wrong[m].sum()), "signals": {},
               "forecast": {"table": auroc(y[m], q[m]), "heimdall": auroc(y[m], logit[m])}}
        for k in ["tracker"] + NINE:
            sc = rd[f"s_{k}"]
            row["signals"][k] = {"pooled": auroc(wrong[m], sc[m]), "called_fast": auroc(y[f], sc[f]),
                                 "called_slow": auroc(~y[s], sc[s])}
        out[str(w)] = row
    return out


def one(work, arm, name):
    run = json.loads((work / "runs" / arm / f"{name}.json").read_text())
    return {"trace": name, "arm": arm, **{k: run[k] for k in KEPT}, "windows": windows(work, arm, name)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--work", type=Path, default=ROOT / "data" / "retraining")
    ap.add_argument("--arms", default="heimdall_s7,minute,slide10,grow10")
    a = ap.parse_args()
    for arm in a.arms.split(","):
        names = a.traces or sorted(p.stem for p in (a.work / "reads" / arm).glob("*.npz"))
        od = OUT / arm
        od.mkdir(parents=True, exist_ok=True)
        for n in names:
            (od / f"{n}.json").write_text(json.dumps(one(a.work, arm, n), indent=1) + "\n")
        print(f"{arm}: {len(names)} traces", flush=True)


if __name__ == "__main__":
    main()
