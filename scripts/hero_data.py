#!/usr/bin/env python3
"""The data behind the picture at the top of the README and its second version (figures/hero.py), for one
baseline replay.

strips  Sort the trace's later reads from most to least suspect under one signal and cut the order
        into 240 equal runs; per run, the share of Heimdall's guesses that were wrong.  A signal that
        knows which guess is wrong puts the wrong ones first.  Signals: the table, and the study's
        two distances.
rain    The later reads on a grid of submission time by latency (log scale); per grid cell, how many
        reads fell there, how many the table called slow (its entry at least 1/2), and how many the
        Euclidean gate flagged (distance beyond the 90th percentile of the training distances).

Only these binned counts leave the replay; no request of any trace is written.

Usage:
  hero_data.py NAME --replay FILE.parquet --models FILE.npz --out DIR [--duration S] [--rain]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from admission import signals as sig  # noqa: E402
from admission import tracker as trk  # noqa: E402
from admission.heimdall import FEATURES  # noqa: E402
from admission.protocol import load_trace  # noqa: E402
from admission.train import heimdall_split, load_models  # noqa: E402
from direct import Replay  # noqa: E402

RUNS = 240
RAIN_T, RAIN_L = 600, 260                 # grid: submission time by log latency
RAIN_LAT_US = (40.0, 1600.0)


def strip(score, wrong, rng):
    """Share of wrong guesses in each of RUNS equal runs of the reads, most suspect first (ties in random order)."""
    order = np.lexsort((rng.random(len(score)), -score))
    edges = np.linspace(0, len(score), RUNS + 1).round().astype(int)
    w = wrong[order].astype(np.float64)
    return np.array([w[a:b].mean() if b > a else np.nan for a, b in zip(edges[:-1], edges[1:])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--rain", action="store_true")
    a = ap.parse_args()
    D = load_trace(Replay(a.name, a.replay, a.duration))
    F = D["F"]
    R = F[F["io_type"].to_numpy() == 1].reset_index(drop=True)
    X = R[FEATURES].to_numpy(np.float64)
    later = R["win"].to_numpy() >= 1
    ts = R["ts_us"].to_numpy(np.float64)
    lat = R["latency"].to_numpy(np.float64)
    slow = R["y_req"].to_numpy().astype(bool)
    T0 = float(D["T_read_w0"])
    model = load_models(a.models)[0][0]
    pred = model.logit(X) > 0
    q = trk.evidence(ts, lat, R["size"].to_numpy(np.float64), R["out_bytes"].to_numpy(np.float64),
                     R["oldest_age_us"].to_numpy(np.float64), R["since_done_us"].to_numpy(np.float64), T0)
    x_tr = heimdall_split(D["train_X"], D["train_y"])[0]
    Zt = (x_tr - model.x_min) / model.x_range
    Z = (X - model.x_min) / model.x_range
    scores = {"table": trk.mistake_score(q, pred), "euclid": sig.euclid(Z, Zt), "mahal": sig.mahal(Z, Zt)}
    wrong = pred != slow
    rng = np.random.default_rng(0)
    out = {"reads": int(later.sum()), "wrong": int(wrong[later].sum()), "runs": RUNS}
    for k, s in scores.items():
        out[f"strip_{k}"] = strip(s[later], wrong[later], rng).astype(np.float32)
    # the same with the guess held fixed: reads Heimdall admits as fast, wrong when they come back slow
    adm = later & ~pred
    out["admitted_reads"] = int(adm.sum())
    for k, s in scores.items():
        out[f"admitted_strip_{k}"] = strip(s[adm], wrong[adm], rng).astype(np.float32)
    if a.rain:
        flagged = scores["euclid"] > np.percentile(sig.euclid(Zt, Zt), 90)
        t0, t1 = a.duration / 3 * 1e6, a.duration * 1e6
        ti = np.clip(((ts[later] - t0) / (t1 - t0) * RAIN_T).astype(int), 0, RAIN_T - 1)
        lo, hi = np.log(RAIN_LAT_US[0]), np.log(RAIN_LAT_US[1])
        # latencies are whole microseconds: spread each read evenly within its microsecond
        l = lat[later] + rng.random(int(later.sum())) - 0.5
        li = np.clip(((np.log(np.maximum(l, 1.0)) - lo) / (hi - lo) * RAIN_L).astype(int), 0, RAIN_L - 1)
        for name, mask in (("reads", np.ones(int(later.sum()), bool)), ("table_says_slow", (q >= 0.5)[later]),
                           ("gate_flags", flagged[later]), ("slow", slow[later]), ("heimdall_says_slow", pred[later])):
            g = np.zeros((RAIN_L, RAIN_T), np.uint16)
            np.add.at(g, (li[mask], ti[mask]), 1)
            out[f"rain_{name}"] = g
        out["rain_latency_us"] = np.array(RAIN_LAT_US)
        out["rain_seconds"] = np.array([a.duration / 3, a.duration])
        out["knee_us"] = float(D["T_read_full"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"{a.name}.npz", **out)
    print(f"{a.name}: {out['reads']} later reads, {out['wrong']} wrong guesses"
          + (", rain grid written" if a.rain else ""), flush=True)


if __name__ == "__main__":
    main()
