#!/usr/bin/env python3
"""Which admitted read comes back slow?  Mistake ranking with the direction of the guess held fixed.

Heimdall calls "slow" far more often than reads are slow, so most of its wrong guesses are "slow"
calls on reads that come back fast, and any signal that knows the base rate ranks them above the rest
without telling one "slow" call from another.  This script removes that effect.  For one baseline
replay it recomputes every signal exactly as the per-window detection files were made (admission.compare, the
stored networks) and scores each later window three ways:

  pooled        every read: is the guess wrong?          (the detection files' statistic, reproduced as a check)
  called_fast   reads Heimdall admits as fast: which come back slow?
  called_slow   reads Heimdall would move as slow: which would have been fast?

and each of the three again on reads of one size only (same_size_*: the trace's commonest size class
among the later reads), because large reads are slow for being large and a table indexed by size
knows it.

Two controls say what the direction alone is worth: "said_slow" (the guess itself as the score) and
"one_cell" (the table collapsed to a single running slow-read frequency).  The study's two distance
gates are also scored as they act: a read is flagged when its distance exceeds the 90th percentile of
the training distances, and the script counts how often flagged and unflagged guesses are wrong.
"euclid_raw" is the Euclidean distance on unnormalised inputs, should the study have meant that.

Usage:
  classwise.py NAME --replay FILE.parquet --models FILE.npz --out DIR [--duration S] [--detection FILE.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from admission import signals as sig  # noqa: E402
from admission import tracker as trk  # noqa: E402
from admission.compare import score_all  # noqa: E402
from admission.heimdall import FEATURES  # noqa: E402
from admission.metrics import auroc  # noqa: E402
from admission.protocol import CAL_LABELS, load_trace  # noqa: E402
from admission.train import heimdall_split, load_models  # noqa: E402
from direct import Replay  # noqa: E402

KEEP = ["tracker", "euclid", "mahal", "conformal100", "conformal1000", "bnn", "margin", "ensemble", "knn", "trust",
        "euclid_aug", "mahal_aug", "conformal100_aug", "conformal1000_aug", "tracker_size_only", "feedback16",
        "supervised_w0_G2"]
GATE_PERCENTILE = 90
DETECTION = {"own_latency": "signals_G2_request", "period_labels": "signals"}


def run(name, replay, models_path, out_dir, duration, detection):
    t0 = time.time()
    D = load_trace(Replay(name, str(replay), duration))
    F = D["F"]
    R = F[F["io_type"].to_numpy() == 1].reset_index(drop=True)
    X = R[FEATURES].to_numpy(np.float64)
    win = R["win"].to_numpy()
    R.attrs["T_read_w0"] = D["T_read_w0"]
    models, bnn = load_models(models_path)
    sc, lg, pred, _ = score_all(R, X, models, bnn, D["train_X"], D["train_y"], float(D["labels_w0"].ip_latency), win,
                                R["y_w0"].to_numpy().astype(np.int64), CAL_LABELS,
                                extra_cal={"G2": R["y_req_w0"].to_numpy().astype(np.int64)})
    pred = pred.astype(bool)
    sc = {k: sc[k] for k in KEEP if k in sc}
    # controls: the guess itself, and the table collapsed to one cell
    ts = R["ts_us"].to_numpy(np.float64)
    lat = R["latency"].to_numpy(np.float64)
    T0 = float(D["T_read_w0"])
    q1 = trk.track(ts, lat, np.zeros(len(R), np.int64), (lat > T0).astype(np.float64), np.ones(len(R), np.bool_),
                   trk.ALPHA)
    sc["said_slow"] = pred.astype(np.float64)
    sc["one_cell"] = trk.mistake_score(q1, pred)
    # the study's gates as they act: flag above the 90th percentile of the training distances
    x_tr = heimdall_split(D["train_X"], D["train_y"])[0]
    Zt = (x_tr - models[0].x_min) / models[0].x_range
    flags = {"euclid": sc["euclid"] > np.percentile(sig.euclid(Zt, Zt), GATE_PERCENTILE),
             "mahal": sc["mahal"] > np.percentile(sig.mahal(Zt, Zt), GATE_PERCENTILE)}
    sc["euclid_raw"] = sig.euclid(X, x_tr)
    truths = {"own_latency": R["y_req"].to_numpy().astype(bool), "period_labels": R["y"].to_numpy().astype(bool)}
    # the commonest size class among the later reads (the table's own classes: floor(log2(bytes)))
    cls = np.floor(np.log2(np.maximum(R["size"].to_numpy(np.float64), 1.0))).astype(np.int64)
    modal = int(np.bincount(cls[win >= 1]).argmax())
    same = cls == modal
    out = {"trace": name, "n_reads": int(len(R)), "commonest_size_class": modal,
           "later_reads_of_that_size": int((same & (win >= 1)).sum()), "windows": {}}
    check = 0.0
    for w in sorted(set(win.tolist())):
        m = win == w
        row = {"n": int(m.sum()), "called_slow": int(pred[m].sum())}
        for tname, y in truths.items():
            fast_call, slow_call = m & ~pred, m & pred
            blk = {"slow": int(y[m].sum()), "called_fast_and_slow": int(y[fast_call].sum()),
                   "called_slow_and_fast": int((~y[slow_call]).sum()), "signals": {}}
            ms, fs, ss = m & same, fast_call & same, slow_call & same
            blk["same_size"] = {"n": int(ms.sum()), "slow": int(y[ms].sum()), "called_slow": int(pred[ms].sum()),
                                "called_fast_and_slow": int(y[fs].sum()),
                                "called_slow_and_fast": int((~y[ss]).sum())}
            for k, s in sc.items():
                blk["signals"][k] = {"pooled": auroc(pred[m] != y[m], s[m]),
                                     "called_fast": auroc(y[fast_call], s[fast_call]),
                                     "called_slow": auroc(~y[slow_call], s[slow_call]),
                                     "same_size_pooled": auroc(pred[ms] != y[ms], s[ms]),
                                     "same_size_called_fast": auroc(y[fs], s[fs]),
                                     "same_size_called_slow": auroc(~y[ss], s[ss])}
                if detection is not None and k in detection["windows"][str(w)][DETECTION[tname]]:
                    a, b = detection["windows"][str(w)][DETECTION[tname]][k]["auroc"], blk["signals"][k]["pooled"]
                    if not (np.isnan(a) and np.isnan(b)):
                        check = max(check, abs(a - b))
            wrong = pred != y
            blk["gates"] = {g: {"flagged": int(f[m].sum()), "flagged_wrong": int((f & wrong)[m].sum()),
                                "unflagged_wrong": int((~f & wrong)[m].sum()),
                                "called_fast_flagged": int((f & ~pred)[m].sum()),
                                "called_fast_flagged_slow": int((f & ~pred & y)[m].sum()),
                                "called_fast_unflagged_slow": int((~f & ~pred & y)[m].sum())}
                            for g, f in flags.items()}
            row[tname] = blk
        out["windows"][str(w)] = row
    if detection is not None:
        out["detection_file_max_abs_difference"] = check
    out["seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1))
    later = [w for w in out["windows"] if w != "0"]

    def worst(k, part):
        v = [out["windows"][w]["own_latency"]["signals"][k][part] for w in later]
        v = [x for x in v if x == x]
        return min(v) if v else float("nan")
    print(f"{name}: reads={len(R)} admitted reads that come back slow, worst window: "
          + " ".join(f"{k}={worst(k, 'called_fast'):.3f}" for k in ("tracker", "euclid", "mahal", "margin", "one_cell"))
          + (f"  detection file check {check:.1e}" if detection is not None else "") + f"  [{out['seconds']:.0f}s]",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--detection", default=None)
    a = ap.parse_args()
    pub = json.loads(Path(a.detection).read_text()) if a.detection else None
    run(a.name, Path(a.replay), Path(a.models), Path(a.out), a.duration, pub)


if __name__ == "__main__":
    main()
