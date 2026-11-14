#!/usr/bin/env python3
"""The comparison: every candidate signal on every trace and later window.

Per device trace:
  * Heimdall's network retrained with Heimdall's own pipeline on W0 (seed 0
    is the deployed model; seeds 1-5 form the deep ensemble); a Bayesian
    version for the study's BNN estimator.
  * Every read request of every window is scored by every signal from the
    same decision-time information (admission.protocol, admission.compare).
  * Metrics per window: AUROC and AURC/E-AURC for "decision != label",
    against Heimdall's period labels ("signals") and against each read's own
    latency ("signals_G2_request", the ground truth of every headline).
Outputs results/detection/<trace>.json and per-request arrays in
results/big/scores/<trace>.npz (read by scripts/export_ranked.py and the gate runs).
"""
import argparse
import json
import os
import time

import numpy as np

from admission import RESULTS
from admission.compare import score_all, window_metrics
from admission.heimdall import FEATURES
from admission.protocol import CAL_LABELS, load_trace
from admission.provenance import stamp
from admission.train import load_models, save_models, train_bnn, train_mlps
from admission.traces import ALL_DEVICE_TRACES, by_name

N_ENSEMBLE = 5


RETRAIN = False   # --retrain: train new networks instead of loading results/models (training is not bit-reproducible across machines)


def get_models(name, Xtr, ytr):
    mpath = RESULTS / "models" / f"{name}.npz"
    if mpath.exists() and not RETRAIN:
        return load_models(mpath)
    models = train_mlps(Xtr, ytr, seeds=list(range(N_ENSEMBLE + 1)))
    bnn = train_bnn(Xtr, ytr, seed=0)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    save_models(mpath, models, bnn)
    return models, bnn


def run(name, out_dir, big_dir):
    t0 = time.time()
    D = load_trace(by_name(name))
    F = D["F"]
    Xtr, ytr = D["train_X"], D["train_y"]
    models, bnn = get_models(name, Xtr, ytr)
    R = F[F["io_type"].to_numpy() == 1].reset_index(drop=True)
    X = R[FEATURES].to_numpy(np.float64)
    win = R["win"].to_numpy()
    y = R["y"].to_numpy().astype(np.int64)
    y_w0 = R["y_w0"].to_numpy().astype(np.int64)
    T = float(D["labels_w0"].ip_latency)
    R.attrs["T_read_w0"] = D["T_read_w0"]
    sc, lg, pred, unc = score_all(R, X, models, bnn, Xtr, ytr, T, win, y_w0, CAL_LABELS,
                                  extra_cal={"G2": R["y_req_w0"].to_numpy().astype(np.int64)})
    mistake = pred != y
    y_lat = R["y_req"].to_numpy().astype(np.int64)
    out = {"trace": name, "n_reads": int(len(R)), "train_rows": int(len(Xtr)),
           "train_epochs": [m.epochs for m in models], "T_us": T,
           "T_read_full_us": D["T_read_full"], "T_read_w0_us": D["T_read_w0"],
           "uncertain_fraction_later": unc,
           "windows": window_metrics(sc, mistake, pred, y, win, {"G2_request": pred != y_lat})}
    out["seconds"] = time.time() - t0
    (out_dir / f"{name}.json").write_text(json.dumps(stamp(out), indent=1))
    np.savez_compressed(
        big_dir / f"{name}.npz", win=win, mistake=mistake, pred=pred, y=y, y_lat=y_lat, logit=lg,
        ts_us=R["ts_us"].to_numpy(), row=R["row"].to_numpy(),
        queue_len=R["queue_len"].to_numpy(np.int16), n_inflight=R["n_inflight"].to_numpy(np.int16),
        prev_latency_1=R["prev_latency_1"].to_numpy(np.int32), size=R["size"].to_numpy(np.int32),
        latency=R["latency"].to_numpy(np.int32), ip_latency_w0=T, T_read_w0=D["T_read_w0"],
        out_bytes=R["out_bytes"].to_numpy(np.float64), oldest_age_us=R["oldest_age_us"].to_numpy(np.float64),
        since_done_us=R["since_done_us"].to_numpy(np.float64),
        ip_latency_full=D["labels_full"].ip_latency,
        **{f"s_{k}": v.astype(np.float32) for k, v in sc.items()})
    later = [w for w in out["windows"] if w != "0"]
    print(f"{name}: reads={len(R)} err(later)={mistake[win >= 1].mean():.3f} worst-window AUROC: "
          + " ".join(f"{k}={min(out['windows'][w]['signals'][k]['auroc'] for w in later):.3f}" for k in sc)
          + f"  [{out['seconds']:.0f}s]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*", default=[t.name for t in ALL_DEVICE_TRACES])
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    global RETRAIN
    RETRAIN = a.retrain
    out_dir = RESULTS / "detection"
    big_dir = RESULTS / "big" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    big_dir.mkdir(parents=True, exist_ok=True)
    for name in a.traces:
        run(name, out_dir, big_dir)


if __name__ == "__main__":
    main()
