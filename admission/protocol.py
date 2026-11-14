"""The evaluation protocol shared by every experiment on the SSD-admission traces.

* Training window W0: requests submitted in the first third of the trace
  (60 s of the 180 s client-level traces).  Heimdall's model and every
  signal's fitted state see only W0.
* Later windows W1..W4: the remaining two thirds in four equal windows
  ([60,90), [90,120), [120,150), [150,180] s for the client-level traces).
  Every read request in them is scored; nothing is excluded.
* Labels: Heimdall's period labels, the artefact's own tail_v1 labelling of
  the full baseline replay (bit-identical to Heimdall's temp*/mldrive* files,
  with Heimdall's outliers given their own verdicts instead of being dropped:
  outlier1 -> fast, outlier2 -> slow); and the ground truth of every headline,
  each read's own latency above the knee of the trace's read-latency CDF (y_req).
* Decision-time information (identical for every signal): the 12 model
  inputs as io_replayer.c computes them, the model itself, decision-time
  device state visible to the client (in-flight count, oldest in-flight age,
  time since the last completion), the training window, and at most
  ``CAL_LABELS`` labelled requests from the end of W0 (conformal prediction
  with 1,000 calibration reads, and the supervised reference, use more).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .heimdall import (FEATURES, deploy_features, feat_v6, neighbourhood, online_tail_state, read_feedback, tail_v1,
                       tangent_based)
from .tracker import live_load
from .traces import DeviceTrace

CLIENT_LEVEL_DURATION_S = 180.0
CAL_LABELS = 100


def window_edges(duration_s: float) -> np.ndarray:
    """Start times (s) of W1..W4."""
    return duration_s / 3 + np.arange(4) * (2 * duration_s / 3) / 4


def window_of(ts_us: np.ndarray, duration_s: float = CLIENT_LEVEL_DURATION_S) -> np.ndarray:
    """0 for the training window, 1..4 for the later windows."""
    return np.searchsorted(window_edges(duration_s), ts_us / 1e6, side="right").astype(np.int8)


def load_trace(t: DeviceTrace) -> dict:
    """Baseline replay -> labels, deployment features, training set."""
    df = pd.read_parquet(t.replay_parquet("baseline"))
    duration = getattr(t, "duration_s", CLIENT_LEVEL_DURATION_S)
    # ground truth: tail_v1 over the full log exactly as Heimdall runs it
    lab = tail_v1(df.iloc[1:].reset_index(drop=True))
    y = np.full(len(df), -1, np.int8)
    y[lab.order + 1] = lab.y_all.astype(np.int8)
    outl = np.zeros(len(df), np.int8)
    outl[lab.order + 1] = lab.outlier
    F = deploy_features(df)
    rows = F["row"].to_numpy()
    F["y"] = y[rows]
    F["outlier"] = outl[rows]
    F["win"] = window_of(F["ts_us"].to_numpy(), duration)
    # row 0 (dropped by Heimdall's CSV header quirk) lies in W0; label it by
    # the same per-I/O rule tail_v1 uses for outlier2 so no row is unlabelled
    if (F["y"] < 0).any():
        miss = F["y"] < 0
        F.loc[miss, "y"] = (F.loc[miss, "latency"] > lab.ip_latency).astype(np.int8)

    # training set: Heimdall's pipeline run on W0 alone
    train_end_ms = window_edges(duration)[0] * 1000.0
    w0 = df[df["ts_submit"] < train_end_ms]
    lab0 = tail_v1(w0.iloc[1:].reset_index(drop=True))
    w0s = w0.iloc[1:].reset_index(drop=True)
    kept = lab0.order[lab0.kept]
    labelled = w0s.loc[kept, ["latency", "io_type", "size", "ts_submit"]].reset_index(drop=True)
    labelled["reject"] = lab0.tail[lab0.kept].astype(int)
    train = feat_v6(labelled)
    # W0 labels for calibration (what is known at deployment time)
    y0 = np.full(len(w0), -1, np.int8)
    y0[lab0.order + 1] = lab0.y_all.astype(np.int8)
    w0_rows = np.flatnonzero(df["ts_submit"].to_numpy() < train_end_ms)
    ycal = np.full(len(df), -1, np.int8)
    ycal[w0_rows] = y0
    F["y_w0"] = ycal[rows]
    # online labeller state at decision time (thresholds from W0 only)
    o, r = online_tail_state(df, lab0)
    F["tail_open"] = o
    F["tail_run"] = r
    # per-request ground truth G2: the read's own latency above the knee of
    # the read-latency CDF (hindsight, full trace); the threshold known at
    # deployment is the same knee computed on the training window
    is_read = df["io_type"].to_numpy() == 1
    T_read_full = float(tangent_based(df["latency"].to_numpy()[is_read])[0])
    T_read_w0 = float(tangent_based(w0.loc[w0["io_type"] == 1, "latency"].to_numpy())[0])
    F["y_req"] = (F["latency"].to_numpy() > T_read_full).astype(np.int8)
    y_req_w0 = np.where(F["win"].to_numpy() == 0, (F["latency"].to_numpy() > T_read_w0).astype(np.int8), -1)
    F["y_req_w0"] = y_req_w0
    frac, cfrac, last, oldest_read, out_w, out_wb = read_feedback(df, T_read_w0)
    F["fb_frac4"], F["fb_frac16"] = frac[:, 0], frac[:, 1]
    F["fb_size_class"] = cfrac
    F["fb_last_read"] = last
    F["oldest_read_age_us"] = oldest_read
    F["out_writes"], F["out_write_bytes"] = out_w, out_wb
    qa, qr = neighbourhood(df, T_read_w0)
    F["nb_all"], F["nb_reads"] = qa, qr
    # live load for the situation tracker (F is in submission order)
    tsF = F["ts_us"].to_numpy(np.float64)
    byt, _old = live_load(tsF, tsF + F["latency"].to_numpy(np.float64), F["size"].to_numpy(np.float64))
    F["out_bytes"] = byt
    return {
        "trace": t, "F": F, "train_X": train[FEATURES].to_numpy(np.float64),
        "train_y": train["reject"].to_numpy(np.int64), "labels_full": lab, "labels_w0": lab0,
        "T_read_full": T_read_full, "T_read_w0": T_read_w0, "duration_s": duration,
    }
