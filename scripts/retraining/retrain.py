#!/usr/bin/env python3
"""Heimdall retrained during the replay: four schedules scored on one held-out trace (METHOD.md, Retraining).

The paper's protocol is called unchanged: admission.protocol.load_trace builds the labels, the decision-time
inputs and the first minute's training set, admission.train trains the networks (seeds 0 to 5, seed 0 deployed,
and the Bayesian one), admission.compare.score_all scores the nine estimators and the table. The one thing varied
is when, and on what, the network and the estimators are fitted.

A schedule is a list of model segments. Segment 0 starts at 60 s with networks trained afresh on the first minute
(the paper's own training set). Segment k >= 1 starts at a refit time t_k (seconds) and holds networks fitted on a
stretch [a_k, t_k) of the replay log, using only I/Os submitted in it and completed before t_k: a read's latency,
and so its period label, exists only once it has completed, and Heimdall's period labeller (tail_v1) is run on the
stretch alone, so its labels need the whole stretch and are known only at t_k. The new networks act from t_k on
(an instant swap, which favours Heimdall).

Each refit is whole: the deployed network, the ensemble, the Bayesian network, the input scaling, the distance
centroid and covariance, the reference points of the nearest-neighbour and trust estimators and the conformal
calibration reads (the last 100 and 1,000 labelled reads of the stretch, with the stretch's period labels) are all
fitted on the stretch. The table is never refitted: it learns from every completed read, its T the first
minute's knee, as in the paper.

  heimdall_s7  Heimdall's Section 7: at 120 s, the deployed network's accuracy over the last minute's reads,
               against that minute's period labels, and a refit on the last minute when it is below 80%
  minute       a refit on the last minute at 120 s, whatever the accuracy
  slide10      a refit every 10 s from 70 s to 170 s, each on the last minute
  grow10       a refit every 10 s from 70 s to 170 s, each on everything since 0 s

What a rerun needs that the repository does not hold: the 57 held-out replay logs (a replay log carries its
trace, NOTICE.md), as parquet files named <trace>.parquet. results/heldout/traces.json records each one's SHA-256,
and scripts/heldout with bench/pair_replayer makes them from the public traces (METHOD.md, Run it). Training is
not bit-reproducible across machines, so a rerun's networks, and its numbers, differ a little from ours. Ours ran
on one machine with the torch and numpy versions each result file records.

Writes into WORK (per read, so never into the repository): reads/<arm>/<trace>.npz (per later read: Heimdall's
call and logit, the table's score and the nine estimators' scores), reads/shared/<trace>.npz (window, time, size,
both truths, the table's entry q), runs/<arm>/<trace>.json (the schedule as run) and models/ (the networks, kept
so that schedules sharing a stretch share its networks). scripts/retraining/metrics.py turns them into
results/retraining.

Usage: retrain.py TRACE [TRACE ...] [--replays DIR] [--work DIR] [--arms a,b,...]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402

from admission import compare as _compare  # noqa: E402
from admission import tracker as trk  # noqa: E402
from admission.compare import score_all  # noqa: E402
from admission.heimdall import FEATURES, feat_v6, tail_v1  # noqa: E402
from admission.protocol import CAL_LABELS, load_trace  # noqa: E402
from admission.train import load_models, save_models, train_bnn, train_mlps  # noqa: E402
from direct import Replay  # noqa: E402

NINE = ["euclid", "mahal", "conformal100", "conformal1000", "bnn", "margin", "ensemble", "knn", "trust"]
N_SEEDS = 6                      # as scripts/pipeline/run_detection.py: seed 0 deployed, 1 to 5 the ensemble
MIN_STRETCH_READS = 200          # a stretch with fewer reads keeps the previous networks
TRIGGER = 0.80                   # Heimdall's Section 7: retrain when accuracy over the last minute drops below 80%
EVERY10 = [float(t) for t in range(70, 180, 10)]

# arm -> (refit times, stretch rule, trigger, what is refitted, first-minute networks)
ARMS = {
    "heimdall_s7": dict(times=[120.0], stretch="last60", trigger=True, refit="all", first="fresh"),
    "minute": dict(times=[120.0], stretch="last60", trigger=False, refit="all", first="fresh"),
    "slide10": dict(times=EVERY10, stretch="last60", trigger=False, refit="all", first="fresh"),
    "grow10": dict(times=EVERY10, stretch="all", trigger=False, refit="all", first="fresh"),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def revision():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def stretch_training(df, a_s, t_s):
    """Heimdall's pipeline on the I/Os submitted in [a_s, t_s) and completed before t_s, as load_trace runs it
    on the first minute. Returns the training set, the period labels of the stretch's rows of df (-1 elsewhere),
    the labeller's latency knee and the stretch's size."""
    ts = df["ts_submit"].to_numpy(np.float64)
    fin = ts + df["latency"].to_numpy(np.float64) / 1000.0
    mask = (ts >= a_s * 1000.0) & (ts < t_s * 1000.0) & (fin < t_s * 1000.0)
    rows = np.flatnonzero(mask)
    st = df.iloc[rows]
    info = {"stretch": [a_s, t_s], "ios": int(len(rows)), "reads": int((st["io_type"].to_numpy() == 1).sum()),
            "submitted_not_completed": int(((ts >= a_s * 1000.0) & (ts < t_s * 1000.0) & ~(fin < t_s * 1000.0)).sum())}
    if info["reads"] < MIN_STRETCH_READS:
        return None, info
    sts = st.iloc[1:].reset_index(drop=True)          # the artefact's header quirk, as load_trace applies it
    lab = tail_v1(sts)
    kept = lab.order[lab.kept]
    labelled = sts.loc[kept, ["latency", "io_type", "size", "ts_submit"]].reset_index(drop=True)
    labelled["reject"] = lab.tail[lab.kept].astype(int)
    train = feat_v6(labelled)
    Xtr = train[FEATURES].to_numpy(np.float64)
    ytr = train["reject"].to_numpy(np.int64)
    if len(Xtr) < 100:
        # (a stretch whose period labels are all one class is trained on regardless, as the paper's pipeline
        # trains on such a first minute: msr.mds_1.x8)
        info["skipped"] = "fewer than 100 training rows"
        return None, info
    y_df = np.full(len(df), -1, np.int8)
    y_df[rows[1:][lab.order]] = lab.y_all.astype(np.int8)
    info.update(train_rows=int(len(Xtr)), train_slow_share=float(ytr.mean()))
    return (Xtr, ytr, y_df, float(lab.ip_latency)), info


def get_models(models_dir, name, key, Xtr, ytr):
    """Networks trained by the paper's recipe on one training set, cached so that arms sharing a stretch share
    its networks."""
    p = models_dir / name / f"{key}.npz"
    if p.exists():
        return load_models(p)
    models = train_mlps(Xtr, ytr, seeds=list(range(N_SEEDS)))
    bnn = train_bnn(Xtr, ytr, seed=0)
    p.parent.mkdir(parents=True, exist_ok=True)
    save_models(p, models, bnn)
    return models, bnn


class _NoReference:
    def __init__(self, **kw):
        pass

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


def score_segment(R, X, seg_idx, cal_idx, cal_y, models, bnn, Xtr, ytr, T):
    """score_all, unchanged, on the segment's reads with the calibration reads in front as its window 0.
    Every one of the nine depends only on the fitted state and the read itself, so this equals scoring the
    segment inside the whole trace (scripts/retraining/check_segment_scoring.py)."""
    idx = np.concatenate([cal_idx, seg_idx])
    Rs = R.iloc[idx].reset_index(drop=True)
    Rs.attrs = dict(R.attrs)
    win = np.concatenate([np.zeros(len(cal_idx), np.int8), np.ones(len(seg_idx), np.int8)])
    ycal = np.concatenate([cal_y, np.full(len(seg_idx), -1)]).astype(np.int64)
    # score_all also fits a boosted-tree reference on the window-0 reads (not one of the nine, reported nowhere
    # here, and 95% of the run time). In a refitted segment it is replaced by a stub that fits nothing
    real = _compare.HistGradientBoostingClassifier
    _compare.HistGradientBoostingClassifier = _NoReference
    try:
        sc, lg, pred, _ = score_all(Rs, X[idx], models, bnn, Xtr, ytr, T, win, ycal, CAL_LABELS)
    finally:
        _compare.HistGradientBoostingClassifier = real
    k = len(cal_idx)
    return {s: sc[s][k:] for s in NINE}, lg[k:], pred[k:]


def run(name, arms, replays, work):
    t0 = time.time()
    path = replays / f"{name}.parquet"
    models_dir = work / "models"
    D = load_trace(Replay(name, str(path), 180.0))
    df = pd.read_parquet(path)
    F = D["F"]
    R = F[F["io_type"].to_numpy() == 1].reset_index(drop=True)
    R.attrs["T_read_w0"] = D["T_read_w0"]
    X = R[FEATURES].to_numpy(np.float64)
    win = R["win"].to_numpy()
    ts_s = R["ts_us"].to_numpy(np.float64) / 1e6
    df_row = R["row"].to_numpy()
    later = np.flatnonzero(win >= 1)
    y_req = R["y_req"].to_numpy().astype(np.int8)
    y_w0 = R["y_w0"].to_numpy().astype(np.int64)
    T0 = float(D["labels_w0"].ip_latency)

    # the table: unchanged, learning from every completed read, its T the first minute's knee
    since = R["since_done_us"].to_numpy(np.float64)
    q = trk.evidence(R["ts_us"].to_numpy(np.float64), R["latency"].to_numpy(np.float64),
                     R["size"].to_numpy(np.float64), R["out_bytes"].to_numpy(np.float64),
                     R["oldest_age_us"].to_numpy(np.float64), since, float(D["T_read_w0"]))
    sh = work / "reads" / "shared"
    sh.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sh / f"{name}.npz", win=win[later], ts_us=R["ts_us"].to_numpy()[later],
                        size=R["size"].to_numpy()[later], y_req=y_req[later],
                        y_period=R["y"].to_numpy().astype(np.int8)[later], q=q[later].astype(np.float64))

    first = {}

    def first_minute():
        """The paper's own call (scripts/classwise.py) on every read, with networks trained on the first minute."""
        if not first:
            models, bnn = get_models(models_dir, name, "w0", D["train_X"], D["train_y"])
            sc, lg, pred, _ = score_all(R, X, models, bnn, D["train_X"], D["train_y"], T0, win, y_w0, CAL_LABELS,
                                        extra_cal={"G2": R["y_req_w0"].to_numpy().astype(np.int64)})
            first["w0"] = ({s: sc[s] for s in NINE + ["tracker"]}, lg, pred)
        return first["w0"]

    stretches = {}

    def stretch(a, t):
        if (a, t) not in stretches:
            stretches[(a, t)] = stretch_training(df, a, t)
        return stretches[(a, t)]

    for arm in arms:
        spec = ARMS[arm]
        sc0, lg0, pred0 = first_minute()
        scores = {s: sc0[s][later].astype(np.float64).copy() for s in NINE}
        logit = lg0[later].astype(np.float64).copy()
        segs = [{"start_s": 60.0, "fitted_on": "first minute (the paper's training set)", "retrained": False}]
        seg_of = np.zeros(len(later), np.int8)
        cur = None                     # the networks and fitted state in force after the last refit
        for k, t in enumerate(spec["times"], start=1):
            a = 0.0 if spec["stretch"] == "all" else t - 60.0
            res, info = stretch(a, t)
            seg = {"start_s": t, **info}
            in_seg = ts_s[later] >= t
            if res is None:
                seg["retrained"] = False
                seg["why_not"] = info.get("skipped", f"fewer than {MIN_STRETCH_READS} completed reads")
                segs.append(seg)
                continue
            Xtr, ytr, y_df, T_k = res
            y_r = y_df[df_row]                               # period labels of the stretch's reads
            st_reads = np.flatnonzero(y_r >= 0)
            if spec["trigger"]:
                # Heimdall's monitor: the deployed network's accuracy on the last minute's reads against that
                # minute's period labels, known at t
                assert cur is None, "the monitor is only run while the first minute's network is deployed"
                acc = float((pred0[st_reads] == y_r[st_reads]).mean())
                seg["accuracy_last_minute"] = acc
                if acc >= TRIGGER:
                    seg["retrained"] = False
                    seg["why_not"] = f"accuracy {acc:.3f} >= {TRIGGER}"
                    segs.append(seg)
                    continue
            key = f"s{a:g}-{t:g}"
            models_k, bnn_k = get_models(models_dir, name, key, Xtr, ytr)
            seg_idx = later[in_seg]
            cal_idx = st_reads[-max(1000, CAL_LABELS):]
            sc_k, lg_k, _ = score_segment(R, X, seg_idx, cal_idx, y_r[cal_idx], models_k, bnn_k, Xtr, ytr, T_k)
            for s in NINE:
                scores[s][in_seg] = sc_k[s]
            logit[in_seg] = lg_k
            seg_of[in_seg] = k
            cur = key
            seg.update(retrained=True, models=key,
                       epochs=[m.epochs for m in models_k], ip_latency_us=T_k)
            segs.append(seg)
        pred = (logit > 0)
        table = trk.mistake_score(q[later], pred)
        out = work / "reads" / arm
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{name}.npz", pred=pred.astype(np.int8), logit=logit.astype(np.float64),
                            segment=seg_of, s_tracker=table.astype(np.float64),
                            **{f"s_{s}": scores[s].astype(np.float64) for s in NINE})
        meta = {"trace": name, "arm": arm, "spec": spec, "segments": segs, "later_reads": int(len(later)),
                "replay": str(path), "replay_sha256": sha256(path), "repo_revision": revision(),
                "first_minute_models": str(models_dir / name / "w0.npz"),
                "torch": torch.__version__, "numpy": np.__version__, "seconds_so_far": round(time.time() - t0, 1)}
        rd = work / "runs" / arm
        rd.mkdir(parents=True, exist_ok=True)
        (rd / f"{name}.json").write_text(json.dumps(meta, indent=1))
        err = float((pred != y_req[later]).mean())
        print(f"{name} {arm}: refits={sum(s['retrained'] for s in segs[1:])}/{len(segs)-1} "
              f"wrong={err:.3f} called_slow={pred.mean():.3f} [{time.time()-t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--replays", type=Path, default=ROOT / "data" / "replays")
    ap.add_argument("--work", type=Path, default=ROOT / "data" / "retraining")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("RETRAIN_THREADS", "3")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    arms = a.arms.split(",")
    for n in a.traces:
        todo = [x for x in arms if not (a.work / "runs" / x / f"{n}.json").exists()]
        if todo:
            run(n, todo, a.replays, a.work)
        else:
            print(f"{n}: done before", flush=True)


if __name__ == "__main__":
    main()
