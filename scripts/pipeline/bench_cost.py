#!/usr/bin/env python3
"""Measure the added per-decision cost of each gate on one CPU core.

Builds bench/gate_cost input from a real trace (decision-time records of
every later-window read: the 12 model inputs as io_replayer.c forms them,
the submission clock, the oldest outstanding submission, the bytes
outstanding, the read's latency and the last completion before it), the retrained network in flashnet_algo.c fixed point, and the
gates' parameters; compiles bench/gate_cost.c with -O2 and runs it pinned to
one core.  Writes results/cost/cost.json.
"""
import argparse
import json
import os
import platform
import subprocess
import tempfile

import numpy as np

from admission import RESULTS, ROOT
from admission.heimdall import FEATURES, FixedPointModel
from admission.protocol import load_trace
from admission.provenance import stamp
from admission.evidence import rule_params
from admission.train import heimdall_split, load_models
from admission.traces import by_name


def build_input(name, path):
    D = load_trace(by_name(name))
    F = D["F"]
    R = F[(F["io_type"] == 1) & (F["win"] >= 1)].reset_index(drop=True)
    models, _ = load_models(RESULTS / "models" / f"{name}.npz")
    m = models[0]
    fpm = FixedPointModel(m)
    X = R[FEATURES].to_numpy(np.int64)
    now = (R["ts_us"].to_numpy() * 1000).astype(np.int64)
    oldest = now - (R["oldest_age_us"].to_numpy() * 1000).astype(np.int64)   # oldest outstanding I/O
    recs = np.zeros(len(R), dtype=[("x", "<i8", 12), ("now", "<i8"), ("oldest", "<i8"), ("out_bytes", "<i8"),
                                   ("lat", "<i8"), ("last", "<i8")])
    recs["x"], recs["now"], recs["oldest"] = X, now, oldest
    recs["out_bytes"], recs["lat"] = R["out_bytes"].to_numpy(np.int64), R["latency"].to_numpy(np.int64)
    since = R["since_done_us"].to_numpy(np.float64)                          # last completion (quiet bit)
    recs["last"] = np.where(since >= 0, now - (np.maximum(since, 0) * 1000).astype(np.int64), -1)
    x_tr, *_ = heimdall_split(D["train_X"], D["train_y"])
    Zt = (x_tr - m.x_min) / m.x_range
    mu = Zt.mean(0)
    Pinv = np.linalg.pinv(np.cov(Zt.T) + 1e-6 * np.eye(12))
    de = np.sqrt(((Zt - mu) ** 2).sum(1))
    dm = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Zt - mu, Pinv, Zt - mu), 0))
    rp = rule_params(D)
    # conformal calibration set: last 1000 labelled W0 reads
    W0 = F[(F["io_type"] == 1) & (F["win"] == 0) & (F["y_w0"] >= 0)].tail(1000)
    Xc = W0[FEATURES].to_numpy(np.float64)
    Zc = ((Xc - m.x_min) / m.x_range).astype(np.float32)
    pc = 1 / (1 + np.exp(-m.logit(Xc)))
    yc = W0["y_w0"].to_numpy()
    sc = np.where(yc == 1, 1 - pc, pc).astype(np.float32)
    with open(path, "wb") as f:
        np.array([len(recs)], "<i8").tofile(f)
        recs.tofile(f)
        for a in (fpm.w0, fpm.b0, fpm.W1.ravel(), fpm.b1, fpm.W2.ravel(), fpm.b2, fpm.W3, np.array([fpm.b3])):
            np.asarray(a, "<i8").tofile(f)
        for a in (m.x_min, m.x_range, mu, Pinv.ravel()):
            np.asarray(a, "<f4").tofile(f)
        np.array([np.percentile(de, 90), np.percentile(dm, 90)], "<f4").tofile(f)
        np.array([int(rp["t_read_us"] * 1000), int(rp["t_read_us"] * 1000)], "<i8").tofile(f)
        np.array([len(Zc)], "<i4").tofile(f)
        Zc.tofile(f)
        sc.tofile(f)
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="P4d1")
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    inp = os.path.join(tmp, "in.bin")
    n = build_input(a.trace, inp)
    exe = os.path.join(tmp, "gate_cost")
    subprocess.run(["cc", "-O2", "-march=native", "-o", exe, str(ROOT / "bench" / "gate_cost.c"), "-lm"], check=True)
    cpu = sorted(os.sched_getaffinity(0))[0] if hasattr(os, "sched_getaffinity") else 0
    out = subprocess.run([exe, inp, str(a.reps), str(cpu)], capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    cpuinfo = platform.processor()
    try:
        cpuinfo = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    res = {"trace": a.trace, "n_decisions": n, "cpu": cpuinfo, "core": cpu, **r,
           # the larger of: the tracker's own instructions (decision + completion update) timed alone
           # in a tight loop, and the same instructions timed in situ inside the full decision path
           # (serialising time-stamp-counter brackets, bracket cost subtracted).  Differences of whole
           # ~4 us paths ("added_tracker_ns", "paired_added_tracker_ns") are reported but sit within
           # run-to-run noise of the path itself.
           "added_ns": max(r["insitu_tracker_ns"], r["tracker_hot_ns"]),
           "added_ns_definition": "max(decision + completion update timed in situ inside the full path with "
                                  "serialising brackets, the same timed alone on cache-resident requests); the "
                                  "decision part alone is insitu_decide_ns"}
    out_dir = RESULTS / "cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cost.json").write_text(json.dumps(stamp(res), indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
