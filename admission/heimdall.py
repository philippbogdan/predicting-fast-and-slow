"""Faithful, fast reimplementation of Heimdall's data pipeline.

* ``tail_v1``: Heimdall's period-based labelling
  (integration/client-level/experiment/flashnet/training/TailAlgorithms/tail_v1.py)
* ``feat_v6``: Heimdall's training features
  (.../training/FeatureExtractors/feat_v6.py): history = previous rows in
  submission order.
* ``deploy_features``: the features Heimdall's C replayer actually feeds the
  model at decision time (.../flashnet/io_replayer.c + flashnet_algo.c):
  queue length = outstanding I/Os of this client on the device including the
  new one; history = the three most recently *completed* I/Os.
* ``ArtefactModel``: the trained 12-128-16-1 network from the artefact's
  weight CSVs (MinMax scaler + dense layers), evaluated in float.

Every function is checked against the artefact's own intermediate files by
``scripts/validate_pipeline.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
import pandas as pd

N_HISTORY = 3
N_FUTURE = 3
THPT_DROP_RATE = 1.7
FEATURES = [
    "io_type", "size", "queue_len",
    "prev_queue_len_1", "prev_queue_len_2", "prev_queue_len_3",
    "prev_latency_1", "prev_latency_2", "prev_latency_3",
    "prev_throughput_1", "prev_throughput_2", "prev_throughput_3",
]


# --------------------------------------------------------------------------
# tail_v1 labelling
# --------------------------------------------------------------------------
def tangent_based(values: np.ndarray) -> tuple[int, float]:
    """Heimdall's inflection-point finder (TailAlgorithms/default_ip_finder.py)."""
    a = np.asarray(values, dtype=np.float64)
    a = a[~np.isinf(a)]
    p97 = np.percentile(a, 97)
    a = a[a <= p97]
    mx = np.max(a)
    a = np.sort(a / mx)
    x = np.linspace(0, 1, num=10000)
    y = np.searchsorted(a, x, side="right") / len(a)  # statsmodels ECDF
    i = int(np.argmax(y - x))
    return int(x[i] * mx), float(y[i] * 0.97 * 100)


@numba.njit(cache=True)
def _mark_tail(start1, fut_ok):
    n = start1.shape[0]
    tail = np.zeros(n, np.bool_)
    idx = 0
    while idx <= n - 1:
        if start1[idx]:
            tail[idx] = True
            while idx < n - 1:
                idx += 1
                if fut_ok[idx]:
                    break
                tail[idx] = True
        idx += 1
    return tail


@numba.njit(cache=True)
def _short_runs(tail_f, maxlen):
    n = tail_f.shape[0]
    out = np.zeros(n, np.bool_)
    idx = 0
    while idx <= n - 1:
        if tail_f[idx]:
            n_tail = 1
            while idx < n - 1:
                idx += 1
                if not tail_f[idx]:
                    if n_tail <= maxlen:
                        for k in range(idx - n_tail, idx):
                            out[k] = True
                    break
                n_tail += 1
        idx += 1
    return out


@dataclass
class TailLabels:
    order: np.ndarray        # row indices of the input, sorted by ts_submit (Heimdall's order)
    tail: np.ndarray         # period label per sorted row ("Tail-Period")
    outlier: np.ndarray      # 0 none, 1/2/3 = Heimdall's outlier classes, per sorted row
    y_all: np.ndarray        # label for every sorted row: tail, with outlier1->0, outlier2->1
    ip_latency: int
    ip_throughput: int
    median_latency: float
    median_throughput: float

    @property
    def kept(self) -> np.ndarray:
        """Rows Heimdall keeps for training (sorted-row mask)."""
        return self.outlier == 0


# _aheapsort and generic_argsort port NumPy's generic argsort (aheapsort_ and aquicksort_ in
# numpy/_core/src/npysort), Copyright (c) 2005-2026, NumPy Developers, under the BSD 3-Clause licence;
# its full text is in NOTICE.md.
@numba.njit(cache=True)
def _aheapsort(v, a, lo, n):
    """NumPy's generic aheapsort_ on tosort[lo:lo+n] (1-based heap indexing)."""
    for l in range(n >> 1, 0, -1):
        tmp = a[lo + l - 1]
        i = l
        j = l << 1
        while j <= n:
            if j < n and v[a[lo + j - 1]] < v[a[lo + j]]:
                j += 1
            if v[tmp] < v[a[lo + j - 1]]:
                a[lo + i - 1] = a[lo + j - 1]
                i = j
                j += j
            else:
                break
        a[lo + i - 1] = tmp
    while n > 1:
        tmp = a[lo + n - 1]
        a[lo + n - 1] = a[lo]
        n -= 1
        i = 1
        j = 2
        while j <= n:
            if j < n and v[a[lo + j - 1]] < v[a[lo + j]]:
                j += 1
            if v[tmp] < v[a[lo + j - 1]]:
                a[lo + i - 1] = a[lo + j - 1]
                i = j
                j += j
            else:
                break
        a[lo + i - 1] = tmp


@numba.njit(cache=True)
def generic_argsort(v):
    """Argsort identical to NumPy's *generic* (non-SIMD) aquicksort_: median-of-3
    introsort, insertion sort below 16 elements, heapsort past 2*floor(log2 n)
    depth.  NumPy dispatches argsort to SIMD sorts on some x86 CPUs, which order
    equal keys differently; Heimdall's tail_v1 depends on that order (pandas
    sort_values, an unstable quicksort), so we fix it to the generic one: the
    order that reproduces the artefact's label files exactly.  No NaNs expected."""
    num = v.shape[0]
    a = np.arange(num)
    if num < 2:
        return a
    SMALL = 15
    stack_l = np.empty(256, np.int64)
    stack_r = np.empty(256, np.int64)
    depth_s = np.empty(256, np.int64)
    sp = 0
    cdepth = 0
    t = num
    while t > 1:
        t >>= 1
        cdepth += 1
    cdepth *= 2
    pl = 0
    pr = num - 1
    while True:
        if cdepth < 0:
            _aheapsort(v, a, pl, pr - pl + 1)
        else:
            while pr - pl > SMALL:
                pm = pl + ((pr - pl) >> 1)
                if v[a[pm]] < v[a[pl]]:
                    a[pm], a[pl] = a[pl], a[pm]
                if v[a[pr]] < v[a[pm]]:
                    a[pr], a[pm] = a[pm], a[pr]
                if v[a[pm]] < v[a[pl]]:
                    a[pm], a[pl] = a[pl], a[pm]
                vp = v[a[pm]]
                pi = pl
                pj = pr - 1
                a[pm], a[pj] = a[pj], a[pm]
                while True:
                    pi += 1
                    while v[a[pi]] < vp:
                        pi += 1
                    pj -= 1
                    while vp < v[a[pj]]:
                        pj -= 1
                    if pi >= pj:
                        break
                    a[pi], a[pj] = a[pj], a[pi]
                pk = pr - 1
                a[pi], a[pk] = a[pk], a[pi]
                cdepth -= 1
                if pi - pl < pr - pi:
                    stack_l[sp] = pi + 1
                    stack_r[sp] = pr
                    depth_s[sp] = cdepth
                    sp += 1
                    pr = pi - 1
                else:
                    stack_l[sp] = pl
                    stack_r[sp] = pi - 1
                    depth_s[sp] = cdepth
                    sp += 1
                    pl = pi + 1
            # insertion sort
            for pi in range(pl + 1, pr + 1):
                vi = a[pi]
                vp = v[vi]
                pj = pi
                pk = pi - 1
                while pj > pl and vp < v[a[pk]]:
                    a[pj] = a[pk]
                    pj -= 1
                    pk -= 1
                a[pj] = vi
        if sp == 0:
            break
        sp -= 1
        pl = stack_l[sp]
        pr = stack_r[sp]
        cdepth = depth_s[sp]
    return a


def tail_v1(df: pd.DataFrame) -> TailLabels:
    """Label a replay log exactly like Heimdall's tail_v1.py.

    ``df`` has columns latency (us), size (B), ts_submit (ms) and is in log
    order.  Returns labels over the rows sorted by ts_submit with the order
    pandas' quicksort produces in NumPy's generic implementation (as Heimdall
    did; see generic_argsort; platform independent).
    """
    order = df.index.to_numpy()[generic_argsort(df["ts_submit"].to_numpy(np.float64))]
    lat = df["latency"].to_numpy(np.float64)[order]
    size = df["size"].to_numpy(np.float64)[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        thpt = np.round(size / lat, 0)
    ip_lat, _ = tangent_based(lat)
    ip_thpt, _ = tangent_based(thpt)
    med_thpt = float(np.percentile(thpt, 50))
    med_lat = float(np.percentile(lat, 50))

    prev = np.concatenate([[0.0], thpt[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = np.round(prev / (thpt + 0.1), 1)
    start1 = ~((thpt > ip_thpt) | (lat < ip_lat)) & (drop >= THPT_DROP_RATE)

    n = len(thpt)
    fut_ok = np.ones(n, np.bool_)
    for k in range(N_FUTURE):
        f = np.zeros(n)
        f[: n - k] = thpt[k:]
        fut_ok &= f >= med_thpt
    tail = _mark_tail(start1, fut_ok)

    out1 = tail & (lat <= med_lat) & (thpt >= med_thpt)
    out2 = ~tail & (lat > ip_lat) & (thpt < med_thpt)
    outlier = np.zeros(n, np.int8)
    outlier[out1] = 1
    outlier[out2] = 2
    keep = outlier == 0
    out3_f = _short_runs(tail[keep], N_HISTORY)
    idx_keep = np.flatnonzero(keep)
    outlier[idx_keep[out3_f]] = 3

    y_all = tail.copy()
    y_all[out1] = False
    y_all[out2] = True
    return TailLabels(order, tail, outlier, y_all, ip_lat, ip_thpt, med_lat, med_thpt)


# --------------------------------------------------------------------------
# feat_v6 training features (history in submission order)
# --------------------------------------------------------------------------
def _queue_len_submission(lat_us: np.ndarray, ts_submit_ms: np.ndarray) -> np.ndarray:
    """feat_v6.append_queue_len: 1 + earlier rows still in flight at submission."""
    finish = ts_submit_ms + lat_us / 1000.0
    # rows are sorted by ts_submit; count j<i with finish_j >= ts_submit_i
    return _qlen_numba(finish, ts_submit_ms)


@numba.njit(cache=True)
def _qlen_numba(finish, ts):
    n = finish.shape[0]
    out = np.empty(n, np.int64)
    # maintain a min-heap of finish times
    heap = np.empty(n, np.float64)
    hn = 0
    for i in range(n):
        t = ts[i]
        while hn > 0 and heap[0] < t:
            # pop min
            hn -= 1
            last = heap[hn]
            j = 0
            while True:
                c = 2 * j + 1
                if c >= hn:
                    break
                if c + 1 < hn and heap[c + 1] < heap[c]:
                    c += 1
                if heap[c] < last:
                    heap[j] = heap[c]
                    j = c
                else:
                    break
            if hn > 0:
                heap[j] = last
        # push own finish
        v = finish[i]
        j = hn
        hn += 1
        while j > 0:
            p = (j - 1) // 2
            if heap[p] > v:
                heap[j] = heap[p]
                j = p
            else:
                break
        heap[j] = v
        out[i] = hn
    return out


def feat_v6(labelled: pd.DataFrame) -> pd.DataFrame:
    """Heimdall's feat_v6 on a labelled frame (ts_submit-sorted, outliers removed).

    ``labelled`` columns: latency, io_type, size, ts_submit, reject.
    Returns the same columns as the artefact's mldrive*.csv.
    """
    df = labelled[["io_type", "size", "reject", "latency", "ts_submit"]].copy()
    df["queue_len"] = _queue_len_submission(df["latency"].to_numpy(np.float64),
                                            df["ts_submit"].to_numpy(np.float64))
    df = df.drop(columns=["ts_submit"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["throughput"] = df["size"] / df["latency"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["throughput"] = df["throughput"].round(0).astype(int)
    for col in ("queue_len", "latency", "throughput"):
        for i in range(1, N_HISTORY + 1):
            df[f"prev_{col}_{i}"] = df[col].shift(i).fillna(0).to_numpy().astype(np.int64)
    df = df.iloc[N_HISTORY:]
    df = df.drop(columns=["throughput"])
    cols = ["io_type", "size", "reject", "queue_len"] + [
        f"prev_{c}_{i}" for c in ("queue_len", "latency", "throughput") for i in range(1, N_HISTORY + 1)
    ] + ["latency"]
    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Deployment features (what io_replayer.c computes at decision time)
# --------------------------------------------------------------------------
@numba.njit(cache=True)
def _deploy_numba(ts_us, fin_us, lat, size, admitted, nhist):
    """Per request (rows sorted by submission time):
    queue length at submission (incl. itself) over admitted in-flight I/Os,
    history of the last ``nhist`` admitted completions before submission,
    and in-flight state: oldest outstanding age, time since last completion.
    """
    n = ts_us.shape[0]
    qlen = np.zeros(n, np.int64)
    hq = np.zeros((n, nhist), np.int64)
    hl = np.zeros((n, nhist), np.int64)
    ht = np.zeros((n, nhist), np.int64)
    oldest_age = np.zeros(n, np.float64)
    since_done = np.full(n, -1.0)
    n_inflight = np.zeros(n, np.int64)
    # completion events sorted by finish time
    comp_order = np.argsort(fin_us, kind="mergesort")
    # active set (small: bounded by replayer threads) stored as list arrays
    act_ts = np.empty(n, np.float64)
    act_fin = np.empty(n, np.float64)
    act_idx = np.empty(n, np.int64)
    na = 0
    ring_q = np.zeros(nhist, np.int64)
    ring_l = np.zeros(nhist, np.int64)
    ring_t = np.zeros(nhist, np.int64)
    nr = 0
    cp = 0
    last_fin = -1.0
    q_at = np.zeros(n, np.int64)
    for i in range(n):
        t = ts_us[i]
        # retire completions that happened strictly before t
        while cp < n and fin_us[comp_order[cp]] < t:
            j = comp_order[cp]
            cp += 1
            if not admitted[j]:
                continue
            # push into history ring (most recent first)
            for k in range(nhist - 1, 0, -1):
                ring_q[k] = ring_q[k - 1]
                ring_l[k] = ring_l[k - 1]
                ring_t[k] = ring_t[k - 1]
            ring_q[0] = q_at[j]
            ring_l[0] = lat[j]
            ring_t[0] = size[j] // (lat[j] + 1)
            if nr < nhist:
                nr += 1
            last_fin = fin_us[j]
        # drop finished from active set
        k = 0
        while k < na:
            if act_fin[k] < t:
                na -= 1
                act_ts[k] = act_ts[na]
                act_fin[k] = act_fin[na]
                act_idx[k] = act_idx[na]
            else:
                k += 1
        oa = 0.0
        for k in range(na):
            a = t - act_ts[k]
            if a > oa:
                oa = a
        oldest_age[i] = oa
        n_inflight[i] = na
        since_done[i] = t - last_fin if last_fin >= 0 else -1.0
        for k in range(nhist):
            if k < nr:
                hq[i, k] = ring_q[k]
                hl[i, k] = ring_l[k]
                ht[i, k] = ring_t[k]
        # io_replayer.c increments the queue length before inference, so the
        # model sees in-flight + 1 whether or not the request is then rerouted
        qlen[i] = na + 1
        if admitted[i]:
            q_at[i] = na + 1
            act_ts[na] = t
            act_fin[na] = fin_us[i]
            act_idx[na] = i
            na += 1
    return qlen, hq, hl, ht, oldest_age, since_done, n_inflight


def deploy_features(df: pd.DataFrame, admitted: np.ndarray | None = None) -> pd.DataFrame:
    """Decision-time features for every request of a replay log.

    ``df``: replay log (latency us, io_type, size, ts_submit ms), any order.
    ``admitted``: which requests were served by this device (default: all,
    i.e. a baseline replay).  Rows are returned sorted by ts_submit with an
    ``row`` column pointing back to ``df``.
    """
    order = np.argsort(df["ts_submit"].to_numpy(), kind="mergesort")
    ts_us = df["ts_submit"].to_numpy(np.float64)[order] * 1000.0
    lat = df["latency"].to_numpy(np.int64)[order]
    size = df["size"].to_numpy(np.int64)[order]
    fin_us = ts_us + lat
    adm = np.ones(len(order), np.bool_) if admitted is None else np.asarray(admitted, np.bool_)[order]
    qlen, hq, hl, ht, oldest, since, ninf = _deploy_numba(ts_us, fin_us, lat, size, adm, N_HISTORY)
    out = pd.DataFrame({
        "row": order,
        "ts_us": ts_us,
        "latency": lat,
        "io_type": df["io_type"].to_numpy(np.int64)[order],
        "size": size,
        "queue_len": qlen,
    })
    for k in range(N_HISTORY):
        out[f"prev_queue_len_{k+1}"] = hq[:, k]
    for k in range(N_HISTORY):
        out[f"prev_latency_{k+1}"] = hl[:, k]
    for k in range(N_HISTORY):
        out[f"prev_throughput_{k+1}"] = ht[:, k]
    out["oldest_age_us"] = oldest
    out["since_done_us"] = since
    out["n_inflight"] = ninf
    return out


# --------------------------------------------------------------------------
# The artefact's trained network
# --------------------------------------------------------------------------
class ArtefactModel:
    """12-128-16-1 ReLU MLP with MinMax input scaling, from nnK.py's CSV dump."""

    def __init__(self, weights_dir: Path, prefix: str):
        def load(name):
            return np.loadtxt(weights_dir / f"{prefix}.{name}.csv", delimiter=",", ndmin=1)
        self.x_min = load("weight_0")
        self.x_range = load("bias_0")
        self.W1 = load("weight_1").reshape(12, 128)
        self.b1 = load("bias_1")
        self.W2 = load("weight_2").reshape(128, 16)
        self.b2 = load("bias_2")
        self.W3 = load("weight_3").reshape(16, 1)
        self.b3 = load("bias_3").reshape(1)

    def hidden(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = (X - self.x_min) / self.x_range
        H1 = np.maximum(Z @ self.W1 + self.b1, 0)
        H2 = np.maximum(H1 @ self.W2 + self.b2, 0)
        return H1, H2

    def logit(self, X: np.ndarray) -> np.ndarray:
        _, H2 = self.hidden(X)
        return (H2 @ self.W3 + self.b3)[:, 0]


# --------------------------------------------------------------------------
# Exact fixed-point inference of flashnet_algo.c (weights from mlHeaderGen+2.py)
# --------------------------------------------------------------------------
class FixedPointModel:
    """Integer arithmetic exactly as flashnet_algo.c with mlHeaderGen+2.py headers.

    Header generation: w0 = round(min), b0 = round(2^30 / range),
    W1, W2, W3 scaled by 10^3; b1 by 10^3, b2 by 10^6, b3 by 10^9.
    Decision: final >= 0 -> reject (reroute).
    """

    def __init__(self, m: ArtefactModel):
        r = lambda a, s: np.array([int(round(float(v) * s)) for v in np.ravel(a)], np.int64)
        self.w0 = r(m.x_min, 1)
        self.b0 = np.array([int(round((2 << 29) / float(v))) for v in m.x_range], np.int64)
        self.W1 = r(m.W1.T, 1e3).reshape(128, 12)   # [j, input]
        self.b1 = r(m.b1, 1e3)
        self.W2 = r(m.W2.T, 1e3).reshape(16, 128)
        self.b2 = r(m.b2, 1e6)
        self.W3 = r(m.W3.T, 1e3).reshape(16)
        self.b3 = r(m.b3, 1e9)[0]

    def final(self, X: np.ndarray) -> np.ndarray:
        return _fixed_final(np.asarray(X, np.int64), self.w0, self.b0, self.W1, self.b1,
                            self.W2, self.b2, self.W3, self.b3)


@numba.njit(cache=True)
def _fixed_one(x, w0, b0, W1, b1, W2, b2, W3, b3):
    mid = np.empty(12, np.int64)
    for j in range(12):
        mid[j] = (x[j] - w0[j]) * b0[j]
    h1 = np.empty(128, np.int64)
    for j in range(128):
        s = 0
        for i in range(12):
            s += (mid[i] * W1[j, i]) >> 30
        s += b1[j]
        h1[j] = s if s > 0 else 0
    h2 = np.empty(16, np.int64)
    for j in range(16):
        s = 0
        for i in range(128):
            s += h1[i] * W2[j, i]
        s += b2[j]
        h2[j] = s if s > 0 else 0
    s = 0
    for i in range(16):
        s += h2[i] * W3[i]
    return s + b3


@numba.njit(cache=True)
def _fixed_final(X, w0, b0, W1, b1, W2, b2, W3, b3):
    n = X.shape[0]
    out = np.empty(n, np.int64)
    for k in range(n):
        out[k] = _fixed_one(X[k], w0, b0, W1, b1, W2, b2, W3, b3)
    return out


@numba.njit(cache=True)
def _reconstruct_numba(ts_us, lat, size, io_type, w0, b0, W1, b1, W2, b2, W3, b3, nhist):
    """Replay the client-side decision logic of io_replayer.c over a logged run.

    Requests in submission order; a read is rerouted when the fixed-point
    network says so.  Rerouted requests leave the queue length and history
    untouched (as in the C code).  Returns (rerouted mask, final scores).
    """
    n = ts_us.shape[0]
    rer = np.zeros(n, np.bool_)
    fin_score = np.zeros(n, np.int64)
    fin = ts_us + lat
    # pending completions of admitted I/Os: small array (<= replay threads)
    act_fin = np.empty(64, np.float64)
    act_q = np.empty(64, np.int64)
    act_lat = np.empty(64, np.int64)
    act_size = np.empty(64, np.int64)
    na = 0
    rq = np.zeros(nhist, np.int64)
    rl = np.zeros(nhist, np.int64)
    rt = np.zeros(nhist, np.int64)
    nr = 0
    x = np.zeros(12, np.int64)
    for i in range(n):
        t = ts_us[i]
        # retire admitted completions before t, in completion order
        while True:
            best = -1
            bf = 1e300
            for k in range(na):
                if act_fin[k] < t and act_fin[k] < bf:
                    bf = act_fin[k]
                    best = k
            if best < 0:
                break
            for h in range(nhist - 1, 0, -1):
                rq[h] = rq[h - 1]; rl[h] = rl[h - 1]; rt[h] = rt[h - 1]
            rq[0] = act_q[best]; rl[0] = act_lat[best]; rt[0] = act_size[best] // (act_lat[best] + 1)
            if nr < nhist:
                nr += 1
            na -= 1
            act_fin[best] = act_fin[na]; act_q[best] = act_q[na]
            act_lat[best] = act_lat[na]; act_size[best] = act_size[na]
        q = na + 1
        admit = True
        if io_type[i] == 1:
            x[0] = 1; x[1] = size[i]; x[2] = q
            for h in range(nhist):
                if h < nr:
                    x[3 + h] = rq[h]; x[6 + h] = rl[h]; x[9 + h] = rt[h]
                else:
                    x[3 + h] = 0; x[6 + h] = 0; x[9 + h] = 0
            s = _fixed_one(x, w0, b0, W1, b1, W2, b2, W3, b3)
            fin_score[i] = s
            if s >= 0:
                admit = False
                rer[i] = True
        if admit:
            if na >= 64:
                # should not happen with 8 replay threads; overwrite the newest slot
                na = 63
            act_fin[na] = fin[i]; act_q[na] = q; act_lat[na] = lat[i]; act_size[na] = size[i]
            na += 1
    return rer, fin_score


def reconstruct_decisions(df: pd.DataFrame, fpm: FixedPointModel):
    """Which reads Heimdall rerouted in a logged flashnet run (sorted by ts_submit)."""
    order = np.argsort(df["ts_submit"].to_numpy(), kind="mergesort")
    ts_us = df["ts_submit"].to_numpy(np.float64)[order] * 1000.0
    lat = df["latency"].to_numpy(np.int64)[order]
    size = df["size"].to_numpy(np.int64)[order]
    io = df["io_type"].to_numpy(np.int64)[order]
    rer, score = _reconstruct_numba(ts_us, lat, size, io, fpm.w0, fpm.b0, fpm.W1, fpm.b1,
                                    fpm.W2, fpm.b2, fpm.W3, fpm.b3, N_HISTORY)
    return order, rer, score


# --------------------------------------------------------------------------
# Online labeller state: tail_v1 run incrementally on what is known at
# decision time (thresholds from the training window only)
# --------------------------------------------------------------------------
@numba.njit(cache=True)
def _online_tail(ts_us, lat, size, admitted, ip_lat, ip_thpt, med_thpt, drop_rate):
    """For each request (submission order): is tail_v1's period open, judged on
    the prefix of requests whose outcome is already known, and how many known
    requests ago did it open/close.  The scan follows tail_v1 exactly; a period
    end needs the next two throughputs, so the last two known requests can
    only keep a period open, never close it (no look-ahead)."""
    n = ts_us.shape[0]
    fin = ts_us + lat
    thpt = np.empty(n)
    for k in range(n):
        thpt[k] = np.round(size[k] / lat[k]) if lat[k] > 0 else 1e18
    open_now = np.zeros(n, np.bool_)
    run_len = np.zeros(n, np.int64)
    # consumed prefix over admitted requests in submission order
    adm_idx = np.empty(n, np.int64)
    na = 0
    for k in range(n):
        if admitted[k]:
            adm_idx[na] = k
            na += 1
    p = 0              # next admitted request (position in adm_idx) not yet known
    scan = 0           # next position the tail_v1 scanner has to examine
    in_tail = False
    since_change = 0
    prev_thpt = 0.0
    for i in range(n):
        t = ts_us[i]
        # extend the known prefix: admitted requests completed before t, in order
        while p < na and fin[adm_idx[p]] < t and adm_idx[p] < i:
            p += 1
        # run the scanner as far as the known prefix allows
        while scan < p:
            k = adm_idx[scan]
            if not in_tail:
                pt = thpt[adm_idx[scan - 1]] if scan > 0 else 0.0
                drop = np.round(pt / (thpt[k] + 0.1), 1)
                if (not (thpt[k] > ip_thpt or lat[k] < ip_lat)) and drop >= drop_rate:
                    in_tail = True
                    since_change = 0
                else:
                    since_change += 1
                scan += 1
            else:
                # end check needs this and the next two known throughputs
                if scan + 2 >= p:
                    break
                ok = True
                for q in range(3):
                    if thpt[adm_idx[scan + q]] < med_thpt:
                        ok = False
                if ok:
                    in_tail = False
                    since_change = 0
                    scan += 1            # tail_v1 skips the end row
                else:
                    since_change += 1
                    scan += 1
        open_now[i] = in_tail
        run_len[i] = since_change
    return open_now, run_len


def online_tail_state(df: pd.DataFrame, lab0: "TailLabels", admitted=None):
    """Online tail_v1 state at each request's submission (rows sorted by
    ts_submit, as deploy_features returns them)."""
    order = np.argsort(df["ts_submit"].to_numpy(), kind="mergesort")
    ts_us = df["ts_submit"].to_numpy(np.float64)[order] * 1000.0
    lat = df["latency"].to_numpy(np.float64)[order]
    size = df["size"].to_numpy(np.float64)[order]
    adm = np.ones(len(order), np.bool_) if admitted is None else np.asarray(admitted, np.bool_)[order]
    o, r = _online_tail(ts_us, lat, size, adm, float(lab0.ip_latency), float(lab0.ip_throughput),
                        float(lab0.median_throughput), THPT_DROP_RATE)
    return o, r


# --------------------------------------------------------------------------
# Outcome feedback: what the client has already observed about its reads
# --------------------------------------------------------------------------
@numba.njit(cache=True)
def _read_feedback(ts, lat, isread, size, T, ks):
    """Per request (submission order), from outcomes already observed:
    fraction of the last k completed reads slower than T (each k in ks; -1 if
    none), the same over reads of the request's own size class (log2 bucket,
    last 8), the last completed read's latency, the oldest outstanding read's
    age, and the count and bytes of outstanding writes."""
    n = ts.shape[0]
    fin = ts + lat
    order = np.argsort(fin, kind="mergesort")
    K = ks.max()
    ring = np.full(K, -1.0)
    NC, KC = 32, 8
    cring = np.full((NC, KC), -1.0)
    frac = np.full((n, ks.shape[0]), -1.0)
    cfrac = np.full(n, -1.0)
    last = np.full(n, -1.0)
    oldest_read = np.zeros(n)
    out_w = np.zeros(n, np.int64)
    out_wb = np.zeros(n)
    done = np.zeros(n, np.bool_)
    cp = 0
    head = 0
    nw = 0
    wb = 0.0
    for i in range(n):
        t = ts[i]
        while cp < n and fin[order[cp]] < t:
            j = order[cp]
            cp += 1
            if j >= i:
                continue
            done[j] = True
            if isread[j]:
                for k in range(K - 1, 0, -1):
                    ring[k] = ring[k - 1]
                ring[0] = lat[j]
                c = min(NC - 1, max(0, int(np.log2(max(size[j], 512.0)))))
                for k in range(KC - 1, 0, -1):
                    cring[c, k] = cring[c, k - 1]
                cring[c, 0] = lat[j]
            else:
                nw -= 1
                wb -= size[j]
        while head < i and (done[head] or not isread[head]):
            head += 1
        oldest_read[i] = t - ts[head] if head < i else 0.0
        last[i] = ring[0]
        for q in range(ks.shape[0]):
            cnt = 0
            sl = 0
            for k in range(ks[q]):
                if ring[k] >= 0:
                    cnt += 1
                    if ring[k] > T:
                        sl += 1
            if cnt > 0:
                frac[i, q] = sl / cnt
        c = min(NC - 1, max(0, int(np.log2(max(size[i], 512.0)))))
        cnt = 0
        sl = 0
        for k in range(KC):
            if cring[c, k] >= 0:
                cnt += 1
                if cring[c, k] > T:
                    sl += 1
        if cnt > 0:
            cfrac[i] = sl / cnt
        out_w[i] = nw
        out_wb[i] = wb
        if not isread[i]:
            nw += 1
            wb += size[i]
    return frac, cfrac, last, oldest_read, out_w, out_wb


def read_feedback(df: pd.DataFrame, T: float, ks=(4, 16)):
    """Outcome-feedback evidence at each request's submission (ts_submit order)."""
    order = np.argsort(df["ts_submit"].to_numpy(), kind="mergesort")
    ts = df["ts_submit"].to_numpy(np.float64)[order] * 1000.0
    lat = df["latency"].to_numpy(np.float64)[order]
    isread = df["io_type"].to_numpy()[order] == 1
    size = df["size"].to_numpy(np.float64)[order]
    return _read_feedback(ts, lat, isread, size, float(T), np.array(ks, np.int64))


@numba.njit(cache=True)
def _neighbourhood(ts, lat, isread, T, K):
    """Share of the last K submitted I/Os (all, and reads only) already known
    to be slow at this submission: completed with latency > T, or still
    outstanding and older than T (a censored slow outcome)."""
    n = ts.shape[0]
    fin = ts + lat
    q_all = np.zeros(n)
    q_rd = np.full(n, -1.0)
    for i in range(n):
        t = ts[i]
        s_all = 0
        c_all = 0
        s_rd = 0
        c_rd = 0
        j = i - 1
        while j >= 0 and c_all < K:
            known_slow = (lat[j] > T) if fin[j] < t else ((t - ts[j]) > T)
            c_all += 1
            if known_slow:
                s_all += 1
            if isread[j] and c_rd < K:
                c_rd += 1
                if known_slow:
                    s_rd += 1
            j -= 1
        q_all[i] = s_all / c_all if c_all > 0 else 0.0
        if c_rd > 0:
            q_rd[i] = s_rd / c_rd
    return q_all, q_rd


def neighbourhood(df: pd.DataFrame, T: float, K: int = 8):
    order = np.argsort(df["ts_submit"].to_numpy(), kind="mergesort")
    ts = df["ts_submit"].to_numpy(np.float64)[order] * 1000.0
    lat = df["latency"].to_numpy(np.float64)[order]
    isread = df["io_type"].to_numpy()[order] == 1
    return _neighbourhood(ts, lat, isread, float(T), int(K))
