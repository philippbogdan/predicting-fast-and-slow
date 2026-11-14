"""The same evaluation on other learned systems decisions.

A transfer task provides, per "trace" (an independent stream such as one
microservice or one storage server), every request with:
  resource (int): the shared resource that serves it (pod, disk, ...)
  start_us, end_us: when it was issued and when its outcome was observed
  decision (bool): whether the controller makes a decision for it
  static features (task-specific numeric columns known at issue time)
The learned controller is a LinnOS/Heimdall-style per-request slowness
classifier: label = latency above the inflection point (Heimdall's
tangent_based) of the training window's latencies; inputs = static features,
outstanding count at the resource and the last three completions there
(latency and outstanding count), exactly as io_replayer.c feeds Heimdall.
Signals are scored exactly as on the SSD traces (scripts/run_detection.py).
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from .heimdall import tangent_based

N_HIST = 3


@numba.njit(cache=True)
def _live_state(res, start, end, n_res, nhist):
    """Per request (sorted by start): decision-time state of its resource."""
    n = start.shape[0]
    qlen = np.zeros(n, np.int64)
    hl = np.zeros((n, nhist), np.float64)
    hq = np.zeros((n, nhist), np.int64)
    oldest = np.zeros(n, np.float64)
    since = np.full(n, -1.0)
    ninf = np.zeros(n, np.int64)
    # completions processed in end-time order
    corder = np.argsort(end, kind="mergesort")
    cp = 0
    # per-resource ring history
    r_hl = np.zeros((n_res, nhist), np.float64)
    r_hq = np.zeros((n_res, nhist), np.int64)
    r_nh = np.zeros(n_res, np.int64)
    r_last = np.full(n_res, -1.0)
    # per-resource in-flight: count and a lazy min-heap of start times keyed by request
    r_inf = np.zeros(n_res, np.int64)
    q_at = np.zeros(n, np.int64)
    done = np.zeros(n, np.bool_)
    # per-resource FIFO of started requests (by start); oldest outstanding = first not done
    nxt = np.full(n, -1, np.int64)
    head = np.full(n_res, -1, np.int64)
    tail = np.full(n_res, -1, np.int64)
    for i in range(n):
        t = start[i]
        while cp < n and end[corder[cp]] < t:
            j = corder[cp]
            cp += 1
            if start[j] >= t:
                continue
            r = res[j]
            done[j] = True
            r_inf[r] -= 1
            for h in range(nhist - 1, 0, -1):
                r_hl[r, h] = r_hl[r, h - 1]
                r_hq[r, h] = r_hq[r, h - 1]
            r_hl[r, 0] = end[j] - start[j]
            r_hq[r, 0] = q_at[j]
            if r_nh[r] < nhist:
                r_nh[r] += 1
            r_last[r] = end[j]
        r = res[i]
        # advance this resource's FIFO head past completed requests
        h = head[r]
        while h >= 0 and done[h]:
            h = nxt[h]
        head[r] = h
        if h < 0:
            tail[r] = -1
        oldest[i] = t - start[h] if h >= 0 else 0.0
        ninf[i] = r_inf[r]
        since[i] = t - r_last[r] if r_last[r] >= 0 else -1.0
        qlen[i] = r_inf[r] + 1
        q_at[i] = r_inf[r] + 1
        for k in range(nhist):
            if k < r_nh[r]:
                hl[i, k] = r_hl[r, k]
                hq[i, k] = r_hq[r, k]
        # admit i to its resource
        r_inf[r] += 1
        if tail[r] >= 0:
            nxt[tail[r]] = i
        else:
            head[r] = i
        tail[r] = i
    return qlen, hl, hq, oldest, since, ninf


def live_state(df: pd.DataFrame) -> pd.DataFrame:
    """Adds decision-time state columns; df must be sorted by start_us."""
    res_codes, _ = pd.factorize(df["resource"])
    qlen, hl, hq, oldest, since, ninf = _live_state(
        res_codes.astype(np.int64), df["start_us"].to_numpy(np.float64), df["end_us"].to_numpy(np.float64),
        int(res_codes.max()) + 1, N_HIST)
    out = df.copy()
    out["queue_len"] = qlen
    for k in range(N_HIST):
        out[f"prev_latency_{k+1}"] = hl[:, k]
    for k in range(N_HIST):
        out[f"prev_queue_len_{k+1}"] = hq[:, k]
    out["oldest_age_us"] = oldest
    out["since_done_us"] = since
    out["n_inflight"] = ninf
    return out


def ip_threshold(latency: np.ndarray) -> float:
    return float(tangent_based(np.asarray(latency, np.float64))[0])


@numba.njit(cache=True)
def _kind_feedback(res, kind, start, end, T, n_res, n_kind, K):
    """Fraction of the last K completed requests of the same kind slower than
    T, on the same resource and on any resource (-1 if none yet)."""
    n = start.shape[0]
    corder = np.argsort(end, kind="mergesort")
    ring_any = np.full((n_kind, K), -1.0)
    # per (resource, kind) rings, hashed into a fixed table to bound memory
    H = 1 << 16
    ring_rk = np.full((H, K), -1.0)
    out_any = np.full(n, -1.0)
    out_rk = np.full(n, -1.0)
    cp = 0
    for i in range(n):
        t = start[i]
        while cp < n and end[corder[cp]] < t:
            j = corder[cp]
            cp += 1
            if start[j] >= t:
                continue
            k = kind[j]
            for q in range(K - 1, 0, -1):
                ring_any[k, q] = ring_any[k, q - 1]
            ring_any[k, 0] = end[j] - start[j]
            h = (res[j] * 1000003 + k) % H
            for q in range(K - 1, 0, -1):
                ring_rk[h, q] = ring_rk[h, q - 1]
            ring_rk[h, 0] = end[j] - start[j]
        k = kind[i]
        c = 0
        s = 0
        for q in range(K):
            if ring_any[k, q] >= 0:
                c += 1
                if ring_any[k, q] > T:
                    s += 1
        if c > 0:
            out_any[i] = s / c
        h = (res[i] * 1000003 + k) % H
        c = 0
        s = 0
        for q in range(K):
            if ring_rk[h, q] >= 0:
                c += 1
                if ring_rk[h, q] > T:
                    s += 1
        if c > 0:
            out_rk[i] = s / c
    return out_any, out_rk


def kind_feedback(df: pd.DataFrame, kind_col: str, T: float, K: int = 8):
    res_codes, _ = pd.factorize(df["resource"])
    kind_codes, _ = pd.factorize(df[kind_col])
    return _kind_feedback(res_codes.astype(np.int64), kind_codes.astype(np.int64),
                          df["start_us"].to_numpy(np.float64), df["end_us"].to_numpy(np.float64), float(T),
                          int(res_codes.max()) + 1, int(kind_codes.max()) + 1, K)


@numba.njit(cache=True)
def _neighbourhood_res(res, start, end, T, n_res, K):
    """Per request: share of the last K requests submitted to the same
    resource that are already known slow (completed above T, or still
    outstanding past T)."""
    n = start.shape[0]
    ring_s = np.full((n_res, K), -1.0)
    ring_e = np.full((n_res, K), -1.0)
    pos = np.zeros(n_res, np.int64)
    cnt = np.zeros(n_res, np.int64)
    out = np.zeros(n)
    for i in range(n):
        r = res[i]
        t = start[i]
        s = 0
        c = min(cnt[r], K)
        for k in range(c):
            st, en = ring_s[r, k], ring_e[r, k]
            if (en < t and en - st > T) or (en >= t and t - st > T):
                s += 1
        out[i] = s / c if c > 0 else 0.0
        ring_s[r, pos[r]] = start[i]
        ring_e[r, pos[r]] = end[i]
        pos[r] = (pos[r] + 1) % K
        cnt[r] += 1
    return out


def neighbourhood_res(df: pd.DataFrame, T: float, K: int = 8):
    res_codes, _ = pd.factorize(df["resource"])
    return _neighbourhood_res(res_codes.astype(np.int64), df["start_us"].to_numpy(np.float64),
                              df["end_us"].to_numpy(np.float64), float(T), int(res_codes.max()) + 1, int(K))


# --------------------------------------------------------------------------
# Rich decision-time information for the transfer ceilings
# --------------------------------------------------------------------------
@numba.njit(cache=True)
def _rich(res, kind, start, end, n_res, n_kind, K):
    """Per request (sorted by start), from what is known at its start:
    for its kind (file / interface): time since the last request of that kind
    started, since the last one completed, that completion's latency, how many
    of that kind were seen; for its resource: the last K completed latencies
    and how long ago each completed."""
    n = start.shape[0]
    corder = np.argsort(end, kind="mergesort")
    k_last_start = np.full(n_kind, -1.0)
    k_last_end = np.full(n_kind, -1.0)
    k_last_lat = np.full(n_kind, -1.0)
    k_count = np.zeros(n_kind)
    r_lat = np.full((n_res, K), -1.0)
    r_end = np.full((n_res, K), -1.0)
    X = np.full((n, 4 + 2 * K), -1.0)
    cp = 0
    for i in range(n):
        t = start[i]
        while cp < n and end[corder[cp]] < t:
            j = corder[cp]
            cp += 1
            if start[j] >= t:
                continue
            kk = kind[j]
            k_last_end[kk] = end[j]
            k_last_lat[kk] = end[j] - start[j]
            r = res[j]
            for q in range(K - 1, 0, -1):
                r_lat[r, q] = r_lat[r, q - 1]
                r_end[r, q] = r_end[r, q - 1]
            r_lat[r, 0] = end[j] - start[j]
            r_end[r, 0] = end[j]
        kk = kind[i]
        r = res[i]
        if k_last_start[kk] >= 0:
            X[i, 0] = t - k_last_start[kk]
        if k_last_end[kk] >= 0:
            X[i, 1] = t - k_last_end[kk]
            X[i, 2] = k_last_lat[kk]
        X[i, 3] = k_count[kk]
        for q in range(K):
            if r_end[r, q] >= 0:
                X[i, 4 + 2 * q] = r_lat[r, q]
                X[i, 5 + 2 * q] = t - r_end[r, q]
        k_last_start[kk] = t
        k_count[kk] += 1
    return X


def rich_info(df: pd.DataFrame, kind_col: str, K: int = 8) -> np.ndarray:
    res_codes, _ = pd.factorize(df["resource"])
    kind_codes, _ = pd.factorize(df[kind_col])
    return _rich(res_codes.astype(np.int64), kind_codes.astype(np.int64), df["start_us"].to_numpy(np.float64),
                 df["end_us"].to_numpy(np.float64), int(res_codes.max()) + 1, int(kind_codes.max()) + 1, K)
