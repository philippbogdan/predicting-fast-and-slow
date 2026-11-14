"""The situation tracker: label-free evidence of whether a request will be slow.

Per resource, a table of running averages, one per *situation cell*:
  size class  c = floor(log2(request bytes)), capped at 31
  load class  b = bit length of floor(outstanding bytes / 4 KiB), capped at 15
  age class   a = floor(oldest outstanding request's age / T), capped at 3
  quiet bit   z = 1 if a request has completed on the resource but none in the
              last 16 T (a burst after an idle period on a power-managed
              SSD; its first requests are outstanding, so the age class
              alone cannot see it)
Every cell starts at 1/2.  When a request completes, the cell it was
submitted in moves 1/64 of the way towards 1 if its latency exceeded T, else
towards 0.  The evidence for a new request is its cell's average q; the model
is distrusted when it says "fast" and q >= 1/2, or says "slow" and q < 1/2.

T is the latency that counts as slow: the knee of the training window's
read-latency CDF (tangent method, as LinnOS/Heimdall compute it), so the
tracker needs no labelled requests: it learns from the outcomes the system
observes.  This module is the reference implementation; admission.sim,
bench/gate_cost.c and docs/rule.md state the same rule.
"""
from __future__ import annotations

import numba
import numpy as np

N_SIZE, N_LOAD, N_AGE, N_QUIET = 32, 16, 4, 2
QUIET = 16.0   # the quiet bit is set when the last completion is more than QUIET * T ago
N_CELLS = N_SIZE * N_LOAD * N_AGE * N_QUIET
ALPHA = 1.0 / 64.0


@numba.njit(cache=True, inline="always")
def cell_of(size, out_bytes, oldest_age, since_done, T):
    """since_done: time since the last completion on the resource (< 0: none yet)."""
    c = 0
    s = int(size)
    while s > 1 and c < N_SIZE - 1:
        s >>= 1
        c += 1
    x = int(out_bytes) >> 12
    b = 0
    while x > 0 and b < N_LOAD - 1:
        x >>= 1
        b += 1
    a = int(oldest_age / T) if T > 0 else 0
    if a > N_AGE - 1:
        a = N_AGE - 1
    if a < 0:
        a = 0
    z = 1 if (since_done >= 0 and since_done > QUIET * T) else 0
    return ((c * N_LOAD + b) * N_AGE + a) * N_QUIET + z


@numba.njit(cache=True)
def cells(size, out_bytes, oldest_age, since_done, T):
    n = size.shape[0]
    out = np.empty(n, np.int64)
    for i in range(n):
        out[i] = cell_of(size[i], out_bytes[i], oldest_age[i], since_done[i], T)
    return out


@numba.njit(cache=True)
def track(ts, lat, cell, slow, observed, alpha):
    """Evidence q at each request's submission (rows in submission order) from
    the outcomes of earlier requests that completed strictly before it and
    whose outcome the system observed."""
    n = ts.shape[0]
    fin = ts + lat
    order = np.argsort(fin, kind="mergesort")
    tab = np.full(N_CELLS, 0.5)
    q = np.empty(n)
    cp = 0
    for i in range(n):
        t = ts[i]
        while cp < n and fin[order[cp]] < t:
            j = order[cp]
            cp += 1
            if observed[j]:
                tab[cell[j]] += alpha * (slow[j] - tab[cell[j]])
        q[i] = tab[cell[i]]
    return q


@numba.njit(cache=True)
def live_load(ts, fin, size):
    """Outstanding bytes and oldest outstanding age at each submission over
    all earlier requests still in flight (rows in submission order)."""
    n = ts.shape[0]
    byt = np.zeros(n)
    old = np.zeros(n)
    act = np.empty(n, np.int64)
    na = 0
    for i in range(n):
        t = ts[i]
        k = 0
        while k < na:
            if fin[act[k]] < t:
                na -= 1
                act[k] = act[na]
            else:
                k += 1
        for k in range(na):
            j = act[k]
            byt[i] += size[j]
            a = t - ts[j]
            if a > old[i]:
                old[i] = a
        act[na] = i
        na += 1
    return byt, old


def situation_age(oldest_age, out_bytes, since_done):
    """Variant (not in the rule): with nothing outstanding, use the time since
    the last completion as the age (0 if none), to capture slow wake-up reads
    after idle periods on power-managed consumer SSDs."""
    since = np.where(np.asarray(since_done, np.float64) < 0, 0.0, np.asarray(since_done, np.float64))
    return np.where(np.asarray(out_bytes, np.float64) > 0, np.asarray(oldest_age, np.float64), since)


def evidence(ts, lat, size, out_bytes, oldest_age, since_done, T, observed=None, alpha=ALPHA, is_target=None):
    """q per request.  ``since_done``: time since the resource's last completion
    (< 0: none yet; all < 0 gives the tracker without its quiet bit).
    ``is_target``: which rows update and read the table (default all);
    load/age/quiet come from all rows."""
    ts = np.asarray(ts, np.float64)
    lat = np.asarray(lat, np.float64)
    n = len(ts)
    sel = np.ones(n, np.bool_) if is_target is None else np.asarray(is_target, np.bool_)
    obs = np.ones(n, np.bool_) if observed is None else np.asarray(observed, np.bool_)
    c = cells(np.asarray(size, np.float64)[sel], np.asarray(out_bytes, np.float64)[sel],
              np.asarray(oldest_age, np.float64)[sel], np.asarray(since_done, np.float64)[sel], float(T))
    slow = (lat[sel] > T).astype(np.float64)
    q = np.full(n, np.nan)
    q[sel] = track(ts[sel], lat[sel], c, slow, obs[sel], float(alpha))
    return q


def mistake_score(q, pred_slow):
    """Higher = more likely the model's decision is wrong."""
    return np.where(pred_slow, 1.0 - q, q)
