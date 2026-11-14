"""Inputs for admission.sim: raw-trace order, baseline work, read templates."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .heimdall import ArtefactModel
from .sim import find_stalls, work_from_baseline, work_from_baseline_stalls
from .traces import DEVICES, PARQUET_ROOT, RAW_ONLY_UNITS, DeviceTrace, RawOnlyTrace


def duration_s(t) -> float:
    if isinstance(t, RawOnlyTrace):
        return float(t.duration_s)
    txt = (t.pair_dir / f"trace_{t.dev + 1}.stats").read_text()
    return float(re.search(r"Duration\s*=.*=\s*([0-9.]+) s", txt).group(1))


def load_raw(t) -> pd.DataFrame:
    if isinstance(t, RawOnlyTrace):
        df = pd.read_csv(t.raw_trace, sep=" ", header=None, names=["ts", "dev", "offset", "size", "io_type"])
        if RAW_ONLY_UNITS[t.name] == "us":
            df["ts"] = df["ts"] / 1000.0
        df["size"] = (df["size"] + 511) // 512 * 512      # our replays log 512-aligned sizes
        return df
    return pd.read_parquet(PARQUET_ROOT / t.pair / "raw" / f"d{t.dev}.parquet")


def match_log_to_raw(raw: pd.DataFrame, log: pd.DataFrame, duration_ms: float) -> np.ndarray:
    """Raw-trace index of every log row (-1 if unmatched).

    Key: trace timestamp as the replayer stores it (float32, printed %.3f),
    size, io type; equal keys are paired in order (raw index vs submission).
    """
    rt = np.round(raw["ts"].to_numpy(np.float32).astype(np.float64) * 1000).astype(np.int64)
    lt = np.round(log["ts_record"].to_numpy(np.float64) * 1000).astype(np.int64)
    sub = log["ts_submit"].to_numpy(np.float64).copy()
    sub[sub > duration_ms * 1.5] /= 1000.0  # hedged rows log the submission in us
    rk = pd.DataFrame({"t": rt, "s": raw["size"].to_numpy(), "o": raw["io_type"].to_numpy(),
                       "i": np.arange(len(raw))})
    lk = pd.DataFrame({"t": lt, "s": log["size"].to_numpy(), "o": log["io_type"].to_numpy(),
                       "sub": sub, "r": np.arange(len(log))})
    rk["n"] = rk.groupby(["t", "s", "o"]).cumcount()
    lk = lk.sort_values(["t", "s", "o", "sub"], kind="mergesort")
    lk["n"] = lk.groupby(["t", "s", "o"]).cumcount()
    m = lk.merge(rk, on=["t", "s", "o", "n"], how="left")
    out = np.full(len(log), -1, np.int64)
    ok = m["i"].notna().to_numpy()
    out[m["r"].to_numpy()[ok]] = m["i"].to_numpy()[ok].astype(np.int64)
    return out


@dataclass
class ClientInputs:
    arr_us: np.ndarray
    size: np.ndarray
    io: np.ndarray
    work: np.ndarray       # own-device work per raw index (cyclic beyond the baseline prefix)
    rd_idx: np.ndarray
    rd_work: np.ndarray
    rd_size: np.ndarray
    k_base: int            # baseline-issued prefix length
    base_lat: np.ndarray   # baseline latency per raw index (<k_base), -1 otherwise
    base_sub_us: np.ndarray  # submit time in the artefact's baseline replay per raw index (us), NaN if not issued;
                             # the windows of the protocol are defined on this timeline (as in the detection)
    hedge_t_us: float
    model: ArtefactModel
    duration_us: float
    st_k: np.ndarray       # device stalls: own-I/O raw index at which each begins
    st_d: np.ndarray       # and its duration (us)


def client_inputs(t, P: float, rep: int = 0, stall_G: float = 0.0) -> ClientInputs:
    raw = load_raw(t)
    dur = duration_s(t)
    base = pd.read_parquet(t.replay_parquet("baseline", rep) if isinstance(t, RawOnlyTrace)
                           else t.replay_parquet("baseline"))
    idx = match_log_to_raw(raw, base, dur * 1000.0)
    ok = idx >= 0
    lat = np.full(len(raw), -1.0)
    sub = np.full(len(raw), np.nan)
    lat[idx[ok]] = base["latency"].to_numpy(np.float64)[ok]
    sub[idx[ok]] = base["ts_submit"].to_numpy(np.float64)[ok] * 1000.0
    issued = np.flatnonzero(lat >= 0)
    k_base = int(issued.max()) + 1 if len(issued) else 0
    # work for issued I/Os under P-lane processor sharing of the baseline concurrency
    o = issued[np.argsort(sub[issued], kind="mergesort")]
    w = np.full(len(raw), np.nan)
    st_k = np.zeros(0, np.int64)
    st_d = np.zeros(0)
    if stall_G > 0:
        # stall-aware device: stalls are device time, triggered at the same point of the own workload
        a, b = find_stalls(sub[o], lat[o], float(stall_G))
        keep = b > a
        a, b = a[keep], b[keep]
        w[o] = work_from_baseline_stalls(sub[o], lat[o], float(P), a, b)
        # own-I/O position at which each stall began: first own I/O issued at/after its start
        pos = np.searchsorted(sub[o], a, side="left")
        ok = pos < len(o)
        st_k = o[pos[ok]].astype(np.int64)
        st_d = (b - a)[ok]
        srt = np.argsort(st_k, kind="mergesort")
        st_k, st_d = st_k[srt], st_d[srt]
    else:
        w[o] = work_from_baseline(sub[o], lat[o], float(P))
    # holes inside the prefix (unmatched rows): nearest issued I/O of the same type
    io = raw["io_type"].to_numpy(np.int64)
    size = raw["size"].to_numpy(np.int64)
    for typ in (0, 1):
        m = (io == typ)
        have = np.flatnonzero(m & ~np.isnan(w))
        need = np.flatnonzero(m & np.isnan(w) & (np.arange(len(raw)) < k_base))
        if len(have) and len(need):
            j = np.clip(np.searchsorted(have, need), 0, len(have) - 1)
            w[need] = w[have[j]]
    # cyclic extension beyond the baseline prefix (same type, cycling over the prefix)
    if k_base < len(raw):
        ext = np.arange(k_base, len(raw))
        w[ext] = w[ext % max(k_base, 1)]
        bad = np.isnan(w)
        w[bad] = np.nanmedian(w)
        if len(st_k):
            # stalls repeat with the work, cyclically over the baseline prefix
            reps_ = int(np.ceil(len(raw) / max(k_base, 1)))
            st_k = np.concatenate([st_k + r * k_base for r in range(reps_)])
            st_d = np.concatenate([st_d] * reps_)
            m_ = st_k < len(raw)
            st_k, st_d = st_k[m_], st_d[m_]
    rd = np.flatnonzero(io == 1)
    reads_b = base[base["io_type"] == 1]["latency"].to_numpy()
    model = ArtefactModel(t.artefact_weights_dir(), f"mldrive{t.dev}.csv") if isinstance(t, DeviceTrace) else None
    return ClientInputs(
        arr_us=raw["ts"].to_numpy(np.float64) * 1000.0, size=size, io=io, work=w,
        rd_idx=rd.astype(np.int64), rd_work=w[rd], rd_size=size[rd].astype(np.float64),
        k_base=k_base, base_lat=lat, base_sub_us=sub, hedge_t_us=float(np.percentile(reads_b, 98)),
        model=model, duration_us=dur * 1e6, st_k=st_k, st_d=st_d)


def size_alpha(t) -> float:
    """Bytes-equivalent fixed cost of a read: fit median latency = c + size/B."""
    base = pd.read_parquet(t.replay_parquet("baseline"), columns=["latency", "size", "io_type"])
    r = base[base["io_type"] == 1]
    g = r.groupby("size")["latency"].agg(["median", "size"])
    g = g[g["size"] >= 200]
    if len(g) < 2:
        return 65536.0
    x = g.index.to_numpy(np.float64)
    y = g["median"].to_numpy(np.float64)
    wts = np.sqrt(g["size"].to_numpy(np.float64))
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A * wts[:, None], y * wts, rcond=None)
    c, inv_b = coef
    if inv_b <= 0 or c <= 0:
        return 65536.0
    return float(c / inv_b)


def replay_latency_by_index(t: DeviceTrace, policy: str, n_raw: int, raw: pd.DataFrame) -> np.ndarray:
    log = pd.read_parquet(t.replay_parquet(policy))
    idx = match_log_to_raw(raw, log, duration_s(t) * 1000.0)
    lat = np.full(n_raw, -1.0)
    ok = idx >= 0
    lat[idx[ok]] = log["latency"].to_numpy(np.float64)[ok]
    return lat
