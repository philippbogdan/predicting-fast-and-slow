"""Decision-time evidence signals (candidates devised here).

The model sees the device only through its three most recent completions.
At decision time the client also knows what is happening *now*: how long its
oldest outstanding I/O has been waiting and how recently anything completed.
An outstanding I/O older than the slow threshold is already a slow outcome,
observed before it completes.  The candidates below score how strongly this
live evidence contradicts the model's decision.

All but frozen100 (which uses 100 labelled reads) need no labels: their
constants are latency thresholds from the training window.
"""
from __future__ import annotations

import numpy as np

from . import tracker as trk


def evidence_scores(R, logit: np.ndarray, ytr_latency_threshold: float) -> dict:
    T = float(max(ytr_latency_threshold, 1.0))
    age = R["oldest_age_us"].to_numpy(np.float64)
    since = R["since_done_us"].to_numpy(np.float64)
    since = np.where(since < 0, 1e9, since)
    lat1 = R["prev_latency_1"].to_numpy(np.float64)
    pred_slow = logit > 0
    p = 1 / (1 + np.exp(-logit))
    # live slowness evidence in [0, 1]: oldest outstanding age relative to T
    live = np.clip(age / (2 * T), 0, 1)
    out = {}
    if "tail_open" in R:
        # the labeller, run on what is already known, says a slow period is
        # open (or not); an outstanding I/O older than T is a known slow outcome
        tail = R["tail_open"].to_numpy().astype(bool) | (age > T)
        contra = np.where(pred_slow, ~tail, tail)
        unc = 0.5 - np.abs(p - 0.5)
        out["labeller_contra"] = contra + unc
        out["labeller_contra_only"] = contra.astype(np.float64)
    if "fb_frac16" in R:
        # outcome feedback: share of the last completed reads that were slow,
        # raised to 1 when an outstanding read is already older than T_read
        Tr = float(R.attrs.get("T_read_w0", T))
        ores = R["oldest_read_age_us"].to_numpy(np.float64)
        for k in (4, 16):
            fr = R[f"fb_frac{k}"].to_numpy(np.float64)
            ev = np.where(fr < 0, 0.5, fr)
            ev = np.maximum(ev, (ores > Tr).astype(np.float64))
            out[f"feedback{k}"] = np.where(pred_slow, 1 - ev, ev)
        fc = R["fb_size_class"].to_numpy(np.float64)
        evc = np.maximum(np.where(fc < 0, 0.5, fc), (ores > Tr).astype(np.float64))
        out["feedback_sizeclass"] = np.where(pred_slow, 1 - evc, evc)
        wb = R["out_write_bytes"].to_numpy(np.float64)
        out["write_backlog"] = np.where(pred_slow, -wb, wb)
        if "nb_all" in R:
            # neighbourhood: share of the last 8 submissions already known slow
            # (completed above T, or outstanding past T)
            nba = R["nb_all"].to_numpy(np.float64)
            nbr = R["nb_reads"].to_numpy(np.float64)
            nbr = np.where(nbr < 0, nba, nbr)
            out["neighbourhood"] = np.where(pred_slow, 1 - nba, nba)
            out["neighbourhood_reads"] = np.where(pred_slow, 1 - nbr, nbr)
            # combined: size-class outcome history and neighbourhood, each in [0, 1]
            fcc = np.where(fc < 0, 0.5, fc)
            comb = 0.5 * (fcc + nbr)
            comb = np.maximum(comb, (ores > Tr).astype(np.float64))
            out["combined"] = np.where(pred_slow, 1 - comb, comb)
        # at most 100 labelled reads: per-size-class slow probabilities estimated once
        # from exactly 100 labelled reads (end of the training window), then frozen;
        # only the outstanding-read age stays live
        if "y_req_w0" in R and "win" in R:
            size = R["size"].to_numpy(np.float64)
            cls = np.floor(np.log2(np.maximum(size, 512.0))).astype(int)
            lab = R["y_req_w0"].to_numpy()
            idx = np.flatnonzero((R["win"].to_numpy() == 0) & (lab >= 0))[-100:]
            prior = (lab[idx].sum() + 1.0) / (len(idx) + 2.0)
            pc = {}
            for c in np.unique(cls):
                m = cls[idx] == c
                pc[c] = (lab[idx][m].sum() + 2 * prior) / (m.sum() + 2.0)   # shrink to the pooled rate
            pq = np.array([pc.get(c, prior) for c in cls])
            evf = np.maximum(pq, (ores > Tr).astype(np.float64))
            out["frozen100"] = np.where(pred_slow, 1 - evf, evf)
    if "out_bytes" in R:
        # the situation tracker (admission.tracker): outcome rate of earlier reads in the
        # same (size class, outstanding bytes, oldest-age, quiet) cell.  "tracker" learns
        # from every completed read (the model in shadow mode); "tracker_admitted"
        # only from reads the model admitted (outcomes of rerouted reads unseen)
        Tr = float(R.attrs.get("T_read_w0", T))
        base = (R["ts_us"].to_numpy(np.float64), R["latency"].to_numpy(np.float64), R["size"].to_numpy(np.float64),
                R["out_bytes"].to_numpy(np.float64))
        since_d = R["since_done_us"].to_numpy(np.float64)   # < 0: nothing completed yet (quiet bit 0)
        out["tracker"] = trk.mistake_score(trk.evidence(*base, age, since_d, Tr), pred_slow)
        out["tracker_admitted"] = trk.mistake_score(trk.evidence(*base, age, since_d, Tr, observed=~pred_slow),
                                                    pred_slow)
        # ablations: the tracker without its quiet bit; the situation without the age class; size alone
        none = np.full(len(age), -1.0)
        zero = np.zeros(len(age))
        out["tracker_no_quiet"] = trk.mistake_score(trk.evidence(*base, age, none, Tr), pred_slow)
        out["tracker_no_age"] = trk.mistake_score(trk.evidence(*base, zero, since_d, Tr), pred_slow)
        out["tracker_size_only"] = trk.mistake_score(trk.evidence(*base[:3], zero, zero, none, Tr), pred_slow)
        # variant without the quiet bit: age class from the idle time when nothing is outstanding
        # (scored for comparison; docs/rule.md)
        sage = trk.situation_age(age, R["out_bytes"].to_numpy(np.float64), since_d)
        out["tracker_idle"] = trk.mistake_score(trk.evidence(*base, sage, none, Tr), pred_slow)
    out.update({
        "age": age,
        "since_done": since,
        # model says fast while an outstanding I/O is already slow; or model
        # says slow while nothing is waiting and completions are fresh
        "contradiction": np.where(pred_slow, 1 - np.clip(np.maximum(age, since) / T, 0, 1), np.clip(age / T, 0, 1)),
        "disagree": np.abs(live - p),
    })
    return out


def rule_params(D) -> dict:
    """Constants of the evidence rule for one device, from its training window only."""
    T = float(max(D["labels_w0"].ip_latency, 1))
    return {"t_slow_us": T, "t_fast_us": T, "t_read_us": float(D["T_read_w0"])}
