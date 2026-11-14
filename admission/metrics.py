"""Mistake-detection metrics.

A signal assigns each decision a score; higher means "more likely a mistake".
* AUROC: probability that a random mistake scores above a random correct
  decision (ties count one half).
* AURC: area under the risk-coverage curve.  Decisions are accepted in order
  of increasing score (most trusted first); risk at coverage k/n is the error
  rate among the k accepted.  Ties are handled exactly in expectation (a tied
  block is accepted in random order).  Lower is better.  E-AURC subtracts the
  AURC of the ideal ranking (all correct decisions first).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(mistake: np.ndarray, score: np.ndarray) -> float:
    m = np.asarray(mistake, bool)
    s = np.asarray(score, np.float64)
    n1 = int(m.sum())
    n0 = m.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)  # average ranks for ties
    return float((r[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aurc(mistake: np.ndarray, score: np.ndarray) -> float:
    m = np.asarray(mistake, np.float64)
    s = np.asarray(score, np.float64)
    n = m.size
    if n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    m_sorted = m[order]
    # tie blocks
    boundaries = np.flatnonzero(np.diff(s_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    total = 0.0
    errs_before = 0.0
    for a, b in zip(starts, ends):
        g = b - a
        e = m_sorted[a:b].sum()
        k = np.arange(a + 1, b + 1, dtype=np.float64)          # coverage counts in block
        cum = errs_before + (k - a) * (e / g)                    # expected errors accepted
        total += float((cum / k).sum())
        errs_before += e
    return total / n


def aurc_optimal(mistake: np.ndarray) -> float:
    m = np.asarray(mistake, bool)
    n = m.size
    n_err = int(m.sum())
    n_ok = n - n_err
    k = np.arange(n_ok + 1, n + 1, dtype=np.float64)
    return float(((k - n_ok) / k).sum() / n) if n else float("nan")


def summarise(mistake: np.ndarray, score: np.ndarray) -> dict:
    a = aurc(mistake, score)
    return {"auroc": auroc(mistake, score), "aurc": a, "eaurc": a - aurc_optimal(mistake),
            "n": int(np.size(mistake)), "n_mistakes": int(np.sum(mistake))}
