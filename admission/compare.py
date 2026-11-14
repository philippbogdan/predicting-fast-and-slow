"""Score every candidate signal on the same requests from the same information.

Used verbatim for the SSD-admission traces and for the transfer tasks.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import signals as S
from .metrics import summarise
from .evidence import evidence_scores
from .train import heimdall_split

KNN_REF = 20000
CAL_SIZES = (100, 1000)


def score_all(R, X, models, bnn, Xtr, ytr, T_us, win, y_w0, cal_labels=100, extra_cal=None):
    """R: per-request frame with oldest_age_us, since_done_us, n_inflight,
    prev_latency_1; X: model inputs; models[0] is the deployed network.
    y_w0: Heimdall's period labels for the training window;
    extra_cal: {name: labels} for other ground truths; label-dependent
    signals (conformal, supervised reference) are recomputed with those."""
    m0 = models[0]
    x_tr, _, y_tr, _, _, _ = heimdall_split(Xtr, ytr)
    Z_train = (x_tr - m0.x_min) / m0.x_range
    Z = (X - m0.x_min) / m0.x_range
    logits = np.stack([m.logit(X) for m in models])
    lg = logits[0]
    p1 = 1 / (1 + np.exp(-lg))
    pred = (lg > 0).astype(np.int64)

    sc = {}
    sc["margin"] = S.margin(lg)
    sc["euclid"] = S.euclid(Z, Z_train)
    sc["mahal"] = S.mahal(Z, Z_train)
    sc["ensemble"] = S.ensemble_std(logits)
    if bnn is not None:
        sc["bnn"] = S.bnn_std(bnn.sample_logits(X, n_samples=32))
    ref = np.random.default_rng(0).choice(len(x_tr), size=min(KNN_REF, len(x_tr)), replace=False)
    sc["knn"] = S.knn_distance(m0.hidden(X)[1], m0.hidden(x_tr[ref])[1])
    sc["trust"] = S.trust_score_inv(Z, pred, Z_train[ref], y_tr[ref])
    sc.update(evidence_scores(R, lg, T_us))

    # identical information for the study's estimators: add the live state
    live = np.column_stack([R["oldest_age_us"].to_numpy(np.float64),
                            np.where(R["since_done_us"].to_numpy() < 0, 0, R["since_done_us"].to_numpy(np.float64)),
                            R["n_inflight"].to_numpy(np.float64)])
    w0m = win == 0
    lo, hi = live[w0m].min(0), live[w0m].max(0)
    live_z = (live - lo) / np.where(hi > lo, hi - lo, 1.0)
    Za = np.column_stack([Z, live_z])
    sc["euclid_aug"] = S.euclid(Za, Za[w0m])
    sc["mahal_aug"] = S.mahal(Za, Za[w0m])
    info = np.column_stack([X, live, lg])
    later = win >= 1
    unc = {}
    for lname, yl in [("", y_w0)] + sorted((extra_cal or {}).items()):
        sfx = f"_{lname}" if lname else ""
        w0_idx = np.flatnonzero((win == 0) & (yl >= 0))
        cal = {n: w0_idx[-n:] for n in sorted(set(CAL_SIZES) | {cal_labels})}
        for n in CAL_SIZES:
            idx = cal[n]
            sc[f"conformal{n}{sfx}"] = S.conformal_alt_pvalue(Z, p1, Z[idx], p1[idx], yl[idx])
            sc[f"conformal{n}_aug{sfx}"] = S.conformal_alt_pvalue(Za, p1, Za[idx], p1[idx], yl[idx])
            unc[f"conformal{n}_alpha0.1{sfx}"] = S.conformal_uncertain_fraction(
                Z[later], p1[later], Z[idx], p1[idx], yl[idx])
        # supervised reference: a detector fit on every labelled W0 request
        # (every labelled read of W0, not a deployable signal; it shows what labels buy)
        fit = w0m & (yl >= 0)
        if fit.sum() > 50 and len(set((pred[fit] != yl[fit]).tolist())) == 2:
            det = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
            det.fit(info[fit], (pred[fit] != yl[fit]).astype(int))
            sc[f"supervised_w0{sfx}"] = det.predict_proba(info)[:, 1]
        else:
            sc[f"supervised_w0{sfx}"] = np.zeros(len(X))
    return sc, lg, pred, unc


def window_metrics(sc, mistake, pred, y, win, extra_labels=None):
    out = {}
    for w in sorted(set(win.tolist())):
        msk = win == w
        row = {"n": int(msk.sum()), "error_rate": float(mistake[msk].mean()),
               "fn_rate": float(((pred == 0) & (y == 1))[msk].mean()),
               "fp_rate": float(((pred == 1) & (y == 0))[msk].mean()),
               "slow_rate": float(y[msk].mean()),
               "signals": {k: summarise(mistake[msk], s[msk]) for k, s in sc.items()}}
        if extra_labels:
            for lname, mist2 in extra_labels.items():
                row[f"error_rate_{lname}"] = float(mist2[msk].mean())
                row[f"signals_{lname}"] = {k: summarise(mist2[msk], s[msk]) for k, s in sc.items()}
        out[str(w)] = row
    return out
