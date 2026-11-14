"""Candidate per-decision signals.  Every signal returns a score per request,
higher = more likely that the model's decision is a mistake.

Study's estimators (MLSys 2026, Section 4.3):
  * euclid:     Euclidean distance of the (MinMax-normalised) input to the
                 centroid of the model's training inputs
  * mahal:      Mahalanobis distance to the same centroid
  * conformal:  distance-weighted conformal prediction (their Algorithm 1,
                 classification form): calibration nonconformity
                 s_i = 1 - p_{y_i}(x_i), weights exp(-||x_i - x|| / tau),
                 tau = 1; score = weighted p-value of the label the model did
                 not predict (high -> both labels plausible -> uncertain)
  * bnn:        predictive std of a Bayesian version of the network
Standard alternatives:
  * margin:     -|logit| (softmax response; entropy ranks identically)
  * ensemble:   std of probabilities across independently seeded networks
  * knn:        distance to the k-th nearest training point in the
                 network's last hidden layer, L2-normalised (Sun et al. 2022)
  * trust:      trust score (Jiang et al. 2018): distance to the nearest
                 training point of the predicted class over that of the other
"""
from __future__ import annotations

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _nograd(f):
    def g(*a, **k):
        with torch.no_grad():
            return f(*a, **k)
    g.__name__, g.__doc__ = f.__name__, f.__doc__
    return g


def _t(a, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=DEVICE)


def margin(logit: np.ndarray) -> np.ndarray:
    return -np.abs(logit)


def euclid(Z: np.ndarray, Z_train: np.ndarray) -> np.ndarray:
    mu = Z_train.mean(0)
    return np.sqrt(((Z - mu) ** 2).sum(1))


def mahal(Z: np.ndarray, Z_train: np.ndarray, ridge=1e-6) -> np.ndarray:
    mu = Z_train.mean(0)
    cov = np.cov(Z_train.T) + ridge * np.eye(Z_train.shape[1])
    P = np.linalg.pinv(cov)
    D = Z - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, P, D), 0))


@_nograd
def conformal_alt_pvalue(Z: np.ndarray, p1: np.ndarray, Zc: np.ndarray, p1c: np.ndarray,
                         yc: np.ndarray, tau: float = 1.0, chunk: int = 1 << 13) -> np.ndarray:
    """Weighted conformal p-value of the non-predicted label.

    p1 = model probability of "slow" for test points; p1c/yc for calibration.
    """
    sc = np.where(yc == 1, 1 - p1c, p1c)             # nonconformity of calibration points
    pred = (p1 > 0.5).astype(np.int64)
    s_alt = np.where(pred == 1, p1, 1 - p1)          # 1 - p(alt) = p(pred)
    Zc_t = _t(Zc)
    sc_t = _t(sc)
    out = np.empty(len(Z))
    for s in range(0, len(Z), chunk):
        z = _t(Z[s:s + chunk])
        d = torch.cdist(z, Zc_t)                      # [B, Nc]
        w = torch.exp(-d / tau)
        sa = _t(s_alt[s:s + chunk]).unsqueeze(1)
        num = (w * (sc_t.unsqueeze(0) >= sa)).sum(1) + 1.0   # test point's own weight exp(0)=1
        den = w.sum(1) + 1.0
        out[s:s + chunk] = (num / den).cpu().numpy()
    return out


@_nograd
def conformal_uncertain_fraction(Z, p1, Zc, p1c, yc, alpha=0.1, tau=1.0):
    """Fraction of requests whose prediction set is not a single label."""
    pv_alt = conformal_alt_pvalue(Z, p1, Zc, p1c, yc, tau)
    # p-value of the predicted label: nonconformity 1 - p(pred)
    sc = np.where(yc == 1, 1 - p1c, p1c)
    pred = (p1 > 0.5)
    s_pred = np.where(pred, 1 - p1, p1)
    Zc_t = _t(Zc)
    sc_t = _t(sc)
    out = np.empty(len(Z))
    for s in range(0, len(Z), 1 << 13):
        z = _t(Z[s:s + (1 << 13)])
        w = torch.exp(-torch.cdist(z, Zc_t) / tau)
        sp = _t(s_pred[s:s + (1 << 13)]).unsqueeze(1)
        out[s:s + (1 << 13)] = (((w * (sc_t.unsqueeze(0) >= sp)).sum(1) + 1) / (w.sum(1) + 1)).cpu().numpy()
    in_pred = out > alpha
    in_alt = pv_alt > alpha
    # trusted only when the prediction set is exactly {predicted label}
    return float(np.mean(~(in_pred & ~in_alt)))


def ensemble_std(logits: np.ndarray) -> np.ndarray:
    """logits: [K, n]."""
    p = 1 / (1 + np.exp(-logits))
    return p.std(0)


def bnn_std(sample_logits: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-sample_logits))
    return p.std(0)


@_nograd
def knn_distance(H: np.ndarray, H_train: np.ndarray, k: int = 10, chunk: int = 1 << 12) -> np.ndarray:
    def l2n(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    Ht = _t(l2n(H_train))
    out = np.empty(len(H))
    Hn = l2n(H)
    for s in range(0, len(H), chunk):
        d = torch.cdist(_t(Hn[s:s + chunk]), Ht)
        out[s:s + chunk] = torch.topk(d, k, dim=1, largest=False).values[:, -1].cpu().numpy()
    return out


@_nograd
def trust_score_inv(Z: np.ndarray, pred: np.ndarray, Z_train: np.ndarray, y_train: np.ndarray,
                    chunk: int = 1 << 12) -> np.ndarray:
    """d(x, nearest train point of predicted class) / d(x, nearest of other class)."""
    A = [_t(Z_train[y_train == c]) for c in (0, 1)]
    out = np.empty(len(Z))
    for s in range(0, len(Z), chunk):
        z = _t(Z[s:s + chunk])
        # a class absent from the training sample is infinitely far away
        inf = torch.full((len(z),), float("inf"), device=z.device, dtype=z.dtype)
        d0 = torch.cdist(z, A[0]).min(1).values if len(A[0]) else inf
        d1 = torch.cdist(z, A[1]).min(1).values if len(A[1]) else inf
        pr = _t(pred[s:s + chunk])
        d_pred = torch.where(pr > 0.5, d1, d0)
        d_other = torch.where(pr > 0.5, d0, d1)
        out[s:s + chunk] = (d_pred / d_other.clamp_min(1e-9)).cpu().numpy()
    return out
