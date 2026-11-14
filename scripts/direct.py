#!/usr/bin/env python3
"""Is the outcome table a better predictor of slow reads than the network it audits?

For one baseline replay, four predictors score every read of the four later windows for the same
event, "this read's latency will exceed the knee":

  table     q, the table's slow-read frequency for the read's cell (admission/tracker.py, unchanged)
  heimdall  the logit of the deployed network, trained by Heimdall's pipeline on period labels
  net       a network of the same architecture and recipe, trained on the first window's reads
            with decision-time inputs and each read's own latency class as the label
  net_live  the same, with the table's live state appended to its inputs (bytes outstanding,
            age of the oldest outstanding request, time since the last completion)

Each is also scored on reads of one size only (the trace's commonest size class among the later
reads), where being large cannot explain being slow.

The table keeps learning from completions; the three networks are frozen at the end of the first
window, as Heimdall is deployed.  Both networks trained here see only reads submitted and completed
inside the first window.  The script also recomputes the table's mistake-ranking AUROC and, when
given the trace's per-window detection file, checks that it is reproduced.

Usage:
  direct.py NAME --replay FILE.parquet --models FILE.npz --out DIR [--duration S] [--detection FILE.json]
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from admission import tracker as trk  # noqa: E402
from admission.heimdall import FEATURES  # noqa: E402
from admission.metrics import auroc  # noqa: E402
from admission.protocol import load_trace, window_edges  # noqa: E402
from admission.train import LAYERS, MLP, load_models  # noqa: E402

LIVE = ["out_bytes", "oldest_age_us", "since_done_us"]


@dataclass(frozen=True)
class Replay:
    """A baseline replay in the shape admission.protocol.load_trace reads."""
    name: str
    path: str
    duration_s: float = 180.0

    def replay_parquet(self, policy="baseline", rep=0):
        return self.path


def train_net(X, y, seed=0, max_epochs=20, batch=32, patience=3, min_delta=0.01, lr=1e-3):
    """Heimdall's recipe (admission/train.py: Glorot weights, Adam, batch 32, at most 20 epochs,
    early stopping on the last fifth of the rows) on the rows given, in time order."""
    torch.manual_seed(seed)
    x_min = X.min(0)
    x_range = X.max(0) - x_min
    x_range[x_range == 0] = 1.0
    Z = torch.tensor((X - x_min) / x_range, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    n_val = int(len(Z) * 0.2)
    Zt, Yt, Zv, Yv = Z[: len(Z) - n_val], Y[: len(Z) - n_val], Z[len(Z) - n_val:], Y[len(Z) - n_val:]
    gen = torch.Generator().manual_seed(seed)
    dims = (X.shape[1],) + LAYERS[1:]
    params = []
    for fi, fo in zip(dims[:-1], dims[1:]):
        lim = float(np.sqrt(6.0 / (fi + fo)))
        params += [((torch.rand((fi, fo), generator=gen) * 2 - 1) * lim).requires_grad_(True),
                   torch.zeros(fo, requires_grad=True)]
    opt = torch.optim.Adam(params, lr=lr, eps=1e-7)

    def forward(z):
        h = torch.relu(z @ params[0] + params[1])
        h = torch.relu(h @ params[2] + params[3])
        return (h @ params[4] + params[5])[:, 0]

    best, wait, epochs = np.inf, 0, 0
    pgen = torch.Generator().manual_seed(seed + 7919)
    for ep in range(max_epochs):
        perm = torch.randperm(len(Zt), generator=pgen)
        for s in range(0, len(Zt), batch):
            idx = perm[s:s + batch]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(forward(Zt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        epochs = ep + 1
        with torch.no_grad():
            vl = float(torch.nn.functional.binary_cross_entropy_with_logits(forward(Zv), Yv)) if len(Zv) else 0.0
        if vl < best - min_delta:
            best, wait = vl, 0
        else:
            wait += 1
            if wait >= patience:
                break
    P = [p.detach().numpy().astype(np.float64) for p in params]
    return MLP(x_min, x_range, P[0], P[1], P[2], P[3], P[4], P[5], epochs)


def rate(a):
    return float(np.mean(a)) if len(a) else float("nan")


def run(name, replay, models_path, out_dir, duration, detection):
    t0 = time.time()
    D = load_trace(Replay(name, str(replay), duration))
    F = D["F"]
    R = F[F["io_type"].to_numpy() == 1].reset_index(drop=True)
    X = R[FEATURES].to_numpy(np.float64)
    win = R["win"].to_numpy()
    ts = R["ts_us"].to_numpy(np.float64)
    lat = R["latency"].to_numpy(np.float64)
    T0, Tfull = float(D["T_read_w0"]), float(D["T_read_full"])
    slow = R["y_req"].to_numpy().astype(bool)          # this read's latency above the full replay's knee
    slow_w0 = lat > T0                                 # the same against the first window's knee
    period = R["y"].to_numpy().astype(bool)            # Heimdall's period label

    models, _ = load_models(models_path)
    lg = models[0].logit(X)
    since = R["since_done_us"].to_numpy(np.float64)
    q = trk.evidence(ts, lat, R["size"].to_numpy(np.float64), R["out_bytes"].to_numpy(np.float64),
                     R["oldest_age_us"].to_numpy(np.float64), since, T0)

    # the two networks trained here: reads submitted and completed inside the first window
    w0_end = window_edges(duration)[0] * 1e6
    fit = (win == 0) & (ts + lat < w0_end)
    live = np.column_stack([R["out_bytes"].to_numpy(np.float64), R["oldest_age_us"].to_numpy(np.float64),
                            np.where(since < 0, 0.0, since)])
    XL = np.column_stack([X, live])
    y_fit = slow_w0[fit].astype(np.float64)
    nets = {}
    if fit.sum() >= 100 and 0 < y_fit.sum() < fit.sum():
        nets["net"] = train_net(X[fit], y_fit)
        nets["net_live"] = train_net(XL[fit], y_fit)
    scores = {"table": q, "heimdall": lg}
    if nets:
        scores["net"] = nets["net"].logit(X)
        scores["net_live"] = nets["net_live"].logit(XL)
    calls = {"table": q >= 0.5, "heimdall": lg > 0}
    for k in nets:
        calls[k] = scores[k] > 0

    pred = lg > 0
    mist = trk.mistake_score(q, pred)
    # reads of one size only: the commonest size class among the later reads
    cls = np.floor(np.log2(np.maximum(R["size"].to_numpy(np.float64), 1.0))).astype(np.int64)
    modal = int(np.bincount(cls[win >= 1]).argmax())
    same = cls == modal
    out = {"trace": name, "n_reads": int(len(R)), "T_read_w0_us": T0, "T_read_full_us": Tfull,
           "commonest_size_class": modal,
           "fit_reads": int(fit.sum()), "fit_slow": int(y_fit.sum()),
           "net_epochs": {k: m.epochs for k, m in nets.items()}, "windows": {}}
    worst_check = 0.0
    for w in sorted(set(win.tolist())):
        m = win == w
        ms = m & same
        row = {"n": int(m.sum()), "slow": int(slow[m].sum()), "slow_w0_knee": int(slow_w0[m].sum()),
               "period_slow": int(period[m].sum()), "same_size_n": int(ms.sum()),
               "same_size_slow": int(slow[ms].sum()),
               "mistakes": int((pred[m] != slow[m]).sum()),
               "mistake_auroc_table": auroc(pred[m] != slow[m], mist[m])}
        for k, s in scores.items():
            row[k] = {"auroc_slow": auroc(slow[m], s[m]), "auroc_slow_w0_knee": auroc(slow_w0[m], s[m]),
                      "auroc_slow_same_size": auroc(slow[ms], s[ms]),
                      "auroc_period": auroc(period[m], s[m]),
                      "wrong": int((calls[k][m] != slow[m]).sum()),
                      "wrong_w0_knee": int((calls[k][m] != slow_w0[m]).sum()),
                      "wrong_period": int((calls[k][m] != period[m]).sum()),
                      "called_slow": int(calls[k][m].sum())}
        out["windows"][str(w)] = row
        if detection is not None:
            pub = detection["windows"][str(w)]["signals_G2_request"]["tracker"]["auroc"]
            got = row["mistake_auroc_table"]
            if not (np.isnan(pub) and np.isnan(got)):
                worst_check = max(worst_check, abs(pub - got))
    if detection is not None:
        out["detection_file_max_abs_difference"] = worst_check
    out["seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1))
    later = [w for w in out["windows"] if w != "0"]
    msg = " ".join(f"{k}={min(out['windows'][w][k]['auroc_slow'] for w in later):.3f}" for k in scores)
    print(f"{name}: reads={len(R)} worst later window, AUROC for slow: {msg}"
          + (f"  detection file check {worst_check:.1e}" if detection is not None else "")
          + f"  [{out['seconds']:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--detection", default=None)
    a = ap.parse_args()
    pub = json.loads(Path(a.detection).read_text()) if a.detection else None
    run(a.name, Path(a.replay), Path(a.models), Path(a.out), a.duration, pub)


if __name__ == "__main__":
    main()
