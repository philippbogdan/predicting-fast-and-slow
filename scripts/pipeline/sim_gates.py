#!/usr/bin/env python3
"""Tail latency of gates acting inside Heimdall's admission path (simulated).

For each trace pair, both clients run Heimdall's network retrained on W0
(results/models/<trace>.npz, seed 0) and a gate:
  * heimdall: Heimdall alone
  * euclid, mahal: the study's distance gates: distance of the normalised
    input to the training centroid above the 90th percentile of training
    distances -> hedge (both SSDs, first wins)
  * tracker-mixed+timeout: the tracker's gate with the timeout, line 5 of the
    table's gate (docs/method.md); tracker-* and feedback-*: other actions and thresholds
  * rule-hedge, rule-flip: an earlier decision-time evidence rule
  * oracle-*: distrusts exactly the wrong decisions
plus baseline (no admission control) and random for context.
Metrics over the reads the artefact's baseline replay submitted in the later
windows (the detection protocol's windows; the same requests for every policy): p50/p99/p99.9/mean read latency per client and
pooled, rerouted fraction (reads that send any I/O to the other SSD).  Writes results/gate/<pair>.json and summary.json.
"""
import argparse
import json

import numpy as np

from admission import RESULTS
from admission.heimdall import FEATURES
from admission.protocol import load_trace, window_edges, CLIENT_LEVEL_DURATION_S
from admission.provenance import stamp
from admission.evidence import rule_params
from admission.sim import simulate
from admission.simdata import client_inputs, duration_s, match_log_to_raw, load_raw, size_alpha
from admission.train import heimdall_split, load_models
from admission.traces import PAIRS, DeviceTrace, RawOnlyTrace
import pandas as pd

# trace pairs: the artefact's six client-level pairs, the kernel-level pair
# (traceL on device 0, traceB on device 1, as its config.conf), and each
# ds_pipeline sample trace mirrored on two identical devices (our replay
# repetitions 0 and 1 are the two devices' baselines)
PAIR_SPECS = {**{p: [(DeviceTrace(p, 0), 0), (DeviceTrace(p, 1), 0)] for p in PAIRS},
              "K": [(RawOnlyTrace("KL"), 0), (RawOnlyTrace("KB"), 0)],
              "S1": [(RawOnlyTrace("S1"), 0), (RawOnlyTrace("S1"), 1)],
              "S2": [(RawOnlyTrace("S2"), 0), (RawOnlyTrace("S2"), 1)],
              "S3": [(RawOnlyTrace("S3"), 0), (RawOnlyTrace("S3"), 1)]}

PCTS = [50, 99, 99.9]
BETA = 0.0   # foreign-read cost exponent (admission.sim); set from --beta
TMPL_W = 1   # foreign-read template window (admission.sim); set from --W
FMULT = {}   # pair -> foreign-read cost multipliers per client, from results/simulator/chosen_P.json


def gate_inputs(t, D, rep=0):
    """Model weights, distance-gate parameters, rule parameters, labels by raw index."""
    models, _ = load_models(RESULTS / "models" / f"{t.name}.npz")
    m = models[0]
    x_tr, *_ = heimdall_split(D["train_X"], D["train_y"])
    Zt = (x_tr - m.x_min) / m.x_range
    mu = Zt.mean(0)
    cov = np.cov(Zt.T) + 1e-6 * np.eye(12)
    Pinv = np.linalg.pinv(cov)
    de = np.sqrt(((Zt - mu) ** 2).sum(1))
    dm = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Zt - mu, Pinv, Zt - mu), 0))
    raw = load_raw(t)
    ylab = np.full(len(raw), -1, np.int8)      # G2: per-request read slowness (primary)
    ylab_g1 = np.full(len(raw), -1, np.int8)   # G1: Heimdall period labels
    dur_ms = duration_s(t) * 1000.0
    if rep == 0:
        base = pd.read_parquet(t.replay_parquet("baseline"))
        idx = match_log_to_raw(raw, base, dur_ms)
        F = D["F"]
        rows = F["row"].to_numpy()
        ok = idx[rows] >= 0
        ylab[idx[rows][ok]] = F["y_req"].to_numpy()[ok]
        ylab_g1[idx[rows][ok]] = F["y"].to_numpy()[ok]
    else:   # second replay of a mirrored pair: per-request labels from its own latencies
        base = pd.read_parquet(t.replay_parquet("baseline", rep))
        idx = match_log_to_raw(raw, base, dur_ms)
        ok = (idx >= 0) & (base["io_type"].to_numpy() == 1)
        ylab[idx[ok]] = (base["latency"].to_numpy()[ok] > D["T_read_full"]).astype(np.int8)
    rp = rule_params(D)
    return {"model": m, "mu": mu, "Pinv": Pinv, "thr_e": float(np.percentile(de, 90)),
            "thr_m": float(np.percentile(dm, 90)), "ylab": ylab, "ylab_g1": ylab_g1, "rule": rp,
            "pct_e": {q: float(np.percentile(de, q)) for q in (50, 70, 80, 90, 95, 99)},
            "pct_m": {q: float(np.percentile(dm, q)) for q in (50, 70, 80, 90, 95, 99)}}


def run_policy(ci, gi, P, alpha, policy, gate=0, action=0, seed=1, k_slow=1.0, k_fast=1.0, dist_pct=None,
               theta=0.5, oracle_label="ylab", fmult=(1.0, 1.0), timeout=False, awake=False, censored=False):
    a, b = ci
    W = [np.stack([getattr(g["model"], k) for g in gi]) for k in ("x_min", "x_range", "W1", "b1", "W2", "b2", "W3", "b3")]
    if dist_pct is None:
        thr = np.array([g["thr_e"] if gate == 1 else g["thr_m"] for g in gi])
    else:
        thr = np.array([g["pct_e"][dist_pct] if gate == 1 else g["pct_m"][dist_pct] for g in gi])
    mu = np.stack([g["mu"] for g in gi])
    Pinv = np.stack([g["Pinv"] for g in gi])
    if gate in (5, 6, 7):   # outcome-feedback rules work on read latency: the read knee of W0
        t_slow = np.array([g["rule"]["t_read_us"] * k_slow for g in gi])
    else:
        t_slow = np.array([g["rule"]["t_slow_us"] * k_slow for g in gi])
    t_fast = np.array([g["rule"]["t_fast_us"] * k_fast for g in gi])
    return simulate(a.arr_us, a.size, a.io, a.work, a.rd_idx, a.rd_work, a.rd_size,
                    b.arr_us, b.size, b.io, b.work, b.rd_idx, b.rd_work, b.rd_size,
                    float(P), float(P), float(alpha), float(min(a.duration_us, b.duration_us)), policy, seed,
                    *W, gate, thr, mu, Pinv, t_slow, t_fast, HEDGE_T[0], HEDGE_T[1],
                    gi[0][oracle_label], gi[1][oracle_label], action, theta, a.st_k, a.st_d, b.st_k, b.st_d,
                    float(BETA), int(TMPL_W), float(fmult[0]), float(fmult[1]), bool(timeout), bool(awake),
                    bool(censored))


HEDGE_T = [0.0, 0.0]   # per client: the hedging timeout, 98th percentile of the training window's read latency


def training_timeout(c_in, t_from_us):
    """98th percentile of the read latencies the baseline replay recorded in the training window (W0):
    deployable: no later-window information."""
    k = c_in.k_base
    sub = np.nan_to_num(c_in.base_sub_us[:k], nan=np.inf)
    lat = c_in.base_lat[:k]
    sel = (c_in.io[:k] == 1) & (sub < t_from_us) & (lat >= 0)
    return float(np.percentile(lat[sel], 98)) if sel.sum() > 50 else float(c_in.hedge_t_us)


def scored_reads(ci, t_from_us):
    """The requests every policy is scored on: the reads the artefact's baseline replay submitted in the
    later windows of its own timeline (the windows of the detection protocol); on busy traces the replay
    lags the trace, so these are not the reads that *arrive* after the first third.  Fixed by the baseline
    replay, never by a policy's simulated issue times, so the set is identical for every policy."""
    horizon = min(ci[0].duration_us, ci[1].duration_us)   # as run_policy passes it to the simulator
    return [(ci[c].io == 1) & (np.arange(len(ci[c].io)) < ci[c].k_base)
            & (np.nan_to_num(ci[c].base_sub_us, nan=-1.0) >= t_from_us) & (ci[c].arr_us <= horizon) for c in (0, 1)]


def metrics(o, ci, t_from_us):
    out = {}
    pooled = []
    for c, reads in enumerate(scored_reads(ci, t_from_us)):
        iss, lat, route = o[4 * c], o[4 * c + 1], o[4 * c + 2]
        if (lat[reads] < 0).any():
            raise RuntimeError(f"client {c}: {(lat[reads] < 0).sum()} scored reads never completed")
        l = lat[reads]
        pooled.append(l)
        out[f"d{c}"] = {"n": int(reads.sum()), "mean": float(l.mean()),
                        **{f"p{p}": float(np.percentile(l, p)) for p in PCTS},
                        "rerouted": float(np.mean(route[reads] >= 1)), "hedged": float(np.mean(route[reads] == 2))}
    l = np.concatenate(pooled)
    n_r = sum(out[f"d{c}"]["n"] for c in (0, 1))
    out["pooled"] = {"n": int(len(l)), "mean": float(l.mean()), **{f"p{p}": float(np.percentile(l, p)) for p in PCTS},
                     "rerouted": float(sum(out[f"d{c}"]["rerouted"] * out[f"d{c}"]["n"] for c in (0, 1)) / n_r),
                     "hedged": float(sum(out[f"d{c}"]["hedged"] * out[f"d{c}"]["n"] for c in (0, 1)) / n_r)}
    return out


POLICIES = {
    "baseline": (0, 0, 0), "random": (1, 0, 0), "heimdall": (2, 0, 0),
    "euclid": (3, 1, 0), "mahal": (3, 2, 0),
    "rule-hedge": (3, 3, 0), "rule-flip": (3, 3, 1), "rule-mixed": (3, 3, 2),
    "feedback-hedge": (3, 5, 0), "feedback-flip": (3, 5, 1), "feedback-mixed": (3, 5, 2),
    "combined-hedge": (3, 6, 0), "combined-flip": (3, 6, 1), "combined-mixed": (3, 6, 2),
    "oracle-hedge": (3, 4, 0), "oracle-flip": (3, 4, 1),
    "tracker-hedge": (3, 7, 0), "tracker-flip": (3, 7, 1), "tracker-mixed": (3, 7, 2),
    "tracker-hedgeslow": (3, 7, 3), "tracker-hedgefast": (3, 7, 4),
    "euclid-hedgeslow": (3, 1, 3), "mahal-hedgeslow": (3, 2, 3),
    "oracle-hedgeslow": (3, 4, 3), "oracle-hedgefast": (3, 4, 4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", default=list(PAIR_SPECS))
    ap.add_argument("--P", type=float, required=True)
    ap.add_argument("--alpha_scale", type=float, default=1.0)
    ap.add_argument("--stall_G", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--W", type=int, default=1)
    ap.add_argument("--out", default=None, help="output directory relative to the repository (default results/gate)")
    ap.add_argument("--only", default=None, help="comma-separated policy names to run (default: all)")
    ap.add_argument("--fmult", default=None, help="sensitivity check: moved-read cost multipliers 'c0,c1' for every pair run")
    a = ap.parse_args()
    global BETA, TMPL_W, FMULT
    BETA, TMPL_W = a.beta, a.W
    cp = RESULTS / "simulator" / "chosen_P.json"
    if cp.exists():
        ch = json.loads(cp.read_text())
        if ch.get("calibrated"):
            FMULT = {k: tuple(v) for k, v in ch.get("fmult_by_pair", {}).items()}
    out_dir = RESULTS / "gate" if a.out is None else RESULTS.parent / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = a.pairs
    for pair in pairs:
        spec = PAIR_SPECS[pair]
        traces = [t for t, _ in spec]
        ci = [client_inputs(t, a.P, rep, stall_G=a.stall_G) if isinstance(t, RawOnlyTrace)
              else client_inputs(t, a.P, stall_G=a.stall_G) for t, rep in spec]
        Dcache = {}
        D = []
        for t, _ in spec:
            if t.name not in Dcache:
                Dcache[t.name] = load_trace(t)
            D.append(Dcache[t.name])
        gi = [gate_inputs(t, d, rep) for (t, rep), d in zip(spec, D)]
        alpha = float(np.median([size_alpha(t) for t in traces])) * a.alpha_scale
        t_from = window_edges(min(duration_s(t) for t in traces))[0] * 1e6
        HEDGE_T[0], HEDGE_T[1] = training_timeout(ci[0], t_from), training_timeout(ci[1], t_from)
        res = {"pair": pair, "P": a.P, "alpha_scale": a.alpha_scale, "stall_G": a.stall_G, "beta": a.beta, "W": a.W,
               "alpha_bytes": alpha,
               "policies": {}}
        runs = [(name, pol, gate, act, {}) for name, (pol, gate, act) in POLICIES.items()]
        # other settings: distance gates at other percentiles, the rule at other multiples of T
        for q in (50, 70, 80, 95, 99):
            runs += [(f"euclid-p{q}", 3, 1, 0, {"dist_pct": q}), (f"mahal-p{q}", 3, 2, 0, {"dist_pct": q})]
        for act, an in ((0, "hedge"), (1, "flip"), (2, "mixed")):
            runs.append((f"oracleG1-{an}", 3, 4, act, {"oracle_label": "ylab_g1"}))
        runs.append(("oracle-mixed", 3, 4, 2, {}))
        for th in (0.25, 0.75, 0.9):
            for act, an in ((0, "hedge"), (1, "flip"), (2, "mixed")):
                runs.append((f"feedback-{an}-th{th}", 3, 5, act, {"theta": th}))
            for act, an in ((0, "hedge"), (1, "flip"), (2, "mixed"), (3, "hedgeslow"), (4, "hedgefast")):
                runs.append((f"tracker-{an}-th{th}", 3, 7, act, {"theta": th}))
        for q in (50, 70, 80, 95, 99):
            runs += [(f"euclid-hedgeslow-p{q}", 3, 1, 3, {"dist_pct": q}), (f"mahal-hedgeslow-p{q}", 3, 2, 3, {"dist_pct": q})]
        # reference routing without the model: the less loaded device (bytes: host-visible; work: oracle)
        for m in (0.0, 65536.0, 262144.0, 1048576.0):
            runs.append((f"jsq-bytes-m{int(m)}", 6, 0, 0, {"theta": m}))
        for m in (0.0, 100.0, 500.0, 2000.0):
            runs.append((f"lwl-oracle-m{int(m)}", 7, 0, 0, {"theta": m}))
        # the artefact's timeout hedging (no model): a duplicate to the other device when a read is not done
        # after the 98th percentile of the training window's read latency; and the same timeout added to
        # Heimdall, to the study's distance gates and to the rule's gate (identical information for the gates)
        runs.append(("timeout-hedging", 4, 0, 0, {}))
        # an oracle that knows which decisions are wrong, given the same timeout: what perfect per-read
        # knowledge adds to the timeout (bounds what any decision-time signal can add)
        runs += [(f"oracle-{an}+timeout", 3, 4, act, {"timeout": True})
                 for act, an in ((0, "hedge"), (1, "flip"), (2, "mixed"), (3, "hedgeslow"), (4, "hedgefast"))]
        runs += [("heimdall+timeout", 2, 0, 0, {"timeout": True}),
                 ("euclid+timeout", 3, 1, 0, {"timeout": True}), ("mahal+timeout", 3, 2, 0, {"timeout": True}),
                 ("tracker-mixed+timeout", 3, 7, 2, {"timeout": True}),
                 ("tracker-hedge+timeout", 3, 7, 0, {"timeout": True}),
                 ("tracker-hedgeslow+timeout", 3, 7, 3, {"timeout": True})]
        for ks in (0.5, 2.0, 4.0, 8.0):
            for act, an in ((0, "hedge"), (1, "flip"), (2, "mixed")):
                runs.append((f"rule-{an}-ks{ks}", 3, 3, act, {"k_slow": ks, "k_fast": 1.0}))
        # foreign-read cost multipliers fitted on the artefact's random-routing replay (simulator stage);
        # pairs without artefact replays (our raw-only traces) keep 1
        fm = tuple(FMULT.get(pair, (1.0, 1.0)))
        if a.fmult:
            fm = tuple(float(x) for x in a.fmult.split(","))
        res["fmult"] = list(fm)
        res["hedge_timeout_us"] = list(HEDGE_T)
        if a.only:
            keep = set(a.only.split(","))
            runs = [r for r in runs if r[0] in keep]
        for name, pol, gate, act, kw in runs:
            o = run_policy(ci, gi, a.P, alpha, pol, gate, act, fmult=fm, **kw)
            res["policies"][name] = metrics(o, ci, t_from)
            m = res["policies"][name]["pooled"]
            print(f"{pair} {name:13s} p50 {m['p50']:7.0f} p99 {m['p99']:7.0f} p99.9 {m['p99.9']:7.0f} "
                  f"rerouted {m['rerouted']:.3f} hedged {m['hedged']:.3f}", flush=True)
        (out_dir / f"{pair}.json").write_text(json.dumps(stamp(res), indent=1))


if __name__ == "__main__":
    main()
