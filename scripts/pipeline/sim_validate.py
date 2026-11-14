#!/usr/bin/env python3
"""Validate the simulator against the artefact's replays.

For every trace pair and lane count P, simulate baseline, random, Heimdall
(artefact weights) and the artefact's hedging, and compare read-latency
statistics with the artefact's replay logs on the identical requests (raw
trace indices both runs issued, within the baseline-issued prefix).
Writes results/simulator/validate_<pair>.json.
"""
import argparse
import json

import numpy as np

from admission import RESULTS
from admission.protocol import window_edges
from admission.provenance import stamp
from admission.sim import simulate
from admission.simdata import client_inputs, duration_s, load_raw, replay_latency_by_index, size_alpha
from admission.traces import PAIRS, DeviceTrace

POL = {"baseline": 0, "random": 1, "flashnet": 2, "hedging": 4}
PCTS = [50, 90, 99, 99.9]


def stack_models(ms):
    return [np.stack([getattr(m, a) for m in ms]) for a in ("x_min", "x_range", "W1", "b1", "W2", "b2", "W3", "b3")]


def run(pair, P, alpha, policy, ci, seed=1, beta=0.0, tmpl_W=1, fmult=(1.0, 1.0)):
    a, b = ci
    W = stack_models([a.model, b.model])
    zeros12 = np.zeros((2, 12))
    out = simulate(a.arr_us, a.size, a.io, a.work, a.rd_idx, a.rd_work, a.rd_size,
                   b.arr_us, b.size, b.io, b.work, b.rd_idx, b.rd_work, b.rd_size,
                   float(P), float(P), float(alpha), float(min(a.duration_us, b.duration_us)), policy, seed,
                   *W, 0, np.zeros(2), zeros12, np.zeros((2, 12, 12)), np.zeros(2), np.zeros(2),
                   a.hedge_t_us, b.hedge_t_us, np.full(len(a.io), -1, np.int8), np.full(len(b.io), -1, np.int8), 0,
                   0.5, a.st_k, a.st_d, b.st_k, b.st_d, float(beta), int(tmpl_W), float(fmult[0]), float(fmult[1]))
    return out


FMULT_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def calibrate_foreign(pair, P, alpha, ci, measured, beta, tmpl_W):
    """Per client, the multiplier on the cost of its reads served by the other
    device that makes the simulated hedging replay match the artefact's: the
    median excess over the hedge timeout of reads that outlived it (min of the
    primary's remaining time and the duplicate's latency on the other device)."""
    target = []
    for c in (0, 1):
        k = ci[c].k_base
        m = measured["hedging"][c][:k]
        rd = (ci[c].io[:k] == 1) & (m > ci[c].hedge_t_us)
        target.append(float(np.median(m[rd] - ci[c].hedge_t_us)) if rd.sum() > 20 else float("nan"))
    best = [(float("inf"), 1.0), (float("inf"), 1.0)]
    for f in FMULT_GRID:
        o = run(pair, P, alpha, POL["hedging"], ci, beta=beta, tmpl_W=tmpl_W, fmult=(f, f))
        for c in (0, 1):
            lat = o[4 * c + 1][:ci[c].k_base]
            rd = (ci[c].io[:ci[c].k_base] == 1) & (lat > ci[c].hedge_t_us)
            if rd.sum() > 20 and target[c] == target[c]:
                err = abs(float(np.median(lat[rd] - ci[c].hedge_t_us)) - target[c])
                if err < best[c][0]:
                    best[c] = (err, f)
    return (best[0][1], best[1][1]), target


def calibrate_on_random_w0(pair, P, alpha, ci, measured, beta, tmpl_W, t_w1):
    """Validation design (device model v3): the cost of a read served by the other device
    is pair-specific and the baseline logs cannot show it (results/simulator/foreign_reads.json).  Per
    client, choose the multiplier on that cost so that the simulated random replay matches the artefact's
    random replay *on the training window only* (reads the baseline replay submitted before the first later
    window; the protocol's windows are on that timeline: median read latency).  The validation statistics are then taken on the later windows only, for baseline, random
    (held out in time) and hedging (held out in policy)."""
    target = []
    w0 = []
    for c in (0, 1):
        k = ci[c].k_base
        m = measured["random"][c][:k]
        sel = (ci[c].io[:k] == 1) & (np.nan_to_num(ci[c].base_sub_us[:k], nan=np.inf) < t_w1) & (m >= 0)
        w0.append(sel)
        target.append(float(np.median(m[sel])) if sel.sum() > 20 else float("nan"))
    best = [(float("inf"), 1.0), (float("inf"), 1.0)]
    for f in FMULT_GRID:
        o = run(pair, P, alpha, POL["random"], ci, beta=beta, tmpl_W=tmpl_W, fmult=(f, f))
        for c in (0, 1):
            lat = o[4 * c + 1][:ci[c].k_base]
            sel = w0[c] & (lat >= 0)
            if sel.sum() > 20 and target[c] == target[c]:
                err = abs(np.log(float(np.median(lat[sel])) / target[c]))
                if err < best[c][0]:
                    best[c] = (err, f)
    return (best[0][1], best[1][1]), target


def stats(lat):
    lat = lat[lat >= 0]
    if len(lat) == 0:
        return None
    return {"n": int(len(lat)), "mean": float(lat.mean()), **{f"p{p}": float(np.percentile(lat, p)) for p in PCTS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", default=list(PAIRS))
    ap.add_argument("--P", nargs="*", type=float, default=[4.0, 8.0, 16.0, 1e9])
    ap.add_argument("--alpha_scale", nargs="*", type=float, default=[1.0])
    ap.add_argument("--stall_G", nargs="*", type=float, default=[0.0],
                    help="device-stall threshold in us (0: no stall model)")
    ap.add_argument("--beta", nargs="*", type=float, default=[0.0],
                    help="foreign-read cost: receiving template^(1-beta) * own work^beta")
    ap.add_argument("--W", nargs="*", type=int, default=[1],
                    help="template drawn from the receiving device's last W own reads")
    ap.add_argument("--calibrate", nargs="*", type=int, default=[0],
                    help="1: fit per-client foreign-read cost multipliers on the hedging replay; "
                         "2: fit them on the training window of the random replay and validate on the later "
                         "windows only (calibrate_on_random_w0)")
    ap.add_argument("--out", default=None, help="output directory relative to the repository (default results/simulator)")
    a = ap.parse_args()
    out_dir = RESULTS / "simulator" if a.out is None else RESULTS.parent / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for pair in a.pairs:
        traces = [DeviceTrace(pair, 0), DeviceTrace(pair, 1)]
        raws = [load_raw(t) for t in traces]
        alpha0 = float(np.median([size_alpha(t) for t in traces]))
        measured = {pol: [replay_latency_by_index(t, pol, len(r), r) for t, r in zip(traces, raws)]
                    for pol in POL}
        res = {"pair": pair, "alpha_bytes": alpha0, "by_P": {}}
        for P, asc, G, beta, W, cal in [(P, asc, G, b, W, cal) for P in a.P for asc in a.alpha_scale
                                        for G in a.stall_G for b in a.beta for W in a.W for cal in a.calibrate]:
            alpha = alpha0 * asc
            ci = [client_inputs(t, P, stall_G=G) for t in traces]
            t_w1 = window_edges(min(duration_s(t) for t in traces))[0] * 1e6
            rp = {}
            fm = (1.0, 1.0)
            if cal == 1:
                fm, tgt = calibrate_foreign(pair, P, alpha, ci, measured, beta, W)
                rp["calibration"] = {"fmult": list(fm), "hedging_median_excess_us": tgt}
            elif cal == 2:
                fm, tgt = calibrate_on_random_w0(pair, P, alpha, ci, measured, beta, W, t_w1)
                rp["calibration"] = {"fmult": list(fm), "random_w0_median_us": tgt,
                                     "fit_on": "random replay, reads arriving before the first later window",
                                     "scored_on": "later windows only"}
            for pol, code in POL.items():
                o = run(pair, P, alpha, code, ci, beta=beta, tmpl_W=W, fmult=fm)
                sim_lat = [o[1], o[5]]
                per = {}
                agg_sim, agg_meas = [], []
                for c in (0, 1):
                    k = ci[c].k_base
                    reads = ci[c].io[:k] == 1
                    if cal == 2:
                        # held out: the later windows (of the baseline replay's timeline, as in the detection)
                        reads = reads & (np.nan_to_num(ci[c].base_sub_us[:k], nan=-1.0) >= t_w1)
                    ms = measured[pol][c][:k]
                    ss = sim_lat[c][:k]
                    both = reads & (ms >= 0) & (ss >= 0)
                    per[f"d{c}"] = {"sim": stats(ss[both]), "replay": stats(ms[both]),
                                    "rerouted_frac_sim": float(np.mean(o[2 + 4 * c][:k][reads] == 1)) if c == 0 else
                                    float(np.mean(o[6][:k][reads] == 1))}
                    agg_sim.append(ss[both]); agg_meas.append(ms[both])
                per["agg"] = {"sim": stats(np.concatenate(agg_sim)), "replay": stats(np.concatenate(agg_meas))}
                rp[pol] = per
                s, m = per["agg"]["sim"], per["agg"]["replay"]
                print(f"{pair} P={P:g} a={asc:g} G={G:g} b={beta:g} W={W} cal={fm} {pol:8s} agg sim/replay p50 {s['p50']:.0f}/{m['p50']:.0f} "
                      f"p99 {s['p99']:.0f}/{m['p99']:.0f} p99.9 {s['p99.9']:.0f}/{m['p99.9']:.0f} | "
                      f"d0 p99.9 {per['d0']['sim']['p99.9']:.0f}/{per['d0']['replay']['p99.9']:.0f} "
                      f"d1 p99.9 {per['d1']['sim']['p99.9']:.0f}/{per['d1']['replay']['p99.9']:.0f}", flush=True)
            key = f"{P:g}" if asc == 1.0 else f"{P:g}|a{asc:g}"
            res["by_P"][key + (f"|g{G:g}" if G > 0 else "") + (f"|b{beta:g}" if beta > 0 else "")
                        + (f"|w{W}" if W > 1 else "") + ("|cal" if cal == 1 else "|cal2" if cal == 2 else "")] = rp
        (out_dir / f"validate_{pair}.json").write_text(json.dumps(stamp(res), indent=1))


if __name__ == "__main__":
    main()
