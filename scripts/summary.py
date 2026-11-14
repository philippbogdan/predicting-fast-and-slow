#!/usr/bin/env python3
"""Every number the paper, README.md and METHOD.md quote, recomputed from the result files.

Reads results/ only (standard library only) and writes results/summary.json.  The paper is typeset
from it (scripts/typeset.py), the figures are drawn through its functions, and check.py runs
them again and compares, so no quoted number can drift from the files.

Conventions, used throughout:
  * A trace's score for a signal is the signal's AUROC in the worst of the four later windows.
  * "Best of nine" is the best of the nine uncertainty estimators on that trace, picked after the fact.
  * A guess is wrong when the read's own latency class disagrees with it, unless period labels are named.
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NINE = ["euclid", "mahal", "conformal100", "conformal1000", "bnn", "margin", "ensemble", "knn", "trust"]
LIVE = ["euclid_aug", "mahal_aug", "conformal100_aug", "conformal1000_aug"]     # the same, given the drive's live state
LATER = ("1", "2", "3", "4")
REQUEST, PERIOD = "signals_G2_request", "signals"      # each read's own latency class; Heimdall's period labels
ARTEFACT_PAIRS = ("P1", "P2", "P3", "P4", "P5", "P6")


def load(path):
    return json.loads(Path(path).read_text())


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def rng(v):
    v = list(v)
    return [min(v), max(v)]


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def later(rec):
    """The later windows a record has, in order."""
    return [rec["windows"][w] for w in LATER if w in rec["windows"]]


# ------------------------------------------------------------------------------------------------
# 1. which guess is wrong: every read (results/heldout/detection, results/development/detection)
# ------------------------------------------------------------------------------------------------
def window_aurocs(d, signal, target=REQUEST):
    v = [w[target].get(signal, {}).get("auroc") for w in later(d)]
    return [a for a in v if finite(a)]


def worst(d, signal, target=REQUEST):
    v = window_aurocs(d, signal, target)
    return min(v) if v else float("nan")


def mean_window(d, signal, target=REQUEST):
    v = window_aurocs(d, signal, target)
    return sum(v) / len(v) if v else float("nan")


def best_of(d, signals, target=REQUEST, f=worst):
    return max(f(d, s, target) for s in signals)


def later_error(d, key):
    n = sum(w["n"] for w in later(d))
    wrong = sum(round(w["n"] * w[key]) for w in later(d))
    return wrong / n, n


def detection_block(files):
    """files: {trace: detection record}."""
    names = sorted(files)
    table = [worst(files[n], "tracker") for n in names]
    best = [best_of(files[n], NINE) for n in names]
    lead = [t - b for t, b in zip(table, best)]
    out = {
        "traces": len(names),
        "table_ahead": sum(d > 0 for d in lead), "table_behind": sum(d < 0 for d in lead),
        "table_range": rng(table), "table_median": st.median(table),
        "table_at_least_080": sum(t >= 0.8 for t in table),
        "best_of_nine_range": rng(best), "best_of_nine_median": st.median(best),
        "best_of_nine_at_least_080": sum(b >= 0.8 for b in best),
        "lead_median": st.median(lead), "lead_range": rng(lead),
        "table_ahead_by_mean_window": sum(mean_window(files[n], "tracker") > best_of(files[n], NINE, f=mean_window)
                                          for n in names),
        "table_ahead_with_live_state_given_to_estimators": sum(
            worst(files[n], "tracker") > best_of(files[n], NINE + LIVE) for n in names),
        "estimators": {},
    }
    for s in NINE:
        w = [worst(files[n], s) for n in names]
        out["estimators"][s] = {
            "median": st.median(w), "range": rng(w), "table_ahead": sum(t > x for t, x in zip(table, w)),
            "median_period_labels": st.median(worst(files[n], s, PERIOD) for n in names)}
    out["estimator_medians_range"] = rng(v["median"] for v in out["estimators"].values())
    out["estimator_medians_range_period_labels"] = rng(v["median_period_labels"] for v in out["estimators"].values())
    ahead = total = 0
    for n in names:
        for w in later(files[n]):
            sig = w[REQUEST]
            if finite(sig["tracker"]["auroc"]):
                total += 1
                ahead += sig["tracker"]["auroc"] > max(sig[s]["auroc"] for s in NINE)
    out["later_windows"], out["table_ahead_in_windows"] = total, ahead
    # Heimdall's own period labels as the truth
    tp = [worst(files[n], "tracker", PERIOD) for n in names]
    bp = [best_of(files[n], NINE, PERIOD) for n in names]
    out["period_labels"] = {"table_ahead": sum(t > b for t, b in zip(tp, bp)),
                            "table_behind": sum(t < b for t, b in zip(tp, bp)),
                            "table_median": st.median(tp), "best_of_nine_median": st.median(bp)}
    # how often the controller is wrong on the later reads
    er = [later_error(files[n], "error_rate_G2_request")[0] for n in names]
    ep = [later_error(files[n], "error_rate")[0] for n in names]
    out["controller_wrong"] = {"own_latency_range": rng(er), "own_latency_median": st.median(er),
                               "period_labels_range": rng(ep), "period_labels_median": st.median(ep)}
    out["later_reads"] = sum(later_error(files[n], "error_rate")[1] for n in names)
    # reference: boosted trees fitted on every labelled read of the first window
    sup = [worst(files[n], "supervised_w0_G2") for n in names]
    out["supervised_reference"] = {"median": st.median(sup), "range": rng(sup),
                                   "table_at_least_reference": sum(t >= s for s, t in zip(sup, table)),
                                   "correlation_with_table": pearson(sup, table)}
    out["admitted_only_change_range"] = rng(worst(files[n], "tracker_admitted") - t for n, t in zip(names, table))
    out["size_only_table_range"] = rng(worst(files[n], "tracker_size_only") for n in names)
    out["size_only_table_median"] = st.median(worst(files[n], "tracker_size_only") for n in names)
    return out


def heldout_records():
    index = load(RES / "heldout" / "traces.json")
    rows = {t["name"]: t for t in index["traces"]}
    recs = {n: load(RES / "heldout" / "detection" / f"{n}.json") for n in rows}
    scored = {n: r for n, r in recs.items() if rows[n]["scored"] and not rows[n]["development_exposed"]}
    return index, rows, recs, scored


def heldout():
    index, rows, recs, scored = heldout_records()
    out = detection_block(scored)
    out["selected"] = len(rows)
    out["development_exposed"] = sorted(n for n in rows if rows[n]["development_exposed"])
    out["unscored"] = sorted(n for n in rows if not rows[n]["scored"])
    out["volumes"] = len({(rows[n]["source"], rows[n]["volume"]) for n in scored})
    out["rejected_attempts"] = index["rejected_attempts"]
    behind = sorted((worst(scored[n], "tracker") - best_of(scored[n], NINE), n) for n in scored)
    out["behind"] = [{"trace": n, "lead": d,
                      "reads_in_table_worst_window": min((w[REQUEST]["tracker"]["auroc"], w["n"])
                                                         for w in later(scored[n]))[1]}
                     for d, n in behind if d < 0]
    out["exposed_pair"] = {n: {"table": worst(recs[n], "tracker"), "best_of_nine": best_of(recs[n], NINE)}
                           for n in out["development_exposed"]}
    return out


def development():
    recs = {p.stem: load(p) for p in sorted((RES / "development" / "detection").glob("*.json"))}
    device = {n: r for n, r in recs.items() if n[0] == "P"}
    raw = {n: r for n, r in recs.items() if n[0] != "P"}
    out = {"device_traces": detection_block(device), "raw_replays": detection_block(raw)}
    out["raw_replays"]["table_by_trace"] = {n: worst(raw[n], "tracker") for n in sorted(raw)}
    pipe = [load(p) for p in sorted((RES / "development" / "pipeline").glob("*.json"))]
    out["rows_filtered_as_noise_range"] = rng(1 - p["outlier_counts"]["0"] / sum(p["outlier_counts"].values())
                                              for p in pipe)
    out["artefact_accuracy"] = {
        "traces": len(pipe), "reported_range": rng(p["model_acc_ref"] for p in pipe),
        "reproduced_max_abs_difference": max(abs(p["model_acc_ref"] - p["model_acc_mine"]) for p in pipe),
        "labels_and_features_identical": all(p["matches_artefact"] and p["labels_agree_frac"] == 1.0
                                             and min(p["feat_col_agree"].values()) == 1.0 for p in pipe)}
    return out


def example_table():
    """The table one held-out trace taught, and the two cells Figure 2 names."""
    d = load(RES / "heldout" / "table" / "msr.mds_0.orig.json")
    cells = {(c["size_class"], c["load_class"], c["age_class"], c["quiet"]): c for c in d["cells"]}
    calm, storm = cells[(12, 0, 0, 0)], cells[(16, 7, 0, 1)]

    def describe(c):
        return {"q": c["q"], "reads": c["reads"], "slow_share": c["slow"] / c["reads"],
                "heimdall_called_slow_share": c["heimdall_called_slow"] / c["reads"]}
    return {"trace": d["trace"], "cells_visited": d["cells_visited"],
            "small_read_idle_drive": describe(calm), "large_read_backlog_after_quiet": describe(storm),
            "reference_implementation_lines": len((ROOT / "admission" / "tracker.py").read_text().splitlines())}


def cost():
    c = load(RES / "cost" / "cost.json")
    return {"cpu": c["cpu"], "decisions": c["n_decisions"], "table_ns": c["tracker_only_ns"],
            "table_in_decision_path_ns": c["insitu_tracker_ns"], "euclidean_ns": c["euclid_only_ns"],
            "mahalanobis_ns": c["mahal_only_ns"], "conformal_100_ns": c["conformal100_only_ns"],
            "conformal_1000_ns": c["conformal_full_only_ns"], "heimdall_inference_ns": c["model_ns"],
            "table_bytes_single_precision": 32 * 16 * 4 * 2 * 4}


# ------------------------------------------------------------------------------------------------
# 2. the same with the direction of the guess held fixed (results/classwise)
# ------------------------------------------------------------------------------------------------
PARTS = ("pooled", "called_fast", "called_slow", "same_size_pooled", "same_size_called_fast", "same_size_called_slow")
CONTROLS = ["said_slow", "one_cell", "tracker_size_only", "feedback16", "supervised_w0_G2"]


def classwise_block(recs, truth="own_latency"):
    names = sorted(recs)

    def w(n, k, part):
        v = [x[truth]["signals"][k][part] for x in later(recs[n])]
        v = [a for a in v if finite(a)]
        return min(v) if v else float("nan")

    out = {"traces": len(names)}
    for part in PARTS:
        have = [n for n in names if all(finite(w(n, k, part)) for k in ["tracker"] + NINE)]
        table = [w(n, "tracker", part) for n in have]
        best = [max(w(n, k, part) for k in NINE) for n in have]
        blk = {"traces": len(have), "table_median": st.median(table), "table_range": rng(table),
               "best_of_nine_median": st.median(best),
               "table_ahead": sum(t > b for t, b in zip(table, best)),
               "lead_median": st.median(t - b for t, b in zip(table, best)), "signals": {}}
        for k in NINE + CONTROLS:
            v = [w(n, k, part) for n in have]
            blk["signals"][k] = {"median": st.median(v), "table_ahead": sum(t > x for t, x in zip(table, v))}
        blk["estimator_medians_range"] = rng(blk["signals"][k]["median"] for k in NINE)
        blk["table_ahead_of_each_range"] = rng(blk["signals"][k]["table_ahead"] for k in NINE)
        one = [w(n, "one_cell", part) for n in have]
        blk["one_cell_ahead_of_best_of_nine"] = sum(o > b for o, b in zip(one, best))
        blk["one_cell_ahead_of_each_range"] = rng(sum(o > w(n, k, part) for o, n in zip(one, have)) for k in NINE)
        out[part] = blk
    # how the controller's calls compare with what the reads did
    calls, slow, slow_call_share = [], [], []
    for n in names:
        W = later(recs[n])
        tot = sum(x["n"] for x in W)
        calls.append(sum(x["called_slow"] for x in W) / tot)
        slow.append(sum(x[truth]["slow"] for x in W) / tot)
        fa = sum(x[truth]["called_slow_and_fast"] for x in W)
        fn = sum(x[truth]["called_fast_and_slow"] for x in W)
        if fa + fn:
            slow_call_share.append(fa / (fa + fn))
    out["heimdall_calls_slow_median"] = st.median(calls)
    out["reads_slow_median"] = st.median(slow)
    out["mistakes_that_are_slow_calls_median"] = st.median(slow_call_share)
    # the study's two gates as they act: a read is flagged above the 90th percentile of the training distances
    out["gates"] = {}
    for g in ("euclid", "mahal"):
        flagged, wrong_f, wrong_u, slow_f, slow_u = [], [], [], [], []
        for n in names:
            W = later(recs[n])
            G = [x[truth]["gates"][g] for x in W]
            tot, f = sum(x["n"] for x in W), sum(x["flagged"] for x in G)
            flagged.append(f / tot)
            if 0 < f < tot:
                wrong_f.append(sum(x["flagged_wrong"] for x in G) / f)
                wrong_u.append(sum(x["unflagged_wrong"] for x in G) / (tot - f))
            cf = sum(x["called_fast_flagged"] for x in G)
            cu = sum(x["n"] - x["called_slow"] for x in W) - cf
            if cf > 0 and cu > 0:
                slow_f.append(sum(x["called_fast_flagged_slow"] for x in G) / cf)
                slow_u.append(sum(x["called_fast_unflagged_slow"] for x in G) / cu)
        out["gates"][g] = {
            "flagged_median": st.median(flagged), "flagged_range": rng(flagged), "traces": len(wrong_f),
            "wrong_when_flagged_median": st.median(wrong_f), "wrong_when_passed_median": st.median(wrong_u),
            "flagged_more_often_wrong": sum(a > b for a, b in zip(wrong_f, wrong_u)),
            # trace by trace: how much more often a flagged guess is wrong than an unflagged one
            "flagged_minus_passed_median": st.median(a - b for a, b in zip(wrong_f, wrong_u)),
            "admitted_traces": len(slow_f),
            "admitted_slow_when_flagged_median": st.median(slow_f),
            "admitted_slow_when_passed_median": st.median(slow_u),
            "admitted_flagged_more_often_slow": sum(a > b for a, b in zip(slow_f, slow_u))}
    out["euclidean_on_unnormalised_inputs"] = {
        part: st.median(v for v in (w(n, "euclid_raw", part) for n in names) if finite(v))
        for part in ("pooled", "called_fast", "called_slow")}
    out["share_of_later_reads_in_commonest_size_class_median"] = st.median(
        recs[n]["later_reads_of_that_size"] / sum(x["n"] for x in later(recs[n])) for n in names)
    return out


def recomputed():
    """Largest difference, per signal, between the every-read AUROC recomputed for results/classwise and the
    one in the detection files.  The table and the distances are deterministic; the Bayesian network is resampled."""
    out = {}
    for group, pub_dir in (("heldout", RES / "heldout" / "detection"),
                           ("development", RES / "development" / "detection")):
        diff = {}
        for p in sorted((RES / "classwise" / group).glob("*.json")):
            mine, pub = load(p), load(pub_dir / p.name)
            for x, row in mine["windows"].items():
                for truth, key in (("own_latency", REQUEST), ("period_labels", PERIOD)):
                    for k, v in row[truth]["signals"].items():
                        ref = pub["windows"][x][key].get(k, {}).get("auroc")
                        if finite(ref) and finite(v["pooled"]):
                            diff[k] = max(diff.get(k, 0.0), abs(ref - v["pooled"]))
        out[group] = {"largest": max(diff.values()), "largest_signal": max(diff, key=diff.get),
                      "largest_excluding_bnn": max(v for k, v in diff.items() if k != "bnn"),
                      "table": diff["tracker"], "euclid": diff["euclid"], "mahal": diff["mahal"]}
    return out


def classwise_records(group):
    recs = {p.stem: load(p) for p in sorted((RES / "classwise" / group).glob("*.json"))}
    if group == "heldout":
        return {n: r for n, r in recs.items() if "hm_0" not in n}
    return {n: r for n, r in recs.items() if n[0] == "P"}


def classwise():
    held, dev = classwise_records("heldout"), classwise_records("development")
    return {"heldout": classwise_block(held), "heldout_period_labels": classwise_block(held, "period_labels"),
            "development_device_traces": classwise_block(dev),
            "development_device_traces_period_labels": classwise_block(dev, "period_labels"),
            "recomputed_against_detection_files": recomputed()}


# ------------------------------------------------------------------------------------------------
# 3. the table as a predictor of slow reads (results/direct)
# ------------------------------------------------------------------------------------------------
def direct_block(recs):
    names = sorted(recs)

    def w(n, p, key):
        v = [x[p][key] for x in later(recs[n]) if p in x and finite(x[p][key])]
        return min(v) if v else float("nan")

    def wrong(n, p):
        return sum(x[p]["wrong"] for x in later(recs[n])) / sum(x["n"] for x in later(recs[n]))

    out = {"traces": len(names), "max_difference_from_detection_files": max(
        recs[n]["detection_file_max_abs_difference"] for n in names)}
    for key, label in (("auroc_slow", "every_read"), ("auroc_slow_same_size", "same_size")):
        blk = {}
        for p in ("table", "heimdall", "net", "net_live"):
            have = [n for n in names if finite(w(n, p, key)) and finite(w(n, "table", key))]
            v = [w(n, p, key) for n in have]
            blk[p] = {"traces": len(have), "median": st.median(v), "range": rng(v)}
            if p != "table":
                d = [w(n, "table", key) - w(n, p, key) for n in have]
                blk[p]["table_ahead"] = sum(x > 0 for x in d)
                blk[p]["table_lead_median"] = st.median(d)
        out[label] = blk
    out["wrong_calls_median"] = {p: st.median(wrong(n, p) for n in names if p in later(recs[n])[0])
                                 for p in ("table", "heimdall", "net", "net_live")}

    # the forecast that never says slow: wrong on exactly the reads that come back slow
    def never(n):
        return sum(x["slow"] for x in later(recs[n])) / sum(x["n"] for x in later(recs[n]))
    out["wrong_calls_median"]["never_slow"] = st.median(never(n) for n in names)
    out["wrong_more_often_than_never_slow"] = {p: sum(wrong(n, p) > never(n) for n in names if p in later(recs[n])[0])
                                               for p in ("table", "heimdall", "net", "net_live")}
    return out


def direct():
    held = {p.stem: load(p) for p in sorted((RES / "direct" / "heldout").glob("*.json")) if "hm_0" not in p.stem}
    dev = {p.stem: load(p) for p in sorted((RES / "direct" / "development").glob("*.json"))}
    return {"heldout": direct_block(held),
            "development_device_traces": direct_block({n: r for n, r in dev.items() if n[0] == "P"}),
            "development_raw_replays": direct_block({n: r for n, r in dev.items() if n[0] != "P"})}


# ------------------------------------------------------------------------------------------------
# 4. what cuts the tail (results/routing)
# ------------------------------------------------------------------------------------------------
RULE, TIMEOUT, TABLE_GATE = "tracker-mixed+timeout", "timeout-hedging", "tracker-mixed"
NO_TIMEOUT = ["heimdall", "euclid", "mahal", TABLE_GATE]
WITH_TIMEOUT = [TIMEOUT, "heimdall+timeout", "euclid+timeout", "mahal+timeout", RULE]
TABLE_ONLY = "tracker-hedgeslow"       # hedge exactly when the table's entry is at least 1/2: no network in the decision


def artefact_replays():
    """Heimdall's artefact: its own replay logs of four policies on two NVMe drives, six pairs."""
    a = load(RES / "routing" / "artefact_replays.json")["pairs"]
    out = {"pairs": {}}
    for pair, pol in a.items():
        b = pol["baseline"]["agg"]["replay"]
        out["pairs"][pair] = {
            name: {m: pol[key]["agg"]["replay"][m] / b[m] for m in ("mean", "p50", "p99", "p99.9")}
            for name, key in (("heimdall", "flashnet"), ("hedging", "hedging"), ("random", "random"))}
        out["pairs"][pair]["reads"] = b["n"]
    P = list(out["pairs"].values())
    out["hedging_p999_range"] = rng(p["hedging"]["p99.9"] for p in P)
    out["heimdall_p999_range"] = rng(p["heimdall"]["p99.9"] for p in P)
    out["heimdall_over_hedging_p999_range"] = rng(p["heimdall"]["p99.9"] / p["hedging"]["p99.9"] for p in P)
    out["hedging_lower_p999_pairs"] = sum(p["hedging"]["p99.9"] < p["heimdall"]["p99.9"] for p in P)
    out["heimdall_mean_range"] = rng(p["heimdall"]["mean"] for p in P)
    out["hedging_mean_range"] = rng(p["hedging"]["mean"] for p in P)
    out["heimdall_lower_mean_pairs"] = sum(p["heimdall"]["mean"] < p["hedging"]["mean"] for p in P)

    # how well the simulator reproduces those logs
    def err(pol, m):
        return [abs(a[p][pol]["agg"]["simulated"][m] / a[p][pol]["agg"]["replay"][m] - 1) for p in a]
    out["simulator"] = {
        "comparisons": 3 * 2 * len(a),
        "worst_error_baseline_random_hedging": max(max(err(pol, m)) for pol in ("baseline", "random", "hedging")
                                                   for m in ("p50", "p99.9")),
        "heimdall_p999_error_by_pair": {p: a[p]["flashnet"]["agg"]["simulated"]["p99.9"]
                                        / a[p]["flashnet"]["agg"]["replay"]["p99.9"] - 1 for p in a}}
    return out


def timeout_floor(baselines, H):
    """The soonest the timeout alone can answer the reads that make the tail, over the tail.

    baselines: one record a round of the run with no admission control, {"p99.9", "clients": [{"p50", "p99.9",
    "above_pooled_p999"}] * 2, "capped_p999"}.  The tail client of a round is the one with more reads above the
    pooled p99.9 (client 0 on a tie).  A read of it unanswered after its H is copied to the other drive, so it
    returns no sooner than H plus one read there: the floor is H of the tail client plus the other drive's median,
    over the pooled p99.9.  "capped" is the pooled p99.9 with every read capped that way, over the pooled p99.9."""
    floor, capped, tail = [], [], []
    for b in baselines:
        a = [c["above_pooled_p999"] for c in b["clients"]]
        t = 0 if a[0] >= a[1] else 1
        tail.append(t)
        floor.append((H[t] + b["clients"][1 - t]["p50"]) / b["p99.9"])
        capped.append(b["capped_p999"] / b["p99.9"])
    return {"H_us": H, "tail_client_rounds": tail,
            "tail_share_rounds": [b["clients"][t]["above_pooled_p999"] / sum(c["above_pooled_p999"] for c in b["clients"])
                                  for b, t in zip(baselines, tail)],
            "drive_p50_us_median": [st.median(b["clients"][c]["p50"] for b in baselines) for c in (0, 1)],
            "drive_p999_us_median": [st.median(b["clients"][c]["p99.9"] for b in baselines) for c in (0, 1)],
            "floor_rounds": floor, "floor_median": st.median(floor),
            "capped_rounds": capped, "capped_median": st.median(capped)}


def trace_lengths():
    """Each pair's trace length in seconds (the same nine pairs are replayed on every testbed)."""
    return {p: v["trace_s"] for p, v in load(RES / "routing" / "enterprise_nvme.json")["pairs"].items()}


def quiet_drives():
    """Two unlike drives of one quiet machine: every policy, five rounds a pair."""
    q = load(RES / "routing" / "quiet_drives.json")
    length = trace_lengths()
    out = {"accepted_runs": q["accepted_runs"], "rejected_attempts": q["rejected_attempts"], "pairs": {}}
    for pair, pol in q["pairs"].items():
        rounds = sorted(pol["baseline"])

        def val(p, r, m="p99.9"):
            return pol[p][r]["pooled"][m]

        def rel(p, m="p99.9"):
            return [val(p, r, m) / val("baseline", r, m) for r in rounds]

        dist = [min(val("euclid", r), val("mahal", r)) for r in rounds]
        out["pairs"][pair] = {
            "rounds": len(rounds),
            "baseline_p999_ms_median": st.median(val("baseline", r) for r in rounds) / 1000,
            "p999_over_baseline_median": {p: st.median(rel(p)) for p in NO_TIMEOUT + WITH_TIMEOUT},
            "p999_over_baseline_rounds": {p: rel(p) for p in NO_TIMEOUT + WITH_TIMEOUT},
            "mean_over_baseline_median": {p: st.median(rel(p, "mean")) for p in NO_TIMEOUT + WITH_TIMEOUT},
            "moved_median": {p: st.median(val(p, r, "rerouted") for r in rounds) for p in NO_TIMEOUT + WITH_TIMEOUT},
            "distance_cut_of_heimdall_p999_median": st.median(1 - d / val("heimdall", r)
                                                              for d, r in zip(dist, rounds)),
            # every run of the pair, each policy in each round: how long the replay took over the trace's length
            "replay_over_trace_median": st.median(pol[p][r]["replay_seconds"] for p in pol for r in pol[p])
            / length[pair],
            "timeout_floor": timeout_floor(
                [{"p99.9": pol["baseline"][r]["pooled"]["p99.9"], "capped_p999": pol["baseline"][r]["capped_p999"],
                  "clients": [pol["baseline"][r][f"client{c}"]
                              | {"above_pooled_p999": pol["baseline"][r]["clients_above_pooled_p999"][c]}
                              for c in (0, 1)]} for r in rounds], q["H_us"][pair]),
        }
    art = [out["pairs"][p] for p in ("P2", "P5", "P6")]
    out["artefact_pairs"] = ["P2", "P5", "P6"]
    out["with_timeout_p999_range"] = rng(a["p999_over_baseline_median"][p] for a in art for p in WITH_TIMEOUT)
    out["gates_without_timeout_p999_range_P2_P5"] = rng(
        out["pairs"][x]["p999_over_baseline_median"][p] for x in ("P2", "P5") for p in NO_TIMEOUT)
    out["timeout_alone_p999_range"] = rng(a["p999_over_baseline_median"][TIMEOUT] for a in art)
    out["distance_cut_of_heimdall_p999_range"] = rng(a["distance_cut_of_heimdall_p999_median"] for a in art)
    out["timeout_floor_range"] = rng(a["timeout_floor"]["floor_median"] for a in art)
    out["baseline_p999_ms_range"] = rng(a["baseline_p999_ms_median"] for a in art)
    return out


ORACLES = ["oracle-hedge+timeout", "oracle-flip+timeout", "oracle-mixed+timeout", "oracle-hedgeslow+timeout",
           "oracle-hedgefast+timeout"]
DIST_T = ["euclid+timeout", "mahal+timeout"]
GATES_T = DIST_T + [RULE]                       # the gates given the timeout that the oracle is set against
EVERY = NO_TIMEOUT + [TABLE_ONLY] + WITH_TIMEOUT + ORACLES


def whole_rounds(e, pair):
    """The rounds in which every policy of the pair has a record: a kept run, or missing after four attempts."""
    return [r for r, v in e["pairs"][pair]["rounds"].items() if len(v) == len(e["policies"])]


def drive_block(rounds):
    """One pair's comparisons over the given rounds ({round: {policy: record}}), each ratio taken within a round.
    A policy missing from a round, or a round without its baseline, gives no ratio there."""
    def val(r, p, m="p99.9"):
        x = rounds[r].get(p)
        return x["pooled"][m] if x and x["kind"] != "missing" else None

    def per_round(f):
        out = []
        for r in rounds:
            try:
                out.append(f(r))
            except TypeError:                   # a value the round does not have
                pass
        return out

    def med(v):
        return st.median(v) if v else None

    rel = {p: per_round(lambda r, p=p: val(r, p) / val(r, "baseline")) for p in EVERY}
    mean = {p: per_round(lambda r, p=p: val(r, p, "mean") / val(r, "baseline", "mean")) for p in EVERY}
    moved = {p: per_round(lambda r, p=p: val(r, p, "rerouted") + 0) for p in EVERY}
    dist_over_rule = per_round(lambda r: min(val(r, p) for p in DIST_T) / val(r, RULE))
    oracle_over_gate = per_round(lambda r: min(val(r, p) for p in ORACLES) / min(val(r, p) for p in GATES_T))
    oracle = per_round(lambda r: min(val(r, p) for p in ORACLES) / val(r, "baseline"))
    cut = per_round(lambda r: 1 - min(val(r, "euclid"), val(r, "mahal")) / val(r, "heimdall"))
    return {
        "rounds": list(rounds),
        "baseline_p999_ms_median": med(per_round(lambda r: val(r, "baseline") / 1000)),
        "p999_over_baseline_rounds": rel,
        "p999_over_baseline_median": {p: med(v) for p, v in rel.items()},
        "mean_over_baseline_median": {p: med(v) for p, v in mean.items()},
        "moved_median": {p: med(v) for p, v in moved.items()},
        "distance_over_rule_same_timeout_rounds": dist_over_rule,
        "distance_over_rule_same_timeout_median": med(dist_over_rule),
        "oracle_p999_over_baseline_rounds": oracle,
        "oracle_over_best_gate_rounds": oracle_over_gate,
        "oracle_over_best_gate_median": med(oracle_over_gate),
        "distance_cut_of_heimdall_p999_rounds": cut,
        "distance_cut_of_heimdall_p999_median": med(cut),
    }


def enterprise_nvme():
    """Two identical enterprise NVMe drives of one machine: sixteen policies on nine pairs, in rounds.

    A round counts for a pair when it is whole for that pair.  "all kept": every kept run, quiet or shared.
    "quiet only": the rounds in which every policy's kept run is quiet."""
    e = load(RES / "routing" / "enterprise_nvme.json")
    out = {"pairs": {}, "quiet_only": {}, "runs": {}}
    for pair, d in e["pairs"].items():
        counted = {r: d["rounds"][r] for r in whole_rounds(e, pair)}
        quiet = {r: v for r, v in counted.items() if all(x["kind"] == "quiet" for x in v.values())}
        kept = [x for v in counted.values() for x in v.values() if x["kind"] != "missing"]
        out["pairs"][pair] = drive_block(counted)
        base = [v["baseline"]["pooled"] for v in counted.values() if v["baseline"]["kind"] != "missing"]
        out["pairs"][pair]["timeout_floor"] = timeout_floor(base, d["H_us"]) if base else None
        out["pairs"][pair]["replay_over_trace_median"] = (st.median(x["replay_s"] for x in kept) / d["trace_s"]
                                                          if kept else None)
        out["pairs"][pair]["reads"] = sorted({x["pooled"]["n"] for x in kept})
        out["pairs"][pair]["reads_above_p999"] = rng(x["pooled"]["above_p999"] for x in kept) if kept else None
        out["quiet_only"][pair] = drive_block(quiet)
        out["runs"][pair] = {r: {k: sum(x["kind"] == k for x in v.values()) for k in ("quiet", "shared", "missing")}
                             | {"rejected_attempts": sum(x["rejected_attempts"] for x in v.values())}
                             for r, v in counted.items()}
    runs = [c for p in out["runs"].values() for c in p.values()]
    out["rounds_counted"] = {p: len(v) for p, v in out["runs"].items()}
    out["kept_runs"] = sum(c["quiet"] + c["shared"] for c in runs)
    out["shared_runs"] = sum(c["shared"] for c in runs)
    out["missing_runs"] = sum(c["missing"] for c in runs)
    out["missing"] = [{"pair": pair, "round": r, "policy": p} for pair, d in e["pairs"].items()
                      for r in whole_rounds(e, pair) for p, x in d["rounds"][r].items() if x["kind"] == "missing"]
    # the policies the paper compares on the six artefact pairs: how many of the rounds counted hold a kept run of each
    compared = DIST_T + [RULE] + ORACLES + ["heimdall", "euclid", "mahal"]
    out["compared_policies_kept_rounds_range"] = rng(
        sum(e["pairs"][pair]["rounds"][r][p]["kind"] != "missing" for r in whole_rounds(e, pair))
        for pair in ARTEFACT_PAIRS for p in compared)
    out["rejected_attempts"] = sum(c["rejected_attempts"] for c in runs)

    A = [out["pairs"][p] for p in ARTEFACT_PAIRS]

    def over(key, pols, pairs=A):
        return rng(a[key][p] for a in pairs for p in pols)
    out["artefact"] = {
        "timeout_alone_p999_range": over("p999_over_baseline_median", [TIMEOUT]),
        "heimdall_p999_range": over("p999_over_baseline_median", ["heimdall"]),
        "heimdall_timeout_p999_range": over("p999_over_baseline_median", ["heimdall+timeout"]),
        "distance_timeout_p999_range": over("p999_over_baseline_median", DIST_T),
        "rule_p999_range": over("p999_over_baseline_median", [RULE]),
        "table_only_p999_range": over("p999_over_baseline_median", [TABLE_ONLY]),
        "distance_timeout_moved_range": over("moved_median", DIST_T),
        "rule_moved_range": over("moved_median", [RULE]),
        "heimdall_moved_range": over("moved_median", ["heimdall"]),
        "table_only_moved_range": over("moved_median", [TABLE_ONLY]),
        "heimdall_lower_mean_than_table_only_pairs": sum(
            a["mean_over_baseline_median"]["heimdall"] < a["mean_over_baseline_median"][TABLE_ONLY] for a in A),
        "distance_over_rule_same_timeout_range": rng(a["distance_over_rule_same_timeout_median"] for a in A),
        "oracle_over_best_gate_range": rng(a["oracle_over_best_gate_median"] for a in A),
        "distance_cut_of_heimdall_p999_range": rng(a["distance_cut_of_heimdall_p999_median"] for a in A),
        "timeout_floor_range": rng(a["timeout_floor"]["floor_median"] for a in A),
        "baseline_p999_ms_range": rng(a["baseline_p999_ms_median"] for a in A),
    }
    k, s1, s3 = out["pairs"]["K"], out["pairs"]["S1"], out["pairs"]["S3"]
    out["K"] = {"euclid+timeout": k["p999_over_baseline_median"]["euclid+timeout"],
                "tracker-mixed+timeout": k["p999_over_baseline_median"][RULE],
                "euclid+timeout_moved": k["moved_median"]["euclid+timeout"]}
    out["short_samples"] = {
        "reads_above_p999": {"S1": s1["reads_above_p999"], "S3": s3["reads_above_p999"]},
        "gates_timeout_p999_range": over("p999_over_baseline_median", ["heimdall+timeout"] + GATES_T, [s1, s3]),
        "timeout_alone_p999": {"S1": s1["p999_over_baseline_median"][TIMEOUT],
                               "S3": s3["p999_over_baseline_median"][TIMEOUT]}}
    return out


def identical_nvme():
    """Two identical NVMe drives of a shared machine: one run a pair (two for K, S1, S3)."""
    d = load(RES / "routing" / "identical_nvme.json")["pairs"]
    out = {"pairs": {}}
    for pair, runs in d.items():
        for run, rec in runs.items():
            pol = rec["policies"]
            b = pol["baseline"]["p99.9"]
            out["pairs"].setdefault(pair, {})[run] = {
                "reads": pol["baseline"]["n"], "baseline_p999_ms": b / 1000,
                "p999_over_baseline": {p: v["p99.9"] / b for p, v in pol.items() if p != "baseline"},
                "moved": {p: v["rerouted"] for p, v in pol.items() if p != "baseline"}}
    first = {p: r["first"]["p999_over_baseline"] for p, r in out["pairs"].items()}
    for pair in ("P2", "P3"):               # Heimdall with the timeout was not run on these drives
        out[f"{pair}_with_timeout_range"] = rng(first[pair][p] for p in WITH_TIMEOUT if p in first[pair])
    out["P1"] = {TIMEOUT: first["P1"][TIMEOUT]}
    return out


def simulated():
    """The simulator's version of the same policies, and of hedging on the table's call alone."""
    out = {"pairs": {}}
    keep = ["heimdall", "euclid", "mahal", TABLE_GATE, TABLE_ONLY, TIMEOUT, "heimdall+timeout", "euclid+timeout",
            "mahal+timeout", RULE]
    for p in sorted((RES / "routing" / "simulated").glob("*.json")):
        pol = {k: v["pooled"] for k, v in load(p)["policies"].items()}
        b = pol["baseline"]
        out["pairs"][p.stem] = {k: {**{m: pol[k][m] / b[m] for m in ("mean", "p99", "p99.9")},
                                    "moved": pol[k]["rerouted"], "duplicated": pol[k]["hedged"]} for k in keep}
    A = [out["pairs"][p] for p in ARTEFACT_PAIRS]
    out["distance_cut_of_heimdall_p999_range"] = rng(
        1 - min(a["euclid"]["p99.9"], a["mahal"]["p99.9"]) / a["heimdall"]["p99.9"] for a in A)
    return out


# ------------------------------------------------------------------------------------------------
# 5. two other decisions, with controllers of our own trained on each request's outcome
# ------------------------------------------------------------------------------------------------
def other_decisions():
    out = {}
    for task in ("rpc_hedging", "storage_server_hedging"):
        recs = {p.stem: load(p) for p in sorted((RES / "other_decisions" / task).glob("*.json"))}
        table = [worst(r, "tracker", "signals") for r in recs.values()]
        best = [best_of(r, NINE, "signals") for r in recs.values()]
        out[task] = {"traces": len(recs), "table_range": rng(table), "best_of_nine_range": rng(best),
                     "table_ahead": sum(t > b for t, b in zip(table, best)),
                     "margin_range": rng(worst(r, "margin", "signals") for r in recs.values()),
                     "distance_range": rng(worst(r, k, "signals") for r in recs.values() for k in ("euclid", "mahal")),
                     "by_trace": {n: {"table": worst(r, "tracker", "signals"),
                                      **{k: worst(r, k, "signals") for k in NINE}} for n, r in recs.items()}}
    both = list(out.values())
    out["distance_range"] = rng(x for o in both for x in o["distance_range"])
    return out


# ------------------------------------------------------------------------------------------------
# 6. Heimdall retrained during the replay (results/retraining, scripts/retraining)
# ------------------------------------------------------------------------------------------------
RETRAINED = ("heimdall_s7", "minute", "slide10", "grow10")
HEIMDALL_TRIGGER = 0.80      # Heimdall's Section 7: it retrains when the model's accuracy drops below 80%


def retrained_block(recs):
    """One schedule on the held-out traces: the paper's statistics, each trace scored in its worst later window."""
    names = sorted(recs)

    def w(n, k, part):
        v = [x["signals"][k][part] for x in later(recs[n])]
        v = [a for a in v if finite(a)]
        return min(v) if v else float("nan")

    def share(n, key):
        W = later(recs[n])
        return sum(x[key] for x in W) / sum(x["n"] for x in W)

    out = {"traces": len(names)}
    for part in ("pooled", "called_fast", "called_slow"):
        have = [n for n in names if all(finite(w(n, k, part)) for k in ["tracker"] + NINE)]
        table = [w(n, "tracker", part) for n in have]
        best = [max(w(n, k, part) for k in NINE) for n in have]
        out[part] = {"traces": len(have), "table_median": st.median(table), "best_of_nine_median": st.median(best),
                     "table_ahead": sum(t > b for t, b in zip(table, best)),
                     "lead_median": st.median(t - b for t, b in zip(table, best)),
                     "estimator_medians_range": rng(st.median(w(n, k, part) for n in have) for k in NINE)}
    out["heimdall_wrong_median"] = st.median(share(n, "wrong") for n in names)
    out["heimdall_calls_slow_median"] = st.median(share(n, "called_slow") for n in names)
    out["reads_slow_median"] = st.median(share(n, "slow") for n in names)
    out["calls_minus_reads_median"] = st.median(share(n, "called_slow") - share(n, "slow") for n in names)
    out["calls_slow_more_often_than_reads_are"] = sum(share(n, "called_slow") > share(n, "slow") for n in names)
    # the schedule as run
    segs = {n: recs[n]["segments"][1:] for n in names}
    out["schedule"] = {
        "specs": [json.loads(x) for x in sorted({json.dumps(recs[n]["spec"], sort_keys=True) for n in names})],
        "stretches_s": sorted({tuple(s["stretch"]) for n in names for s in segs[n]}),
        "traces_refitted": sum(any(s["retrained"] for s in segs[n]) for n in names),
        "refits": sum(s["retrained"] for n in names for s in segs[n]),
        "scheduled": sum(len(segs[n]) for n in names)}
    checked = [s for n in names for s in segs[n] if "accuracy_last_minute" in s]
    if checked:
        out["schedule"]["checked"] = len(checked)
        out["schedule"]["refitted_when_checked"] = sum(s["retrained"] for s in checked)
        out["schedule"]["accuracy_refitted_max"] = max(
            (s["accuracy_last_minute"] for s in checked if s["retrained"]), default=None)
        out["schedule"]["accuracy_kept_min"] = min(
            (s["accuracy_last_minute"] for s in checked if not s["retrained"]), default=None)
    return out


def retraining():
    """Heimdall's network and the nine estimators refitted during the replay, on four schedules."""
    traces = {t["name"]: t for t in load(RES / "heldout" / "traces.json")["traces"]}
    paper = classwise_records("heldout")
    arms = {a: {p.stem: load(p) for p in sorted((RES / "retraining" / a).glob("*.json"))} for a in RETRAINED}
    out = {"arms": {a: retrained_block(r) for a, r in arms.items()}, "heimdall_trigger_below": HEIMDALL_TRIGGER}
    names = sorted(set().union(*arms.values()))
    out["traces"] = len(names)
    out["traces_in_every_arm"] = sum(all(n in r for r in arms.values()) for n in names)
    # the replays are the paper's, and every window scores the paper's reads under the paper's truth
    out["replays_as_recorded"] = sum(
        all(r[n]["replay_sha256"] == traces[n]["replay_sha256"] for r in arms.values()) for n in names)
    out["windows_as_scored"] = sum(
        all([(x["n"], x["slow"]) for x in later(r[n])]
            == [(x["n"], x["own_latency"]["slow"]) for x in later(paper[n])] for r in arms.values())
        for n in names)
    b = out["arms"].values()
    out["pooled_table_median_range"] = rng(x["pooled"]["table_median"] for x in b)
    out["pooled_table_ahead_range"] = rng(x["pooled"]["table_ahead"] for x in b)
    out["calls_minus_reads_median_range"] = rng(x["calls_minus_reads_median"] for x in b)
    out["called_fast_table_median_range"] = rng(x["called_fast"]["table_median"] for x in b)
    out["called_fast_best_of_nine_median_range"] = rng(x["called_fast"]["best_of_nine_median"] for x in b)
    out["called_fast_lead_median_range"] = rng(x["called_fast"]["lead_median"] for x in b)
    return out


def build():
    return {"heldout": heldout(), "development": development(), "example_table": example_table(), "cost": cost(),
            "classwise": classwise(),
            "direct": direct(), "artefact_replays": artefact_replays(), "enterprise_nvme": enterprise_nvme(),
            "quiet_drives": quiet_drives(), "identical_nvme": identical_nvme(), "simulated": simulated(),
            "other_decisions": other_decisions(), "study_reported": load(RES / "study_reported.json"),
            "heimdall_reported": load(RES / "heimdall_reported.json"), "retraining": retraining()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    s = build()
    (RES / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    if a.print:
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
