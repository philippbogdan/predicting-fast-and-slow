#!/usr/bin/env python3
"""results/tables.md: every trace and every drive measurement in one page, from results/.

Usage: tables.py [--check]   (standard library only; check.py verifies the page is current)
"""
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summary as S  # noqa: E402

OUT = S.RES / "tables.md"
NAMES = {"euclid": "Euclidean", "mahal": "Mahalanobis", "conformal100": "conformal 100", "bnn": "Bayesian",
         "conformal1000": "conformal 1,000", "margin": "margin", "ensemble": "ensemble", "knn": "nearest neighbour",
         "trust": "trust score"}
# Heimdall with an estimator or the table choosing which guesses to distrust, each without and with hedging
POLICIES = [("heimdall", "Heimdall"), ("euclid", "Heimdall + Euclidean"), ("mahal", "Heimdall + Mahalanobis"),
            ("tracker-mixed", "Heimdall + table"), ("timeout-hedging", "hedging alone"),
            ("heimdall+timeout", "Heimdall, hedging"), ("euclid+timeout", "Heimdall + Euclidean, hedging"),
            ("mahal+timeout", "Heimdall + Mahalanobis, hedging"), ("tracker-mixed+timeout", "Heimdall + table, hedging")]
EVERY = POLICIES[:4] + [("tracker-hedgeslow", "table alone")] + POLICIES[4:] + [
    ("oracle-hedge+timeout", "oracle: both drives, hedging"), ("oracle-flip+timeout", "oracle: flip, hedging"),
    ("oracle-mixed+timeout", "oracle: mixed, hedging"),
    ("oracle-hedgeslow+timeout", "oracle: both drives if called slow, hedging"),
    ("oracle-hedgefast+timeout", "oracle: both drives if called fast, hedging")]


def f(x, n=2):
    return "" if not S.finite(x) else f"{x:.{n}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def trace_rows(detection, classwise, direct):
    rows = []
    for n in sorted(detection):
        p = detection[n]
        if "windows" not in p:
            rows.append([n, p["n_reads"], "not scored: no read in its first minute"] + [""] * 8)
            continue
        per = {s: S.worst(p, s) for s in S.NINE}
        best = max(per, key=per.get)
        t = S.worst(p, "tracker")

        def cw(sig, part):
            v = [x["own_latency"]["signals"][sig][part] for x in S.later(classwise[n])]
            v = [a for a in v if S.finite(a)]
            return min(v) if v else float("nan")

        def fc(pred):
            v = [x[pred]["auroc_slow"] for x in S.later(direct[n]) if pred in x and S.finite(x[pred]["auroc_slow"])]
            return min(v) if v else float("nan")

        adm_best = max(cw(s, "called_fast") for s in S.NINE)
        rows.append([n, sum(w["n"] for w in S.later(p)), f"{100 * S.later_error(p, 'error_rate_G2_request')[0]:.0f}%",
                     f(t, 3), f"{per[best]:.3f} ({NAMES[best]})", f"{t - per[best]:+.3f}",
                     f(cw("tracker", "called_fast"), 3), f(adm_best, 3),
                     f(S.worst(p, "tracker", S.PERIOD), 3), f(S.best_of(p, S.NINE, S.PERIOD), 3),
                     f"{f(fc('table'))} / {f(fc('heimdall'))} / {f(fc('net'))} / {f(fc('net_live'))}"])
    return rows


HEAD = ["trace", "later reads", "Heimdall wrong", "table", "best of nine", "lead", "table, admitted reads",
        "best of nine, admitted reads", "table, period labels", "best of nine, period labels",
        "forecast of slow reads: table / Heimdall / retrained / retrained with live state"]


def detection_pages():
    out = []
    for title, note, det, group, keep in (
            ("Held-out traces", "The 57 scored traces, the two cut from a volume the table's design had seen "
             "(msr.hm_0, left out of every summary) and the one that cannot be scored.",
             S.RES / "heldout" / "detection", "heldout", lambda n: True),
            ("The artefact's twelve device traces", "Where the table was shaped.",
             S.RES / "development" / "detection", "development", lambda n: n[0] == "P"),
            ("Five further development replays", "Raw traces of the artefact replayed on a Samsung 990 EVO Plus, "
             "also used while the table was shaped.", S.RES / "development" / "detection", "development",
             lambda n: n[0] != "P")):
        pub = {p.stem: S.load(p) for p in sorted(det.glob("*.json")) if keep(p.stem)}
        cw = {p.stem: S.load(p) for p in sorted((S.RES / "classwise" / group).glob("*.json"))}
        dr = {p.stem: S.load(p) for p in sorted((S.RES / "direct" / group).glob("*.json"))}
        out += [f"## {title}", "", note, "", table(HEAD, trace_rows(pub, cw, dr)), ""]
    return out


def spread(v, n=2):
    """Median (lowest to highest) of a list of per-round values."""
    if not v:
        return ""
    if len(v) == 1:
        return f"{v[0]:.{n}f}"
    return f"{st.median(v):.{n}f} ({min(v):.{n}f} to {max(v):.{n}f})"


def enterprise_pages():
    e = S.enterprise_nvme()
    raw = S.load(S.RES / "routing" / "enterprise_nvme.json")
    out = ["## Two identical enterprise NVMe drives of one machine", "",
           f"Two {raw['drives'][0]}, {raw['layout']}. Nine pairs and sixteen policies, in rounds: each round runs every "
           "policy once on every pair, the policies of a pair in a fresh random order. A run is kept when, in every "
           "second of it, the two drives saw no I/O but the replay's own and nobody else's processes ran beyond the "
           "quiet criterion (\"quiet\"), or when it failed the criterion only on other users' jobs, which ran on "
           "other cores (\"shared\"). A run that is neither is rejected and run again, up to four attempts. A round "
           "counts for a pair when every policy of the pair has a kept run in it or failed four attempts.", "",
           f"{e['kept_runs']} kept runs in the rounds counted: {e['kept_runs'] - e['shared_runs']} quiet, "
           f"{e['shared_runs']} shared and {e['missing_runs']} missing after four attempts. {e['rejected_attempts']} "
           "attempts were rejected and run again."
           + "".join(f" Missing: {m['pair']}, round {m['round']}, "
                     f"{'no admission control' if m['policy'] == 'baseline' else dict(EVERY)[m['policy']]}."
                     for m in e["missing"])
           + (" A round whose run with no admission control is missing gives its pair no ratio to that run, and still "
              "gives the comparisons between policies." if any(m["policy"] == "baseline" for m in e["missing"]) else ""),
           ""]
    rows = []
    for pair, v in e["pairs"].items():
        runs = e["runs"][pair]
        rows.append([pair, f"{raw['pairs'][pair]['trace_s']:.0f}", f(v["replay_over_trace_median"]),
                     ", ".join(f"{n:,}" for n in v["reads"]),
                     "" if v["reads_above_p999"] is None else
                     (f"{v['reads_above_p999'][0]}" if v["reads_above_p999"][0] == v["reads_above_p999"][1]
                      else f"{v['reads_above_p999'][0]} to {v['reads_above_p999'][1]}"),
                     len(runs), len(e["quiet_only"][pair]["rounds"])]
                    + [" / ".join(f"{c['quiet']}, {c['shared']}, {c['rejected_attempts']}"
                                 + (f", {c['missing']} missing" if c["missing"] else "") for c in runs.values())])
    out += ["**Runs.** Replay over trace: the replay's length over the trace's (median of the kept runs). Above "
            "one, the busier client's drive could not keep pace. Rounds: counted, and of them those in which every "
            "run was quiet. Per round counted: quiet, shared, rejected and run again.", "",
            table(["pair", "trace, s", "replay over trace", "scored reads", "reads above p99.9", "rounds",
                   "rounds all quiet", "runs per round"], rows), ""]
    rows = []
    for pair, v in e["pairs"].items():
        q = e["quiet_only"][pair]
        rows.append([pair, spread(v["distance_over_rule_same_timeout_rounds"]),
                     spread(q["distance_over_rule_same_timeout_rounds"]),
                     spread(v["oracle_over_best_gate_rounds"]), spread(q["oracle_over_best_gate_rounds"]),
                     spread(v["distance_cut_of_heimdall_p999_rounds"]),
                     spread(q["distance_cut_of_heimdall_p999_rounds"])])
    out += ["**Comparisons**, each taken within a round, then median (lowest to highest) over rounds: the better "
            "distance with hedging over the table with hedging, the oracle (the lowest p99.9 of its five "
            "actions) over the best of the three with hedging, and the distances' cut of Heimdall's p99.9 "
            "without hedging (1 minus the better distance's p99.9 over Heimdall's).", "",
            table(["pair", "distance over table, all kept", "quiet only", "oracle over best of three, all kept",
                   "quiet only", "distances' cut of Heimdall, all kept", "quiet only"], rows), ""]
    out += ["**Every policy.** p99.9 over that of no admission control in the same round, median (lowest to "
            "highest) over the rounds counted, all kept and quiet only, then mean latency over that of no admission "
            "control and the share of reads sent or copied to the other drive, both medians over the rounds counted "
            "(all kept).", ""]
    for pair, v in e["pairs"].items():
        q = e["quiet_only"][pair]
        rows = [[label, spread(v["p999_over_baseline_rounds"][key]), spread(q["p999_over_baseline_rounds"][key]),
                 f(v["mean_over_baseline_median"][key]), f(v["moved_median"][key])] for key, label in EVERY]
        base = "" if v["baseline_p999_ms_median"] is None else f"{v['baseline_p999_ms_median']:.2f} ms"
        out += [f"**{pair}** (no admission control: p99.9 {base})", "",
                table(["policy", "p99.9, all kept", "quiet only", "mean", "moved"], rows), ""]
    return out


def floor_page():
    """The floor of hedging beside what hedging alone reaches, every pair of both testbeds with rounds."""
    e, q = S.enterprise_nvme()["pairs"], S.quiet_drives()["pairs"]
    rows, below = [], []
    for testbed, pairs in (("identical", e), ("unlike", q)):
        for pair, v in pairs.items():
            fl = v["timeout_floor"]
            if fl is None:
                continue
            t = max(set(fl["tail_client_rounds"]), key=fl["tail_client_rounds"].count)
            alone = v["p999_over_baseline_rounds"]["timeout-hedging"]
            a, f0 = st.median(alone), fl["floor_median"]
            if a < f0:
                below.append(f"{pair} of the {testbed} drives")
            rows.append([testbed, pair, f"{fl['H_us'][0]:.0f}, {fl['H_us'][1]:.0f}",
                         f"{fl['drive_p50_us_median'][0]:.0f} / {fl['drive_p50_us_median'][1]:.0f}",
                         f"{fl['drive_p999_us_median'][0]:.0f} / {fl['drive_p999_us_median'][1]:.0f}",
                         f"{1000 * v['baseline_p999_ms_median']:.0f}",
                         f"client {t} ({100 * st.median(fl['tail_share_rounds']):.0f}%)", f(f0),
                         spread(fl["capped_rounds"]), spread(alone), f(a / f0),
                         f"{100 * (1 - a) / (1 - f0):.0f}%" if a >= f0 and f0 < 1 else "below the floor"])
    holds = "every pair" if not below else "every pair but " + ", ".join(below[:-1]) + (" and " if len(below) > 1 else "") + below[-1]
    out = ["## The floor of hedging", "",
           "With hedging alone a read still unanswered after its client's timeout H is copied to the other drive, so it "
           "returns no sooner than H plus one read there. Per round, the tail's client is the one with more of its reads above the "
           "pooled p99.9 of the run with no admission control, and the floor is its H plus the other drive's median read "
           "latency, over the pooled p99.9 (median over rounds). Capped: the pooled p99.9 of that run with every read "
           "capped at its client's H plus the other drive's median, over the pooled p99.9 (median, lowest to highest over "
           "rounds). Room taken: how much of the cut the floor allows hedging alone takes, (1 minus hedging alone) "
           "over (1 minus the floor). Microseconds, medians over rounds.", "",
           f"Hedging alone ends at or above the floor on {holds}. "
           + ("" if not below else "There the tail is split between the two clients and rests on a few reads, so a copy "
              "answered sooner than the other drive's median can beat the floor, and the floor is no bound."), "",
           table(["testbed", "pair", "H, client 0, 1", "median, drive 0 / 1", "p99.9, drive 0 / 1", "pooled p99.9",
                  "tail's client (share of the reads above)", "floor", "capped", "hedging alone",
                  "hedging alone over floor", "room taken"], rows), ""]
    return out


def routing_pages():
    out = ["## Heimdall's artefact: its own replay logs", "",
           "Read latency over that of no admission control, later reads of each pair, from the logs the artefact "
           "ships (two NVMe drives).", ""]
    art = S.artefact_replays()["pairs"]
    out += [table(["pair", "reads", "Heimdall p99.9", "hedging p99.9", "random p99.9", "Heimdall mean",
                   "hedging mean", "random mean"],
                  [[p, v["reads"]] + [f(v[k]["p99.9"]) for k in ("heimdall", "hedging", "random")]
                   + [f(v[k]["mean"]) for k in ("heimdall", "hedging", "random")] for p, v in art.items()]), ""]
    out += enterprise_pages()
    q = S.quiet_drives()["pairs"]
    out += ["## Two unlike drives of a quiet machine, five rounds", "",
            "p99.9 over that of no admission control in the same round: median (lowest to highest) over the five "
            "rounds, then the mean latency over that of no admission control, and the share of reads sent or copied "
            "to the other drive (medians of the five rounds). S1 and S3 hold 9,332 and 3,432 scored reads, so "
            "their p99.9 rests on a handful of reads.", ""]
    for pair, v in q.items():
        rows = []
        for key, label in POLICIES:
            r = v["p999_over_baseline_rounds"][key]
            rows.append([label, f"{st.median(r):.2f} ({min(r):.2f} to {max(r):.2f})",
                         f(v["mean_over_baseline_median"][key]), f(v["moved_median"][key])])
        out += [f"**{pair}** (no admission control: p99.9 {v['baseline_p999_ms_median']:.1f} ms)", "",
                table(["policy", "p99.9", "mean", "moved"], rows), ""]
    nv = S.identical_nvme()["pairs"]
    out += ["## Two consumer NVMe drives of a busy machine", "",
            "One run a pair, two for K, S1 and S3: p99.9 over that of no admission control, with the share of reads "
            "sent or copied to the other drive in brackets. P4 was cut short. No record of what else the machine "
            "was doing was kept.", ""]
    keys = [k for k, _ in POLICIES if k != "heimdall+timeout"]
    rows = []
    for pair, runs in nv.items():
        for run, v in runs.items():
            rows.append([pair, run, v["reads"], f"{v['baseline_p999_ms']:.1f} ms"]
                        + [f"{v['p999_over_baseline'][k]:.2f} ({v['moved'][k]:.2f})" if k in v["p999_over_baseline"]
                           else "" for k in keys])
    out += [table(["pair", "run", "reads", "no admission control"] + [dict(POLICIES)[k] for k in keys], rows), ""]
    out += floor_page()
    sim = S.simulated()["pairs"]
    out += ["## The simulator", "",
            "p99.9 over that of no admission control. The simulator reproduces the artefact's replays of no "
            "admission control, random routing and hedging to within 15% at p50 and p99.9, but Heimdall's own "
            "replay less well (its p99.9 is 76% and 89% too high on P4 and P6), so nothing in the paper rests on "
            "it alone. Table alone: a read goes to both drives exactly when the table's entry is at least 1/2.", ""]
    keys = [k for k, _ in POLICIES] + ["tracker-hedgeslow"]
    out += [table(["pair"] + [dict(POLICIES + [("tracker-hedgeslow", "table alone")])[k] for k in keys],
                  [[p] + [f(v[k]["p99.9"]) for k in keys] for p, v in sim.items()]), ""]
    return out


def other_page():
    od = S.other_decisions()
    out = ["## Two other decisions", "",
           "The same protocol on two other learned decisions, each with a controller of Heimdall's shape that we "
           "trained on each request's own outcome: hedging calls to four microservices (Alibaba's 2022 traces) and "
           "hedging reads on a storage server (Google's Thesios traces). There the table has no edge over the network's "
           "own margin.", ""]
    rows = []
    for task, label in (("rpc_hedging", "RPC hedging"), ("storage_server_hedging", "storage-server hedging")):
        for n, v in od[task]["by_trace"].items():
            rows.append([label, n, f(v["table"])] + [f(v[k]) for k in S.NINE])
    return out + [table(["decision", "trace", "table"] + [NAMES[k] for k in S.NINE], rows), ""]


def cost_page():
    c = S.cost()
    return ["## Cost of one decision", "",
            f"One core of an {c['cpu'].replace('13th Gen Intel(R) Core(TM) ', 'Intel Core ')}, "
            f"{c['decisions']:,} decisions, each timed alone.", "",
            table(["what is timed", "ns"], [["the table", f"{c['table_ns']:.1f}"],
                                     ["the table inside the decision path, with its bookkeeping",
                                      f"{c['table_in_decision_path_ns']:.1f}"],
                                     ["Euclidean distance", f"{c['euclidean_ns']:.1f}"],
                                     ["Mahalanobis distance", f"{c['mahalanobis_ns']:.1f}"],
                                     ["conformal, 100 calibration reads", f"{c['conformal_100_ns']:,.0f}"],
                                     ["conformal, 1,000 calibration reads", f"{c['conformal_1000_ns']:,.0f}"],
                                     ["Heimdall's inference", f"{c['heimdall_inference_ns']:,.0f}"]]), ""]


def page():
    out = ["# Every measurement", "",
           "Generated by [scripts/tables.py](../scripts/tables.py) from the files of this folder. A trace's score "
           "for an estimator or the table is its AUROC in the worst of the four later windows. A guess is wrong when the "
           "read's own latency class disagrees with it, unless period labels are named. The best of nine is picked per trace "
           "after the fact. The last column is the AUROC of four forecasts that a read will be slow "
           "([METHOD.md](../METHOD.md#the-table-as-a-forecast)).", ""]
    return "\n".join(out + detection_pages() + routing_pages() + other_page() + cost_page())


if __name__ == "__main__":
    text = page()
    if sys.argv[1:] == ["--check"]:
        ok = OUT.exists() and OUT.read_text() == text
        print("results/tables.md", "is current" if ok else "DIFFERS from results/")
        sys.exit(0 if ok else 1)
    OUT.write_text(text)
    print(f"wrote results/tables.md: {len(text.splitlines())} lines")
