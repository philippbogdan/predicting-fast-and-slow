#!/usr/bin/env python3
"""One command: is every quoted number what the result files say?

  1. results/SHA256SUMS      No result file has changed since the numbers were computed.
  2. scripts/summary.py      Recomputes every quoted number from the per-window files in results/
                             and must reproduce results/summary.json.
  3. scripts/typeset.py      paper/numbers.tex, README.md and METHOD.md are that summary, formatted, and the
                             pages carry the numbers of METHOD.md's plates as scripts/plate_data.py
                             recomputes them from results/.
     scripts/tables.py       results/tables.md is the tables of results/.
  4. results/heldout/frozen.sha256
                             admission/ is byte for byte the pipeline that was frozen before the
                             held-out traces were replayed.
  5. The statements the paper, the pages and the plates' captions make in words (all six pairs, all twelve
     traces, level with, ...) hold in the recomputed numbers.

Prints what it found and exits 0 only if all five hold.  Standard library only.
`check.py --seal` rewrites results/SHA256SUMS, results/summary.json, paper/numbers.tex, README.md, METHOD.md
and results/tables.md after results/ has been regenerated (METHOD.md, "Run it").
"""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMS = ROOT / "results" / "SHA256SUMS"
FROZEN = ROOT / "results" / "heldout" / "frozen.sha256"
TABLES = ROOT / "results" / "tables.md"


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_files():
    """Every file of results/ but what is computed from the others (the summary, the checksums) and the pages."""
    skip = {SUMS, ROOT / "results" / "summary.json"}
    return sorted(p for p in (ROOT / "results").rglob("*")
                  if p.is_file() and p not in skip and p.suffix != ".md" and p.name != ".DS_Store")


def sums_text():
    return "".join(f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in result_files())


def differences(a, b, path=""):
    """Where two JSON values differ (floats to one part in a thousand million)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in one" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += differences(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items against {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        try:
            same = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
        except TypeError:
            same = False
        return [] if same else [f"{path}: {a} against {b}"]
    return [] if a == b else [f"{path}: {a!r} against {b!r}"]


def statements(s):
    """The claims the paper, README.md and METHOD.md make in words, each as a test on the recomputed summary."""
    h, dev, cw = s["heldout"], s["development"]["device_traces"], s["classwise"]
    pooled, art, q = cw["heldout"]["pooled"], s["artefact_replays"], s["quiet_drives"]
    en = s["enterprise_nvme"]
    ea, ep = en["artefact"], en["pairs"]
    six = [ep[p] for p in ("P1", "P2", "P3", "P4", "P5", "P6")]
    d, dd, od = s["direct"]["heldout"], s["direct"]["development_device_traces"], s["other_decisions"]
    study = s["study_reported"]
    fig10, acc = study["read_from_figure_10"], s["development"]["artefact_accuracy"]
    nine = h["estimators"]
    return [
        ("none of the nine estimators finds the wrong guesses: every median is within 0.05 of a coin, and a guess "
         "Euclidean distance flags is wrong less than 10 points more often than one it passes, on the median trace",
         all(0.45 < v["median"] < 0.55 for v in nine.values())
         and 0 <= cw["heldout"]["gates"]["euclid"]["flagged_minus_passed_median"] < 0.10),
        ("one running count outranks all nine estimators",
         all(pooled["signals"]["one_cell"]["median"] > v["median"] for v in nine.values())
         and pooled["one_cell_ahead_of_each_range"][0] > h["traces"] / 2),
        ("deployed, Heimdall's accuracy by its own period labels is below the lowest the artefact reports, on every "
         "one of the twelve traces",
         1 - dev["controller_wrong"]["period_labels_range"][0] < acc["reported_range"][0]
         and acc["labels_and_features_identical"]),
        ("the table's level is modest: fewer than half of the held-out traces reach 0.80",
         h["table_at_least_080"] < h["traces"] / 2),
        ("the table is ahead of each estimator on most traces with the guess fixed, level with the hindsight best",
         cw["heldout"]["called_fast"]["table_ahead_of_each_range"][0] > h["traces"] / 2
         and abs(cw["heldout"]["called_fast"]["lead_median"]) < 0.02),
        ("under period labels the table is a coin and the best of the nine is above 0.6, and the two distances fail "
         "under both truths",
         max(nine[k]["median"] for k in ("euclid", "mahal")) < 0.52
         and max(nine[k]["median_period_labels"] for k in ("euclid", "mahal")) < 0.52
         and h["estimator_medians_range_period_labels"][1] > 0.6 > 0.55 > h["period_labels"]["table_median"] > 0.45),
        ("on two other decisions the table has no edge: behind the margin on every RPC trace, ahead of it by less "
         "than 0.06 on every storage trace, and distance trails the margin",
         all(v["table"] < v["margin"] for v in od["rpc_hedging"]["by_trace"].values())
         and all(0 < v["table"] - v["margin"] < 0.06 for v in od["storage_server_hedging"]["by_trace"].values())
         and sum(max(v["euclid"], v["mahal"]) < v["margin"] for o in (od["rpc_hedging"], od["storage_server_hedging"])
                 for v in o["by_trace"].values()) >= 6),
        ("the table predicts slow reads better than the network it checks, on most held-out traces and all twelve of "
         "the artefact's, where the median of Heimdall's output is below a coin",
         d["every_read"]["heimdall"]["table_ahead"] > h["traces"] / 2
         and dd["every_read"]["heimdall"]["table_ahead"] == 12 and dd["every_read"]["heimdall"]["median"] < 0.5),
        ("a retrained network draws level with the table, and with the drive's live state it pulls ahead",
         abs(d["every_read"]["net"]["table_lead_median"]) < 0.02
         and d["every_read"]["net_live"]["table_ahead"] < h["traces"] / 2),
        ("held out, calling slow on the table earns next to nothing over never calling slow, and Heimdall is wrong "
         "more often than never calling slow on most traces",
         abs(d["wrong_calls_median"]["table"] - d["wrong_calls_median"]["never_slow"]) < 0.02
         and d["wrong_more_often_than_never_slow"]["heimdall"] > h["traces"] / 2),
        ("on the identical drives every pair's rounds are counted from the first on with none skipped, each holding a "
         "record of all sixteen policies, a kept run or one missing after four attempts (their number is IdRounds, "
         "and IdRoundsSix on the six artefact pairs)",
         all(list(r) == [str(i) for i in range(1, len(r) + 1)] for r in en["runs"].values())
         and all(sum(c.values()) - c["rejected_attempts"] == 16 for r in en["runs"].values() for c in r.values())),
        ("every policy the paper compares on the six artefact pairs (the distances and the table with hedging, the "
         "oracle's five actions, Heimdall and the distances without hedging) has a kept run in every one of those "
         "rounds",
         en["compared_policies_kept_rounds_range"][0] == en["compared_policies_kept_rounds_range"][1]
         == min(en["rounds_counted"][p] for p in ("P1", "P2", "P3", "P4", "P5", "P6"))
         == max(en["rounds_counted"][p] for p in ("P1", "P2", "P3", "P4", "P5", "P6"))),
        ("one run is missing on the identical drives: P1's run with no admission control in its fifth round",
         en["missing"] == [{"pair": "P1", "round": "5", "policy": "baseline"}]),
        ("whoever picks the reads, the tail lands in the same place: with hedging, the better distance ends within 15% "
         "of the table on every artefact pair",
         1 / 1.15 <= ea["distance_over_rule_same_timeout_range"][0] <= ea["distance_over_rule_same_timeout_range"][1]
         <= 1.15),
        ("the reported cut is not reproduced: on the identical drives the distances cut Heimdall's p99.9 by less than "
         "half of what Gohil et al. report, on every artefact pair",
         ea["distance_cut_of_heimdall_p999_range"][1]
         < min(study["transcribed"]["p999_cut_over_heimdall_percent"].values()) / 200),
        ("in Heimdall's own logs hedging has the lower p99.9 on all six pairs and Heimdall the lower mean on all six",
         art["hedging_lower_p999_pairs"] == 6 and art["heimdall_lower_mean_pairs"] == 6),
        # the measurements the paper leaves out (METHOD.md)
        ("three of the 60 cuts are left out: two the table's design had seen, one that cannot be scored",
         h["selected"] == 60 and len(h["development_exposed"]) == 2 and len(h["unscored"]) == 1 and h["traces"] == 57),
        ("on the artefact's traces the table leads the best of nine on all twelve, both ways",
         dev["table_ahead"] == 12 and cw["development_device_traces"]["called_fast"]["table_ahead"] == 12
         and cw["development_device_traces"]["called_slow"]["table_ahead"] == 12),
        ("all the traces the table loses are UMass Financial volumes",
         all("Financial" in b["trace"] for b in h["behind"]) and len(h["behind"]) == h["table_behind"]),
        ("on the identical drives routing on the table alone has the lower p99.9 on all six artefact pairs, "
         "Heimdall the lower on the three short pairs K, S1 and S3",
         all(a["p999_over_baseline_median"]["tracker-hedgeslow"] < a["p999_over_baseline_median"]["heimdall"]
             for a in six)
         and all(ep[x]["p999_over_baseline_median"]["heimdall"] < ep[x]["p999_over_baseline_median"]["tracker-hedgeslow"]
                 for x in ("K", "S1", "S3"))),
        ("hedging alone ends at or above its floor (the timeout of the tail's client plus the other drive's median, "
         "over the p99.9) on P1 to P6 of the identical drives and P2, P5 and P6 of the unlike pair",
         all(a["p999_over_baseline_median"]["timeout-hedging"] >= a["timeout_floor"]["floor_median"] for a in six)
         and all(q["pairs"][x]["p999_over_baseline_median"]["timeout-hedging"]
                 >= q["pairs"][x]["timeout_floor"]["floor_median"] for x in ("P2", "P5", "P6"))),
        ("on the unlike pair the busier client's drive falls behind the traces of P2 and P5",
         min(q["pairs"][x]["replay_over_trace_median"] for x in ("P2", "P5")) > 1.2),
        ("on the unlike pair every policy with hedging lands in one place, and without it nothing moves P2 and P5",
         q["with_timeout_p999_range"][1] < 0.3 and q["gates_without_timeout_p999_range_P2_P5"][0] > 0.9),
        ("in Figure 10 of Gohil et al., as we read it, hedging and the estimators meet at p99.99",
         abs(fig10["p9999_ms"]["hedging"] - fig10["p9999_ms"]["heimdall_plus_euclidean"]) <= 0.1),
    ] + retraining_statements(s)


def retraining_statements(s):
    """What METHOD.md's paragraph on retraining says in words, each as a test on the recomputed summary."""
    h, cf, r = s["heldout"], s["classwise"]["heldout"]["called_fast"], s["retraining"]
    a = r["arms"]
    s7, s7s = a["heimdall_s7"], a["heimdall_s7"]["schedule"]
    every10 = [float(t) for t in range(70, 180, 10)]
    spec = {arm: x["schedule"]["specs"] for arm, x in a.items()}
    return [
        ("retraining: every schedule holds the held-out traces, each the replay results/heldout/traces.json "
         "records, and scores the paper's reads in every window, with the same reads slow",
         r["traces"] == r["traces_in_every_arm"] == r["replays_as_recorded"] == r["windows_as_scored"] == h["traces"]
         and all(x["traces"] == h["traces"] for x in a.values())),
        ("retraining: the four schedules are Heimdall's check at 120 s, a refit at 120 s, and a refit every 10 s "
         "from 70 s on the last minute or on everything since the start, each refit fitting the network and the "
         "nine estimators afresh",
         sorted(a) == ["grow10", "heimdall_s7", "minute", "slide10"]
         and all(len(v) == 1 and v[0]["refit"] == "all" and v[0]["first"] == "fresh" for v in spec.values())
         and spec["heimdall_s7"][0]["times"] == spec["minute"][0]["times"] == [120.0]
         and spec["heimdall_s7"][0]["trigger"] and not spec["minute"][0]["trigger"]
         and s7s["stretches_s"] == a["minute"]["schedule"]["stretches_s"] == [[60.0, 120.0]]
         and spec["slide10"][0]["times"] == spec["grow10"][0]["times"] == every10
         and a["slide10"]["schedule"]["stretches_s"] == [[t - 60, t] for t in every10]
         and a["grow10"]["schedule"]["stretches_s"] == [[0.0, t] for t in every10]
         and a["minute"]["schedule"]["traces_refitted"] == h["traces"]),
        ("retraining: Heimdall's check is run once on every trace and refits exactly where the accuracy over the "
         "last minute is below 80%",
         s7s["checked"] == s7s["scheduled"] == h["traces"] and s7s["refitted_when_checked"] == s7s["refits"]
         == s7s["traces_refitted"]
         and s7s["accuracy_refitted_max"] < r["heimdall_trigger_below"] <= s7s["accuracy_kept_min"]),
        ("retraining: where Heimdall's check fires nothing moves: the table's median and its lead over the best "
         "of the nine, over every read and among reads called fast, are the paper's frozen ones at two decimals",
         0 < s7s["traces_refitted"] < h["traces"]
         and f"{s7['pooled']['table_median']:.2f}" == f"{h['table_median']:.2f}"
         and s7["pooled"]["table_ahead"] == h["table_ahead"]
         and f"{s7['called_fast']['table_median']:.2f}" == f"{cf['table_median']:.2f}"
         and f"{s7['called_fast']['best_of_nine_median']:.2f}" == f"{cf['best_of_nine_median']:.2f}"),
        ("retraining: on every schedule the table is ahead of the best of the nine on most traces, and Heimdall "
         "calls slow more often than reads come back slow on most traces",
         r["pooled_table_ahead_range"][0] > h["traces"] / 2 and r["calls_minus_reads_median_range"][0] > 0
         and all(x["calls_slow_more_often_than_reads_are"] > h["traces"] / 2 for x in a.values())),
        ("retraining: among reads called fast the tie stays a tie, the table's lead over the best of the nine "
         "within 0.02 of none on the median trace on every schedule, as frozen",
         max(abs(v) for v in r["called_fast_lead_median_range"]) < 0.02 and abs(cf["lead_median"]) < 0.02),
    ]


def seal():
    summary, typeset = module("summary"), module("typeset")
    s = summary.build()
    (ROOT / "results" / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    m = typeset.macros(json.loads(json.dumps(s)))
    typeset.TEX.write_text(typeset.tex(m))
    pm = typeset.page_macros(json.loads(json.dumps(s)))
    for path in (typeset.README, typeset.METHOD):
        path.write_text(typeset.page(path, pm))
    TABLES.write_text(module("tables").page())
    SUMS.write_text(sums_text())
    print(f"sealed: {len(result_files())} result files, {len(m)} quoted numbers")


def check():
    ok = True

    def report(passed, what, detail=()):
        nonlocal ok
        ok &= bool(passed)
        print(("ok    " if passed else "FAIL  ") + what)
        for d in list(detail)[:8]:
            print("        " + d)

    # 1. the result files are the ones the numbers were computed from
    want = dict(line.rstrip("\n").split("  ", 1)[::-1] for line in SUMS.read_text().splitlines())
    have = {p.relative_to(ROOT).as_posix(): sha256(p) for p in result_files()}
    changed = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    report(not changed, f"{len(have)} result files match results/SHA256SUMS", changed)

    # 2. every quoted number, recomputed from them
    summary = module("summary")
    stored = json.loads((ROOT / "results" / "summary.json").read_text())
    fresh = json.loads(json.dumps(summary.build()))
    diff = differences(stored, fresh)
    report(not diff, "results/summary.json is what scripts/summary.py computes from results/", diff)

    # 3. the paper's numbers and the two pages are that summary, formatted
    typeset = module("typeset")
    m, pm = typeset.macros(fresh), typeset.page_macros(fresh)
    for path, text, n in ((typeset.TEX, typeset.tex(m), len(m)),
                          (typeset.README, typeset.page(typeset.README, pm), len(pm)),
                          (typeset.METHOD, typeset.page(typeset.METHOD, pm), len(pm))):
        report(path.exists() and path.read_text() == text,
               f"{path.relative_to(ROOT)} carries the {n} quoted numbers as computed")

    tables = module("tables")
    report(TABLES.exists() and TABLES.read_text() == tables.page(), "results/tables.md is the tables of results/")

    # 4. the pipeline is the frozen one
    bad = [line.split("  ", 1)[1] for line in FROZEN.read_text().splitlines()
           if sha256(ROOT / line.split("  ", 1)[1]) != line.split("  ", 1)[0]]
    report(not bad, "admission/ is the pipeline frozen before the held-out replays", bad)

    # 5. what the paper and the pages say in words rather than numbers
    for said, holds in statements(fresh) + module("plate_data").statements(fresh):
        report(holds, "in words: " + said)

    h, c, d = fresh["heldout"], fresh["classwise"]["heldout"], fresh["direct"]["heldout"]["every_read"]
    g = c["gates"]["euclid"]
    print()
    print(f"held-out traces scored: {h['traces']} of {h['selected']} selected")
    print(f"the nine estimators, median AUROC for Heimdall's wrong guesses: "
          f"{h['estimator_medians_range'][0]:.2f} to {h['estimator_medians_range'][1]:.2f}")
    print(f"on the median trace a guess Euclidean distance flags is wrong "
          f"{100 * g['flagged_minus_passed_median']:.0f} points more often than one it passes")
    print(f"the table: {h['table_median']:.2f}, ahead of the best of nine on {h['table_ahead']} of {h['traces']}")
    print(f"with the guess fixed (reads admitted as fast): the table {c['called_fast']['table_median']:.2f}, "
          f"ahead of the best of nine on {c['called_fast']['table_ahead']} of {c['called_fast']['traces']}")
    print(f"the table's entry against Heimdall's output as a forecast of slow reads: ahead on "
          f"{d['heimdall']['table_ahead']} of {d['heimdall']['traces']}")
    r = fresh["retraining"]
    print(f"Heimdall retrained during the replay, four schedules: the table {r['pooled_table_median_range'][0]:.2f} "
          f"to {r['pooled_table_median_range'][1]:.2f}, ahead of the best of nine on "
          f"{r['pooled_table_ahead_range'][0]} to {r['pooled_table_ahead_range'][1]} of {r['traces']}")
    print()
    en = fresh["enterprise_nvme"]
    print(f"identical drives: {en['kept_runs']} kept runs ({en['shared_runs']} shared, {en['missing_runs']} missing "
          f"after four attempts, {en['rejected_attempts']} attempts rejected and run again). Whole rounds counted: "
          + ", ".join(f"{p} {n}" for p, n in en["rounds_counted"].items()))
    print()
    print("every check passed" if ok else "SOMETHING DOES NOT MATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--seal"]:
        seal()
    else:
        sys.exit(check())
