#!/usr/bin/env python3
"""The numbers the paper, the README and docs/further.md quote, set into all three from results/summary.json.

Every figure quoted in paper/main.tex is a macro of paper/numbers.tex, and README.md and docs/further.md are
docs/README.in.md and docs/further.in.md with the same names filled in, so no text can drift from the result
files: scripts/summary.py recomputes summary.json from results/, this script formats it, and check.py verifies
them all.
Usage: typeset.py [--check]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX, README, TEMPLATE = ROOT / "paper" / "numbers.tex", ROOT / "README.md", ROOT / "docs" / "README.in.md"
FURTHER, FURTHER_TEMPLATE = ROOT / "docs" / "further.md", ROOT / "docs" / "further.in.md"
WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.0f}"


def tail(lo_hi):
    """How many reads lie above a p99.9: the same in every run of a pair, since every run scores the same reads."""
    lo, hi = lo_hi
    return f"{lo}" if lo == hi else f"{lo} to {hi}"


def whole_ms(lo_hi):
    """Milliseconds as one whole number when the range rounds to it at both ends, else "lo to hi"."""
    lo, hi = (round(x) for x in lo_hi)
    return f"{lo}" if lo == hi else f"{lo} to {hi}"


def rounds(counts):
    """Whole rounds a pair, as a word: "five" when every pair has the same, else "four or five" (or "three to five")."""
    lo, hi = min(counts), max(counts)
    return WORDS[lo] if lo == hi else f"{WORDS[lo]} {'or' if hi == lo + 1 else 'to'} {WORDS[hi]}"


def macros(s):
    h, d, c = s["heldout"], s["development"]["device_traces"], s["cost"]
    cw, cwp, cwd = (s["classwise"]["heldout"], s["classwise"]["heldout_period_labels"],
                    s["classwise"]["development_device_traces"])
    e = h["estimators"]
    adm, mov, same = cw["called_fast"], cw["called_slow"], cw["same_size_called_fast"]
    eg, mg = cw["gates"]["euclid"], cw["gates"]["mahal"]
    dr, drd = s["direct"]["heldout"], s["direct"]["development_device_traces"]
    art, q, sim, study = s["artefact_replays"], s["quiet_drives"], s["simulated"], s["study_reported"]
    en = s["enterprise_nvme"]
    ea, ek, es = en["artefact"], en["K"], en["short_samples"]
    od = s["other_decisions"]
    ex, heim = s["example_table"], s["heimdall_reported"]["transcribed"]
    calm, storm = ex["small_read_idle_drive"], ex["large_read_backlog_after_quiet"]
    acc = s["development"]["artefact_accuracy"]
    filt = s["development"]["rows_filtered_as_noise_range"]
    return {
        # held-out traces, every read
        "HeldTraces": h["traces"], "HeldVolumes": h["volumes"],
        "EstMedLo": f2(h["estimator_medians_range"][0]), "EstMedHi": f2(h["estimator_medians_range"][1]),
        "TableMed": f2(h["table_median"]), "TableAhead": h["table_ahead"], "TableBehind": h["table_behind"],
        "TableEighty": h["table_at_least_080"],
        "OneCellMed": f2(cw["pooled"]["signals"]["one_cell"]["median"]),
        "OneCellAheadLo": cw["pooled"]["one_cell_ahead_of_each_range"][0],
        "OneCellAheadHi": cw["pooled"]["one_cell_ahead_of_each_range"][1],
        "CallsSlow": pct(cw["heimdall_calls_slow_median"]), "ReadsSlow": pct(cw["reads_slow_median"]),
        "SlowCallShare": pct(cw["mistakes_that_are_slow_calls_median"]),
        "PeriodSlow": pct(cwp["reads_slow_median"]),
        # the two distance gates as they act
        "EuclidFlags": pct(eg["flagged_median"]), "MahalFlags": pct(mg["flagged_median"]),
        "EuclidFlaggedWrong": pct(eg["wrong_when_flagged_median"]),
        "EuclidPassedWrong": pct(eg["wrong_when_passed_median"]),
        "EuclidFlagGap": pct(eg["flagged_minus_passed_median"]), "MahalFlagGap": pct(mg["flagged_minus_passed_median"]),
        "EuclidFlaggedMore": eg["flagged_more_often_wrong"], "MahalFlaggedMore": mg["flagged_more_often_wrong"],
        "MahalFlaggedWrong": pct(mg["wrong_when_flagged_median"]),
        "MahalPassedWrong": pct(mg["wrong_when_passed_median"]),
        # with the guess fixed
        "AdmTable": f2(adm["table_median"]), "AdmEstLo": f2(adm["estimator_medians_range"][0]),
        "AdmEstHi": f2(adm["estimator_medians_range"][1]), "AdmAheadBest": adm["table_ahead"],
        "AdmAheadLo": adm["table_ahead_of_each_range"][0], "AdmAheadHi": adm["table_ahead_of_each_range"][1],
        "AdmSup": f2(adm["signals"]["supervised_w0_G2"]["median"]),
        "AdmEuclid": f2(adm["signals"]["euclid"]["median"]), "AdmMahal": f2(adm["signals"]["mahal"]["median"]),
        "MovTable": f2(mov["table_median"]), "MovEstLo": f2(mov["estimator_medians_range"][0]),
        "MovEstHi": f2(mov["estimator_medians_range"][1]), "MovAheadBest": mov["table_ahead"],
        "MovTraces": mov["traces"],
        "MovEuclid": f2(mov["signals"]["euclid"]["median"]), "MovMahal": f2(mov["signals"]["mahal"]["median"]),
        "DevAdmTable": f2(cwd["called_fast"]["table_median"]), "DevAdmBest": f2(cwd["called_fast"]["best_of_nine_median"]),
        "DevMovTable": f2(cwd["called_slow"]["table_median"]), "DevMovBest": f2(cwd["called_slow"]["best_of_nine_median"]),
        "DevAdmAhead": WORDS[cwd["called_fast"]["table_ahead"]],
        "DevAdmEuclid": f2(cwd["called_fast"]["signals"]["euclid"]["median"]),
        "DevMovEuclid": f2(cwd["called_slow"]["signals"]["euclid"]["median"]),
        "DevMovMahal": f2(cwd["called_slow"]["signals"]["mahal"]["median"]),
        "SameAdmTable": f2(same["table_median"]), "SameAdmMargin": f2(same["signals"]["margin"]["median"]),
        # Heimdall's period labels as the truth
        "PerBehind": h["period_labels"]["table_behind"], "PerTable": f2(h["period_labels"]["table_median"]),
        "PerMargin": f2(e["margin"]["median_period_labels"]),
        "PerEstHi": f2(h["estimator_medians_range_period_labels"][1]),
        "PerEuclid": f2(e["euclid"]["median_period_labels"]), "PerMahal": f2(e["mahal"]["median_period_labels"]),
        # how often the controller is wrong
        "DevWrongLo": pct(d["controller_wrong"]["own_latency_range"][0]),
        "DevWrongHi": pct(d["controller_wrong"]["own_latency_range"][1]),
        "HeldWrongLo": pct(h["controller_wrong"]["own_latency_range"][0]),
        "HeldWrongHi": pct(h["controller_wrong"]["own_latency_range"][1]),
        "DevWrongPerLo": pct(d["controller_wrong"]["period_labels_range"][0]),
        "DevWrongPerHi": pct(d["controller_wrong"]["period_labels_range"][1]),
        "HeldWrongPerLo": pct(h["controller_wrong"]["period_labels_range"][0]),
        "HeldWrongPerHi": pct(h["controller_wrong"]["period_labels_range"][1]),
        "ArtAccLo": pct(acc["reported_range"][0]), "ArtAccHi": pct(acc["reported_range"][1]),
        "ArtAccDiff": f"{100 * acc['reproduced_max_abs_difference']:.1f}",
        "FilteredLo": pct(filt[0]), "FilteredHi": pct(filt[1]),
        # the artefact's twelve device traces
        "DevTableLo": f2(d["table_range"][0]), "DevTableHi": f2(d["table_range"][1]),
        "DevBestLo": f2(d["best_of_nine_range"][0]), "DevBestHi": f2(d["best_of_nine_range"][1]),
        "DevWorstRaw": f2(min(s["development"]["raw_replays"]["table_by_trace"].values())),
        # cost
        "TableNs": f"{c['table_ns']:.0f}", "TableNsExact": f"{c['table_ns']:.1f}",
        "EuclidNs": f"{c['euclidean_ns']:.1f}", "EuclidNsWhole": f"{c['euclidean_ns']:.0f}",
        "MahalNs": f"{c['mahalanobis_ns']:.0f}",
        "ConfHundredUs": f"{c['conformal_100_ns'] / 1000:.1f}", "ConfThousandUs": f"{c['conformal_1000_ns'] / 1000:.0f}",
        "HeimdallUs": f"{c['heimdall_inference_ns'] / 1000:.1f}", "HeimdallNs": f"{c['heimdall_inference_ns']:,.0f}",
        "TableInsituNs": f"{c['table_in_decision_path_ns']:.1f}",
        # the table as a predictor of slow reads
        "DirHeimAhead": dr["every_read"]["heimdall"]["table_ahead"],
        "DirTableMed": f2(dr["every_read"]["table"]["median"]), "DirHeimMed": f2(dr["every_read"]["heimdall"]["median"]),
        "DirDevHeimMed": f2(drd["every_read"]["heimdall"]["median"]),
        "DirDevTableMed": f2(drd["every_read"]["table"]["median"]),
        "DirNetAhead": dr["every_read"]["net"]["table_ahead"],
        "DirNetLead": f"{dr['every_read']['net']['table_lead_median']:.3f}",
        "DirLiveBehind": dr["every_read"]["net_live"]["traces"] - dr["every_read"]["net_live"]["table_ahead"],
        "WrongTable": pct(dr["wrong_calls_median"]["table"]), "WrongHeim": pct(dr["wrong_calls_median"]["heimdall"]),
        "WrongNet": pct(dr["wrong_calls_median"]["net"]), "WrongLive": pct(dr["wrong_calls_median"]["net_live"]),
        "DirNetMed": f2(dr["every_read"]["net"]["median"]), "DirLiveMed": f2(dr["every_read"]["net_live"]["median"]),
        "DevWrongTable": pct(drd["wrong_calls_median"]["table"]), "DevWrongHeim": pct(drd["wrong_calls_median"]["heimdall"]),
        # the call that never says slow
        "NeverSlowWrong": pct(dr["wrong_calls_median"]["never_slow"]),
        "DevNeverSlowWrong": pct(drd["wrong_calls_median"]["never_slow"]),
        "HeimWorseThanNever": dr["wrong_more_often_than_never_slow"]["heimdall"],
        # routing on the table alone, on the identical drives
        "IdTableOnlyMovedLo": pct(ea["table_only_moved_range"][0]),
        "IdTableOnlyMovedHi": pct(ea["table_only_moved_range"][1]),
        "IdHeimMovedLo": pct(ea["heimdall_moved_range"][0]), "IdHeimMovedHi": pct(ea["heimdall_moved_range"][1]),
        "IdHeimMeanBelow": WORDS[ea["heimdall_lower_mean_than_table_only_pairs"]],
        # the tail
        "ArtRatioLo": f"{art['heimdall_over_hedging_p999_range'][0]:.1f}",
        "ArtRatioHi": f"{art['heimdall_over_hedging_p999_range'][1]:.1f}",
        "ArtHedgeLo": f2(art["hedging_p999_range"][0]), "ArtHedgeHi": f2(art["hedging_p999_range"][1]),
        "ArtHeimLo": f2(art["heimdall_p999_range"][0]), "ArtHeimHi": f2(art["heimdall_p999_range"][1]),
        "ArtHeimMeanLo": f2(art["heimdall_mean_range"][0]), "ArtHeimMeanHi": f2(art["heimdall_mean_range"][1]),
        "ArtHedgeMeanLo": f2(art["hedging_mean_range"][0]), "ArtHedgeMeanHi": f2(art["hedging_mean_range"][1]),
        "QuietNoLo": f2(q["gates_without_timeout_p999_range_P2_P5"][0]),
        "QuietNoHi": f2(q["gates_without_timeout_p999_range_P2_P5"][1]),
        "QuietWithLo": f2(q["with_timeout_p999_range"][0]), "QuietWithHi": f2(q["with_timeout_p999_range"][1]),
        "QuietAloneLo": f2(q["timeout_alone_p999_range"][0]), "QuietAloneHi": f2(q["timeout_alone_p999_range"][1]),
        # the timeout's floor: H of the client that makes the tail plus the other drive's median, over the p99.9
        "QuietTailMs": whole_ms(q["baseline_p999_ms_range"]),
        "FloorQuietLo": f2(q["timeout_floor_range"][0]), "FloorQuietHi": f2(q["timeout_floor_range"][1]),
        "IdTailLoMs": f"{ea['baseline_p999_ms_range'][0]:.1f}", "IdTailHiMs": f"{ea['baseline_p999_ms_range'][1]:.1f}",
        "FloorIdLo": f2(ea["timeout_floor_range"][0]), "FloorIdHi": f2(ea["timeout_floor_range"][1]),
        # two identical enterprise NVMe drives of one machine
        "IdRounds": rounds(en["rounds_counted"].values()),
        "IdRoundsSix": rounds([en["rounds_counted"][p] for p in ("P1", "P2", "P3", "P4", "P5", "P6")]),
        "IdKept": en["kept_runs"], "IdShared": en["shared_runs"],
        "IdDistOverRuleLo": f2(ea["distance_over_rule_same_timeout_range"][0]),
        "IdDistOverRuleHi": f2(ea["distance_over_rule_same_timeout_range"][1]),
        "IdOracleOverGateLo": f2(ea["oracle_over_best_gate_range"][0]),
        "IdOracleOverGateHi": f2(ea["oracle_over_best_gate_range"][1]),
        # the same two comparisons as one number each: how far the better distance ends from the table at most, on
        # any pair, and how much the oracle gains on the best of the three at most
        # (bounds, so rounded up)
        "IdDistWithin": math.ceil(100 * max(1 - ea["distance_over_rule_same_timeout_range"][0],
                                            ea["distance_over_rule_same_timeout_range"][1] - 1) - 1e-9),
        "IdOracleGain": math.ceil(100 * (1 - ea["oracle_over_best_gate_range"][0]) - 1e-9),
        "IdKEuclid": f2(ek["euclid+timeout"]), "IdKTable": f2(ek["tracker-mixed+timeout"]),
        "IdSoneTail": tail(es["reads_above_p999"]["S1"]), "IdSthreeTail": tail(es["reads_above_p999"]["S3"]),
        "IdSGateLo": f2(es["gates_timeout_p999_range"][0]), "IdSGateHi": f2(es["gates_timeout_p999_range"][1]),
        "IdSoneAlone": f2(es["timeout_alone_p999"]["S1"]), "IdSthreeAlone": f2(es["timeout_alone_p999"]["S3"]),
        # two other decisions
        "RpcTableLo": f2(od["rpc_hedging"]["table_range"][0]), "RpcTableHi": f2(od["rpc_hedging"]["table_range"][1]),
        "RpcMarginLo": f2(od["rpc_hedging"]["margin_range"][0]), "RpcMarginHi": f2(od["rpc_hedging"]["margin_range"][1]),
        "StoTableLo": f2(od["storage_server_hedging"]["table_range"][0]),
        "StoTableHi": f2(od["storage_server_hedging"]["table_range"][1]),
        "StoMarginLo": f2(od["storage_server_hedging"]["margin_range"][0]),
        "StoMarginHi": f2(od["storage_server_hedging"]["margin_range"][1]),
        "OtherDistLo": f2(od["distance_range"][0]), "OtherDistHi": f2(od["distance_range"][1]),
        # the table one trace taught (Figure 2) and the reference implementation
        "CalmQ": f2(calm["q"]), "CalmSlow": pct(calm["slow_share"]), "CalmCalledSlow": pct(calm["heimdall_called_slow_share"]),
        "StormQ": f2(storm["q"]), "StormSlow": pct(storm["slow_share"]),
        "StormCalledFast": pct(1 - storm["heimdall_called_slow_share"]),
        "TableLines": ex["reference_implementation_lines"],
        "AdmOneCell": f2(adm["signals"]["one_cell"]["median"]),
        # what the two papers report
        "StudyBnnUs": study["transcribed"]["bayesian_network_inference_us"],
        "StudyConfUs": study["transcribed"]["conformal_1000_samples_us"],
        "HeimAccLo": heim["section_7_accuracy_after_a_single_training_session_percent"][0],
        "HeimAccHi": heim["section_7_accuracy_after_a_single_training_session_percent"][1],
        # the study's testbed and ours
        "StudyCutLo": study["transcribed"]["p999_cut_over_heimdall_percent"]["euclidean"],
        "StudyCutHi": study["transcribed"]["p999_cut_over_heimdall_percent"]["mahalanobis"],
        "StudyFlagLo": f"{study['transcribed']['flagged_percent']['euclidean']:.2f}",
        "StudyFlagHi": f"{study['transcribed']['flagged_percent']['mahalanobis']:.2f}",
        "SimCutLo": pct(sim["distance_cut_of_heimdall_p999_range"][0]),
        "SimCutHi": pct(sim["distance_cut_of_heimdall_p999_range"][1]),
        "IdCutLo": pct(ea["distance_cut_of_heimdall_p999_range"][0]),
        "IdCutHi": pct(ea["distance_cut_of_heimdall_p999_range"][1]),
        "QuietCutLo": math.floor(100 * q["distance_cut_of_heimdall_p999_range"][0]),
        "QuietCutHi": pct(q["distance_cut_of_heimdall_p999_range"][1]),
        "FigHeimMs": f"{study['read_from_figure_10']['p999_ms']['heimdall']:.1f}",
        "FigHedgeMs": f"{study['read_from_figure_10']['p999_ms']['hedging']:.1f}",
        "FigGateMs": f"{study['read_from_figure_10']['p999_ms']['heimdall_plus_euclidean']:.1f}",
    }


def tex(m):
    lines = ["% Generated by scripts/typeset.py from results/summary.json. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    return "\n".join(lines) + "\n"


def filled(template, m):
    def fill(match):
        return str(m[match.group(1)])
    return re.sub(r"\{\{(\w+)\}\}", fill, template.read_text())


def readme(m):
    return filled(TEMPLATE, m)


def further(m):
    return filled(FURTHER_TEMPLATE, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    m = macros(json.loads((ROOT / "results" / "summary.json").read_text()))
    wanted = {TEX: tex(m), README: readme(m), FURTHER: further(m)}
    if a.check:
        bad = [p.relative_to(ROOT) for p, text in wanted.items() if not p.exists() or p.read_text() != text]
        for p in wanted:
            print(p.relative_to(ROOT), "DIFFERS from results/summary.json" if p.relative_to(ROOT) in bad
                  else "matches results/summary.json")
        return 1 if bad else 0
    for p, text in wanted.items():
        p.write_text(text)
    print(f"wrote paper/numbers.tex, README.md and docs/further.md from {len(m)} numbers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
