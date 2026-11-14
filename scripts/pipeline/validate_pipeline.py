#!/usr/bin/env python3
"""Check the reimplemented Heimdall pipeline against the artefact's own files.

For each device trace of the given pairs:
  1. tail_v1 labels  vs  training_results/temp{d}      (rows kept, labels)
  2. feat_v6 features vs training_results/mldrive{d}.csv (every column)
  3. artefact network: reproduce eval.stats accuracy/FPR/FNR on Heimdall's
     own random 50/50 split (sklearn train_test_split, random_state=42).

Purpose: show that the labels, features and model used everywhere else are Heimdall's.
Output: results/validation/pipeline_<pair>.json; exit 1 on any mismatch
beyond the stated tolerance.
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from admission import RESULTS
from admission.provenance import stamp
from admission.heimdall import FEATURES, ArtefactModel, feat_v6, tail_v1
from admission.traces import DEVICES, by_name


def read_eval_stats(path):
    txt = path.read_text()
    acc = float(re.search(r"accuracy\s+([0-9.]+)", txt).group(1))
    fpr = float(re.search(r"FPR = ([0-9.]+)", txt).group(1))
    fnr = float(re.search(r"FNR = ([0-9.]+)", txt).group(1))
    return acc, fpr, fnr


def check(trace_name):
    t = by_name(trace_name)
    tr = t.pair_dir / "...".join(DEVICES) / "flashnet" / "training_results"
    out = {"trace": trace_name}
    df = pd.read_parquet(t.replay_parquet("baseline"))
    # tail_v1.py reads the log with pd.read_csv(path) (no header=None), so the
    # first logged I/O becomes the header and is dropped; replicate that.
    df = df.iloc[1:].reset_index(drop=True)
    lab = tail_v1(df)
    kept_rows = lab.order[lab.kept]
    mine = df.loc[kept_rows, ["ts_record", "latency", "io_type", "size", "offset", "ts_submit"]].reset_index(drop=True)
    mine["reject"] = lab.tail[lab.kept].astype(int)

    ref = pd.read_csv(tr / f"temp{t.dev}")
    out["labels_rows_ref"] = len(ref)
    out["labels_rows_mine"] = len(mine)
    out["ip_latency"] = lab.ip_latency
    out["ip_throughput"] = lab.ip_throughput
    out["outlier_counts"] = {str(k): int((lab.outlier == k).sum()) for k in range(4)}
    if len(ref) == len(mine):
        same_rows = (np.isclose(ref["ts_submit"], mine["ts_submit"]) & (ref["offset"] == mine["offset"])
                     & (ref["latency"] == mine["latency"]))
        out["labels_rows_identical_frac"] = float(same_rows.mean())
        out["labels_agree_frac"] = float((ref["reject"] == mine["reject"]).mean())
    else:
        # align on (ts_submit, offset, latency)
        key = ["ts_submit", "offset", "latency"]
        m = ref.merge(mine, on=key, how="outer", suffixes=("_ref", "_mine"), indicator=True)
        out["labels_rows_only_ref"] = int((m["_merge"] == "left_only").sum())
        out["labels_rows_only_mine"] = int((m["_merge"] == "right_only").sum())
        both = m[m["_merge"] == "both"]
        out["labels_agree_frac"] = float((both["reject_ref"] == both["reject_mine"]).mean())

    feats_mine = feat_v6(mine)
    feats_ref = pd.read_csv(tr / f"mldrive{t.dev}.csv")
    out["feat_rows_ref"] = len(feats_ref)
    out["feat_rows_mine"] = len(feats_mine)
    if len(feats_ref) == len(feats_mine):
        out["feat_col_agree"] = {c: float((feats_ref[c].to_numpy() == feats_mine[c].to_numpy()).mean())
                                 for c in feats_ref.columns}

    # artefact network on Heimdall's own split of its own feature file
    model = ArtefactModel(t.artefact_weights_dir(), f"mldrive{t.dev}.csv")
    x = feats_ref.drop(columns=["reject"])
    y = feats_ref["reject"]
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    pred = model.logit(x_test[FEATURES].to_numpy(np.float64)) > 0  # sigmoid > 0.5
    yt = y_test.to_numpy()
    acc = float((pred == yt).mean())
    tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum()); tp = int(((pred == 1) & (yt == 1)).sum())
    ref_acc, ref_fpr, ref_fnr = read_eval_stats(tr / f"mldrive{t.dev}" / "nnK" / "eval.stats")
    out["model_acc_mine"] = round(acc, 4)
    out["model_fpr_mine"] = round(fp / (fp + tn + 0.1), 3)
    out["model_fnr_mine"] = round(fn / (tp + fn + 0.1), 3)
    out["model_acc_ref"], out["model_fpr_ref"], out["model_fnr_ref"] = ref_acc, ref_fpr, ref_fnr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args()
    outdir = RESULTS / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name in a.traces:
        r = check(name)
        ok = r["labels_rows_ref"] == r["labels_rows_mine"] and r.get("labels_agree_frac", 0) > 0.999
        ok &= abs(r["model_acc_mine"] - r["model_acc_ref"]) <= 0.006
        ok &= "feat_col_agree" in r and min(r["feat_col_agree"].values()) > 0.999
        r["matches_artefact"] = bool(ok)
        print(json.dumps(r, indent=1), flush=True)
        (outdir / f"pipeline_{name}.json").write_text(json.dumps(stamp(r), indent=1))
        print(f"{name}: {'matches' if ok else 'DIFFERS'}", flush=True)
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
