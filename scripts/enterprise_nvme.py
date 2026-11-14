#!/usr/bin/env python3
"""results/routing/enterprise_nvme.json: every run on two identical enterprise NVMe drives of one machine.

Reads the drive experiment's per-run records (RUNS/<pair>/r<round>/<policy>/ with summary.json, reads.npz and
quiet.jsonl.gz, earlier attempts of a run in rejected/<n>/), the quiet criterion, the replayed traces' lengths and
each pair's timeout H per client (train/<pair>/train.json), and writes one record per pair, round and policy:

  * the run that was kept, labelled "quiet" when every second of it meets the quiet criterion, or "shared"
    when it fails the criterion only on other users' processes, which ran on other cores (the two drives saw
    nothing but the replay's own I/O, and our own and the system's processes stayed within the criterion);
  * its read-latency statistics over the scored reads, recomputed from reads.npz and checked against the run's
    own summary, with the number of scored reads above its p99.9, and the same per client;
  * how many attempts before it were rejected and run again; a policy whose four attempts all failed is
    recorded as "missing".

A policy not yet run in a round has no record, so a round is whole for a pair when all sixteen policies have one.
Usage: enterprise_nvme.py EXPERIMENT_DIR   (the folder holding runs/, inputs/, train/ and quiet_criterion.json)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "routing" / "enterprise_nvme.json"
PAIRS = ["P1", "P2", "P3", "P4", "P5", "P6", "K", "S1", "S3"]
POLICIES = ["baseline", "heimdall", "heimdall+timeout", "euclid", "euclid+timeout", "mahal", "mahal+timeout",
            "tracker-mixed", "tracker-mixed+timeout", "timeout-hedging", "oracle-hedge+timeout", "oracle-flip+timeout",
            "oracle-mixed+timeout", "oracle-hedgeslow+timeout", "oracle-hedgefast+timeout", "tracker-hedgeslow"]
ATTEMPTS = 4          # a run that fails the criterion is run again within its round, up to four attempts in all
DRIVE = "Dell Ent NVMe v2 AGN RI U.2 3.84TB, firmware 2.3.0, PCIe 4.0 x4"


def quiet_summary(quiet, replay, drives):
    """The record of every second inside the replay, summarised as the criterion reads it."""
    s0, s1 = replay["start_mono_ns"], replay["end_mono_ns"]
    rows = []
    for a, b in zip(quiet, quiet[1:]):
        if b["t_ns"] <= s0 or a["t_ns"] >= s1:
            continue
        cpu, cls = b["cpu"], b["cls"]
        r = {"users_ticks": cls[2], "seated": b["seated"], "attributed_cpu_ticks": cls[1] + cls[2] + cls[3],
             "nonuser_cpu_ticks": cls[1] + cls[3], "unattributed_cpu_ticks": cpu[0] + cpu[1] + cpu[2] - sum(cls)}
        for k, d in enumerate(drives):
            dk, ours = b["disks"].get(d, [0] * 6), b["our_io"][k]
            r[f"drive{k}_foreign_bytes"] = (dk[1] + dk[3]) * 512 - (ours[1] + ours[3])
        rows.append(r)
    if not rows:
        return {"ticks": 0}

    def win5(xs):
        return max(sum(xs[i:i + 5]) for i in range(max(1, len(xs) - 4)))
    out = {"ticks": len(rows), "users_ticks_max": max(r["users_ticks"] for r in rows),
           "seated_max": max(r["seated"] for r in rows),
           "attributed_cpu_ticks_max": max(r["attributed_cpu_ticks"] for r in rows),
           "unattributed_cpu_5s_max": win5([r["unattributed_cpu_ticks"] for r in rows]),
           "nonuser_cpu_ticks_max": max(r["nonuser_cpu_ticks"] for r in rows)}
    for k in range(len(drives)):
        fb = [r[f"drive{k}_foreign_bytes"] for r in rows]
        out[f"drive{k}_foreign_bytes_sum"] = sum(fb)
        out[f"drive{k}_foreign_bytes_5s_max"] = win5(fb)
    return out


def drives_untouched(q, crit):
    return all(q[f"drive{k}_foreign_bytes_5s_max"] <= crit["drive_foreign_bytes_5s_max"][k]
               and q[f"drive{k}_foreign_bytes_sum"] <= crit["drive_foreign_bytes_replay_max"][k] for k in (0, 1))


def quiet(q, crit):
    """Every second: no other process beyond the limit, no CPU unaccounted for, nobody seated, drives untouched."""
    return (q["ticks"] > 0 and q["attributed_cpu_ticks_max"] <= crit["attributed_cpu_ticks_max"]
            and q["unattributed_cpu_5s_max"] <= crit["unattributed_cpu_5s_max"]
            and q["users_ticks_max"] <= crit["users_ticks_max"] and q["seated_max"] <= crit["seated_max"]
            and drives_untouched(q, crit))


def shared(q, crit):
    """Fails the criterion only on other users' processes: the drives untouched, nobody seated, our own and the
    system's processes within the limit, CPU no process accounts for excused only while another user ran."""
    return (q["ticks"] > 0 and q["seated_max"] <= crit["seated_max"] and drives_untouched(q, crit)
            and q["nonuser_cpu_ticks_max"] <= crit["attributed_cpu_ticks_max"]
            and not (q["users_ticks_max"] <= crit["users_ticks_max"]
                     and q["unattributed_cpu_5s_max"] > crit["unattributed_cpu_5s_max"]))


def verdict(att, crit):
    """'quiet', 'shared' or None for one attempt, from its own record of every second."""
    summ = json.loads((att / "summary.json").read_text())
    with gzip.open(att / "quiet.jsonl.gz", "rt") as f:
        rec = [json.loads(line) for line in f if line.strip()]
    q = quiet_summary(rec, summ["replay"], summ["drives"])
    stored = summ.get("quiet") or {}
    for k, v in q.items():
        if k in stored and stored[k] != v:
            raise SystemExit(f"{att}: {k} recomputed {v}, the run stored {stored[k]}")
    if quiet(q, crit):
        return summ, "quiet"
    if shared(q, crit) and summ.get("shared_ok"):
        return summ, "shared"
    return summ, None


def by_client(lat, n0, H=None):
    """Each client's reads (client c's go to drive c unless a policy moves them): count, median and p99.9, and how
    many of them lie above the pooled p99.9.  Given H, the timeout of each client, also the pooled p99.9 with every
    read capped at H plus the other drive's median, the soonest a copy sent at H could have answered it."""
    p999 = np.percentile(lat, 99.9)
    parts = [lat[:n0], lat[n0:]]
    out = {"clients": [{"n": int(len(x)), "p50": float(np.percentile(x, 50)), "p99.9": float(np.percentile(x, 99.9)),
                        "above_pooled_p999": int(np.sum(x > p999))} for x in parts]}
    if H is not None:
        m = [c["p50"] for c in out["clients"]]
        out["capped_p999"] = float(np.percentile(np.concatenate(
            [np.minimum(parts[0], H[0] + m[1]), np.minimum(parts[1], H[1] + m[0])]), 99.9))
    return out


def statistics(att, summ, H=None):
    """Read latency over the scored reads, recomputed from every read and checked against the run's summary."""
    z = np.load(att / "reads.npz")
    lat, flags = z["latency_us"].astype(np.float64), z["flags"]
    p999 = float(np.percentile(lat, 99.9))
    m = {"n": int(len(lat)), "mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
         "p99": float(np.percentile(lat, 99)), "p99.9": p999,
         "rerouted": float(np.mean((flags & 3) >= 1)), "hedged": float(np.mean((flags & 3) == 2)),
         "above_p999": int(np.sum(lat > p999))}
    s = summ["metrics"]["pooled"]
    for k in ("n", "mean", "p50", "p99", "p99.9", "rerouted", "hedged"):
        if not np.isclose(m[k], s[k], rtol=1e-6, atol=1e-6):
            raise SystemExit(f"{att}: {k} recomputed {m[k]}, the run's summary says {s[k]}")
    return m | by_client(lat, int(z["n_c"][0]), H)


def policy_record(pdir, crit, H):
    """The policy's record in one round, or None while it has not finished."""
    attempts = sorted((pdir / "rejected").glob("*"), key=lambda x: int(x.name)) if (pdir / "rejected").exists() else []
    rejected = sum((a / "summary.json").exists() for a in attempts)
    if not (pdir / "summary.json").exists():
        return None
    summ, kind = verdict(pdir, crit)
    if kind is None:
        return {"kind": "missing", "rejected_attempts": rejected + 1} if rejected + 1 >= ATTEMPTS else None
    return {"kind": kind, "rejected_attempts": rejected, "replay_s": summ["replay"]["run_s"],
            "pooled": statistics(pdir, summ, H if pdir.name == "baseline" else None)}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    crit = json.loads((src / "quiet_criterion.json").read_text())
    if crit.get("version") != 2:
        sys.exit("expected the second version of the quiet criterion")
    out = {"drives": [DRIVE, DRIVE],
           "layout": "the two members of a RAID0 of 512 KiB chunks, each client's I/O kept to its own member's chunks",
           "quiet_criterion": {k: crit[k] for k in ("attributed_cpu_ticks_max", "unattributed_cpu_5s_max",
                                                    "users_ticks_max", "seated_max", "drive_foreign_bytes_5s_max",
                                                    "drive_foreign_bytes_replay_max")},
           "policies": POLICIES, "pairs": {}}
    for pair in PAIRS:
        trace_s = float(np.load(src / "inputs" / f"{pair}.npz")["duration_s"])
        train = json.loads((src / "train" / pair / "train.json").read_text())["clients"]
        H = [train["c0"]["H_us"], train["c1"]["H_us"]]
        rounds = {}
        for rdir in sorted((src / "runs" / pair).glob("r*"), key=lambda x: int(x.name[1:])):
            rec = {}
            for pol in POLICIES:
                if (rdir / pol).is_dir():
                    r = policy_record(rdir / pol, crit, H)
                    if r is not None:
                        rec[pol] = r
            if rec:
                rounds[rdir.name[1:]] = rec
        out["pairs"][pair] = {"trace_s": trace_s, "H_us": H, "rounds": rounds}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    whole = {p: [r for r, v in d["rounds"].items() if len(v) == len(POLICIES)] for p, d in out["pairs"].items()}
    print("wrote", OUT.relative_to(ROOT), "| whole rounds:", "  ".join(f"{p} {','.join(w) or '-'}" for p, w in whole.items()))


if __name__ == "__main__":
    main()
