#!/usr/bin/env python3
"""Adds to results/routing/quiet_drives.json what the timeout's floor needs on the unlike pair.

Per pair, H_us: the timeout of each client, the 98th percentile of its read latencies in the pair's training window
(train/<pair>/train.json of the unlike pair's records).  Per round's run with no admission control, from that run's
every scored read (runs/<pair>/r<round>/baseline/reads.npz, or the earlier attempt in rejected/<n>/ whose p99.9 is
the one the file holds): how many of each client's reads lie above the pooled p99.9 (clients_above_pooled_p999),
and the pooled p99.9 with every read capped at its client's H plus the other drive's median (capped_p999).
Each drive's median is already in the file (client0 and client1, p50: with no admission control client c's reads
all go to drive c).  Usage: quiet_drives_floor.py RECORDS_DIR   (the folder holding train/ and runs/)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enterprise_nvme import by_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "results" / "routing" / "quiet_drives.json"


def attempt(base, p999):
    """The attempt of a run (the last one, or an earlier one set aside) whose pooled p99.9 the file holds."""
    rejected = sorted((base / "rejected").glob("*"), key=lambda x: int(x.name)) if (base / "rejected").exists() else []
    for att in [base] + rejected:
        z = att / "reads.npz"
        if z.exists():
            lat = np.load(z)["latency_us"].astype(np.float64)
            if abs(np.percentile(lat, 99.9) / p999 - 1) < 1e-4:
                return att
    sys.exit(f"{base}: no attempt has the p99.9 {p999}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    q = json.loads(FILE.read_text())
    for pair, pol in q["pairs"].items():
        train = json.loads((src / "train" / pair / "train.json").read_text())["clients"]
        H = [train["c0"]["H_us"], train["c1"]["H_us"]]
        q.setdefault("H_us", {})[pair] = H
        for rnd, rec in pol["baseline"].items():
            att = attempt(src / "runs" / pair / f"r{rnd}" / "baseline", rec["pooled"]["p99.9"])
            z = np.load(att / "reads.npz")
            b = by_client(z["latency_us"].astype(np.float64), int(z["n_c"][0]), H)
            for c in (0, 1):
                if abs(b["clients"][c]["p50"] / rec[f"client{c}"]["p50"] - 1) > 1e-4:
                    sys.exit(f"{pair} round {rnd}: client {c}'s median {b['clients'][c]['p50']} is not the file's")
            rec["clients_above_pooled_p999"] = [c["above_pooled_p999"] for c in b["clients"]]
            rec["capped_p999"] = b["capped_p999"]
    FILE.write_text(json.dumps(q, indent=1) + "\n")
    print("added H, the tail's clients and the capped p99.9 to", FILE.relative_to(ROOT))


if __name__ == "__main__":
    main()
