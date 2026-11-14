#!/usr/bin/env python3
"""The held-out traces, cut from the normalised volumes by a rule fixed before any of them was replayed.

Per volume (an MSR Cambridge volume, or an ASU of a UMass/SPC trace): consecutive 3-minute windows from the
volume's first request, as Heimdall's per_3mins windows are cut, and two variants:
  orig  the window with the most requests (Heimdall's iops_p100), at its own pace;
  x8    the 24 minutes (8 consecutive windows) with the most requests, replayed eight times faster (Heimdall's
        largest rerate in the artefact, rerate_8.00), so that it too lasts 3 minutes.
Ties go to the earliest window.  A variant is kept when it holds at least MIN_READS reads, so that each of the
four 30-second scoring windows holds some hundreds.  Sizes are rounded up to 512 bytes (O_DIRECT).
Writes <out>/traces/<name>.trace ("ts_ms 0 offset size type", type 1 read, sorted by time) for every kept
variant, and <out>/traces.json describing every variant, kept or not.
Usage: prepare_traces.py <norm_dir> <out_dir>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

W_US = 180e6
MIN_READS = 3000


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    norm, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "traces").mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(norm.glob("*.npz")):
        z = np.load(f)
        ts, off, size, rd = z["ts_us"], z["offset"], z["size"], z["read"].astype(np.int8)
        source, volume = f.stem.split("__")
        win = (ts // W_US).astype(np.int64)
        cnt = np.bincount(win)
        b = int(np.argmax(cnt))
        variants = {"orig": (win == b, b, 1.0)}
        if len(cnt) >= 8:
            seg = np.convolve(cnt, np.ones(8, np.int64), mode="valid")       # seg[k] = cnt[k:k+8].sum()
            k8 = int(np.argmax(seg))
            variants["x8"] = ((win >= k8) & (win < k8 + 8), k8, 8.0)
        for var, (m, start, rate) in variants.items():
            t = (ts[m] - start * W_US) / rate
            name = f"{source}.{volume}.{var}"
            reads = int(rd[m].sum())
            row = {"name": name, "source": source, "volume": volume, "variant": var, "first_window": start,
                   "rerate": rate, "requests": int(m.sum()), "reads": reads, "read_share": reads / max(1, int(m.sum())),
                   "iops": float(m.sum() / 180.0), "mean_size": float(size[m].mean()) if m.any() else 0.0,
                   "kept": reads >= MIN_READS}
            if row["kept"]:
                o = np.argsort(t, kind="stable")
                sz = (size[m][o] + 511) // 512 * 512
                p = out / "traces" / f"{name}.trace"
                with open(p, "w") as fo:
                    for a, b_, c, d in zip(t[o] / 1000.0, off[m][o], sz, rd[m][o]):
                        fo.write(f"{a:.6f} 0 {int(b_)} {int(c)} {int(d)}\n")
                row["sha256"] = sha256(p)
                row["duration_s"] = 180.0
            rows.append(row)
            print(f"{name:42s} start={start:5d} req={row['requests']:7d} reads={reads:7d} "
                  f"iops={row['iops']:8.1f} kept={row['kept']}", flush=True)
    (out / "traces.json").write_text(json.dumps(rows, indent=1))
    kept = [r for r in rows if r["kept"]]
    print(f"{len(kept)} of {len(rows)} variants kept; volumes: {len({(r['source'], r['volume']) for r in kept})}")


if __name__ == "__main__":
    main()
