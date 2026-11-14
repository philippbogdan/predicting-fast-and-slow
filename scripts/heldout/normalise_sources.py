#!/usr/bin/env python3
"""Turn the downloaded held-out traces into one array file per volume, in one form.

Sources:
  umass  UMass Trace Repository, SPC format (Financial1, Financial2, WebSearch1-3):
         ASU,LBA(512-byte blocks),size(bytes),opcode(r/R read, w/W write),timestamp(s from trace start).
         Each ASU (a separate storage unit) is one volume.
  msr    MSR Cambridge volumes hm_0, mds_0, mds_1, prn_0, as converted to MQSim's ASCII format in
         github.com/Tina0113/trace: arrival(ns from trace start) device LBA(512-byte sectors) size(sectors)
         type(1 read, 0 write).  One file per volume.
Output, per volume: <out>/<source>__<volume>.npz with ts_us (float64, from the volume's first request, sorted
stably by time), offset (int64 bytes), size (int64 bytes), read (int8), and the raw row count.
Usage: normalise_sources.py <umass_dir> <msr_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def save(out, name, ts_s, off, size, rd):
    o = np.argsort(ts_s, kind="stable")
    ts = ts_s[o]
    np.savez_compressed(out / f"{name}.npz", ts_us=(ts - ts[0]) * 1e6, offset=off[o].astype(np.int64),
                        size=size[o].astype(np.int64), read=rd[o].astype(np.int8), rows=len(ts))
    print(f"{name}: {len(ts)} requests, {100 * rd.mean():.1f}% reads, {(ts[-1] - ts[0]) / 3600:.2f} h", flush=True)


def umass(src, out):
    for f in sorted(Path(src).glob("*.spc")):
        df = pd.read_csv(f, header=None, names=["asu", "lba", "size", "op", "ts"],
                         dtype={"asu": "Int64", "lba": "Int64", "size": "Int64", "op": str, "ts": np.float64})
        n0 = len(df)
        df = df.dropna()                     # Financial2 ends in a truncated record ("7,1")
        df = df[df["size"] > 0]              # WebSearch3 holds requests of 0 bytes
        print(f"{f.stem}: {n0} records, {n0 - len(df)} dropped (truncated or 0 bytes)", flush=True)
        df = df.astype({"asu": np.int64, "lba": np.int64, "size": np.int64}).reset_index(drop=True)
        op = df["op"].str.lower()
        assert set(op.unique()) <= {"r", "w"}, op.unique()
        for asu, g in df.groupby("asu", sort=True):
            save(out, f"umass__{f.stem}.asu{asu}", g["ts"].to_numpy(), g["lba"].to_numpy() * 512, g["size"].to_numpy(),
                 (op[g.index] == "r").to_numpy())


def msr(src, out):
    for f in sorted(Path(src).glob("*/*_mqsim.trace")):
        df = pd.read_csv(f, sep=r"\s+", header=None, names=["ns", "dev", "lba", "sectors", "type"], engine="c",
                         dtype=np.int64)
        assert set(df["type"].unique()) <= {0, 1}
        n0 = len(df)
        df = df[df["sectors"] > 0].reset_index(drop=True)
        print(f"{f.name}: {n0} records, {n0 - len(df)} dropped (0 bytes)", flush=True)
        vol = f.name.replace("_mqsim.trace", "")
        save(out, f"msr__{vol}", df["ns"].to_numpy() / 1e9, df["lba"].to_numpy() * 512, df["sectors"].to_numpy() * 512,
             (df["type"] == 1).to_numpy())


if __name__ == "__main__":
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    umass(sys.argv[1], out)
    msr(sys.argv[2], out)
