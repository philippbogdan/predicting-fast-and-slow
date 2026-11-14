#!/usr/bin/env python3
"""Convert the artefact's per-I/O replay logs and raw traces (CSV) to Parquet.

Purpose: every later step reads the same typed columns quickly.  The
conversion is lossless (all columns, all rows, row order kept) and is
verified: an existing Parquet file is reused only if its row count equals the
CSV's line count.  Files are streamed in batches so memory stays bounded, and
written atomically (temp file + rename) so an interrupted run leaves no
truncated output behind.

Usage: python scripts/convert_replays.py [--workers N]
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from admission.traces import ALL_DEVICE_TRACES, PARQUET_ROOT, POLICIES, REPLAY_COLUMNS

REPLAY_SCHEMA = pa.schema([
    ("ts_record", pa.float64()),
    ("latency", pa.int64()),
    ("io_type", pa.int8()),
    ("size", pa.int64()),
    ("offset", pa.int64()),
    ("ts_submit", pa.float64()),
    ("ret", pa.int64()),
])
RAW_COLUMNS = ["ts", "dev", "offset", "size", "io_type"]
RAW_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("dev", pa.int64()),
    ("offset", pa.int64()),
    ("size", pa.int64()),
    ("io_type", pa.int8()),
])


def line_count(path):
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def convert_one(src, dst, columns, schema, delimiter):
    if not src.exists():  # not downloaded (fetch the data first)
        return f"absent {src}"
    expected = line_count(src)
    if dst.exists():
        try:
            if pq.ParquetFile(dst).metadata.num_rows == expected:
                return f"ok   {dst} rows={expected}"
        except Exception:  # unreadable -> rebuild
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".parquet.tmp")
    reader = pacsv.open_csv(
        src,
        read_options=pacsv.ReadOptions(column_names=columns, block_size=1 << 24),
        parse_options=pacsv.ParseOptions(delimiter=delimiter),
        convert_options=pacsv.ConvertOptions(column_types=schema),
    )
    rows = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
            rows += batch.num_rows
    if rows != expected:
        os.remove(tmp)
        raise RuntimeError(f"{src}: wrote {rows} rows, expected {expected}")
    os.replace(tmp, dst)
    return f"done {dst} rows={rows}"


def convert_task(args):
    kind, trace, policy = args
    if kind == "replay":
        return convert_one(trace.replay_csv(policy), trace.replay_parquet(policy),
                           REPLAY_COLUMNS, REPLAY_SCHEMA, ",")
    dst = PARQUET_ROOT / trace.pair / "raw" / f"d{trace.dev}.parquet"
    return convert_one(trace.raw_trace, dst, RAW_COLUMNS, RAW_SCHEMA, " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    tasks = [("replay", t, p) for t in ALL_DEVICE_TRACES for p in POLICIES]
    tasks += [("raw", t, None) for t in ALL_DEVICE_TRACES]
    failures = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(convert_task, j) for j in tasks]
        for f in futs:
            try:
                print(f.result(), flush=True)
            except Exception as e:  # report and keep going; exit code says it failed
                failures += 1
                print(f"FAIL {e}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
