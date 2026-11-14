"""Registry of the Heimdall artefact's traces.

The client-level dataset (Google Drive folder linked from Heimdall's
documentation/3_client_level_integration.md) holds six trace pairs.  In each
pair, trace_1 is replayed by client 0 against device 0 (nvme14n1) and trace_2
by client 1 against device 1 (nvme15n1), simultaneously.  Every pair carries
per-I/O replay logs for five policies: baseline, flashnet (= Heimdall),
random, hedging, linnos.

The artefact additionally ships five raw traces without replay logs
(ds_pipeline sample traces and kernel-level benchmark traces); see
RAW_ONLY_TRACES.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import DATA

CLIENT_LEVEL_ROOT = DATA / "client_level_gdrive" / "extracted" / "data"
HEIMDALL_REPO = DATA / "heimdall_repo" / "src"
PARQUET_ROOT = DATA / "parquet"
DEVICES = ("nvme14n1", "nvme15n1")
POLICIES = ("baseline", "flashnet", "random", "hedging", "linnos")

PAIRS = {
    "P1": "alibaba.per_3mins.iops_p10.alibaba_50.97/alibaba.per_3mins.iops_p100.alibaba_9086.35/original...modified.rerate_4.00.resize_4.00",
    "P2": "alibaba.per_3mins.iops_p10.alibaba_50.97/msr.per_3mins.overall_p100.src1_0.2108/original...modified.rw_ratio_40_60.rerate_8.00.resize_2.00",
    "P3": "alibaba.per_3mins.iops_p75.alibaba_9086.23/alibaba.per_3mins.offset_p100.alibaba_9086.3/modified.rerate_0.50...modified.rerate_2.00.resize_2.00",
    "P4": "alibaba.per_3mins.iops_p75.alibaba_9086.23/alibaba.per_3mins.offset_p100.alibaba_9086.3/modified.rerate_0.50...modified.rerate_4.00",
    "P5": "alibaba.per_3mins.overall_p25.alibaba_9076.45/tencent.per_3mins.write_p100.2018-10-09-04.12/modified.rerate_0.50...modified.rerate_0.50",
    "P6": "msr.per_3mins.overall_p50.prxy_0.1490/msr.per_3mins.write_p100.src1_0.3047/modified.rw_ratio_40_60.rerate_8.00.resize_2.00...modified.rw_ratio_40_60.rerate_8.00.resize_2.00",
}

# Raw traces in the artefact that come without replay logs.
RAW_ONLY_TRACES = {
    "S1": HEIMDALL_REPO / "ds_pipeline/data/raw_data/msr.cut.per_10k.most_thpt_rand_iops.1006.trace",
    "S2": HEIMDALL_REPO / "ds_pipeline/data/raw_data/msr.cut.per_10k.rw_20_80.105.trace",
    "S3": HEIMDALL_REPO / "ds_pipeline/data/raw_data/msr.cut.per_10k.rw_40_60.1370.trace",
    "KB": HEIMDALL_REPO / "integration/kernel-level/benchmark/sample_dataset/traceB/traceB.trace",
    "KL": HEIMDALL_REPO / "integration/kernel-level/benchmark/sample_dataset/traceL/traceL.trace",
}

REPLAY_COLUMNS = ["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "ret"]


@dataclass(frozen=True)
class DeviceTrace:
    """One client/device stream of a trace pair, e.g. P3 device 1."""

    pair: str
    dev: int  # 0 -> trace_1 on nvme14n1, 1 -> trace_2 on nvme15n1

    @property
    def name(self) -> str:
        return f"{self.pair}d{self.dev}"

    @property
    def pair_dir(self) -> Path:
        return CLIENT_LEVEL_ROOT / PAIRS[self.pair]

    @property
    def raw_trace(self) -> Path:
        return self.pair_dir / f"trace_{self.dev + 1}.trace"

    def replay_csv(self, policy: str) -> Path:
        return self.pair_dir / "...".join(DEVICES) / policy / f"trace_{self.dev + 1}.trace"

    def replay_parquet(self, policy: str) -> Path:
        return PARQUET_ROOT / self.pair / policy / f"d{self.dev}.parquet"

    def artefact_weights_dir(self) -> Path:
        return self.pair_dir / "...".join(DEVICES) / "flashnet" / "training_results" / f"drive{self.dev}weights"


ALL_DEVICE_TRACES = [DeviceTrace(p, d) for p in PAIRS for d in (0, 1)]

# trace -> timestamp unit of the raw file
RAW_ONLY_UNITS = {"S1": "ms", "S2": "ms", "S3": "ms", "KB": "us", "KL": "us"}


@dataclass(frozen=True)
class RawOnlyTrace:
    """An artefact trace shipped without replay logs, replayed by us on a local NVMe
    SSD (not redistributed: docs/reproduce.md); replay 0 is the baseline."""

    name: str
    dev: int = 0

    @property
    def raw_trace(self) -> Path:
        return RAW_ONLY_TRACES[self.name]

    def replay_parquet(self, policy: str = "baseline", rep: int = 0) -> Path:
        assert policy == "baseline"
        return PARQUET_ROOT / "raw_only" / self.name / f"rep{rep}.parquet"

    @property
    def duration_s(self) -> float:
        first = last = None
        with open(self.raw_trace) as f:
            for line in f:
                t = float(line.split()[0])
                first = t if first is None else first
                last = t
        return (last - first) / (1000.0 if RAW_ONLY_UNITS[self.name] == "ms" else 1e6)


ALL_RAW_ONLY = [RawOnlyTrace(n) for n in RAW_ONLY_TRACES]


def by_name(name: str):
    for t in ALL_DEVICE_TRACES + ALL_RAW_ONLY:
        if t.name == name:
            return t
    raise KeyError(name)
