# Reproduce

Three layers, from the cheapest.

## 1. Every quoted number, from the result files

```sh
python check.py
```

Standard library only, a few seconds. It recomputes every number the paper, the README and [further.md](further.md) quote from the per-window files in [results/](../results/) and fails if a result file, the summary, the paper's numbers, either page or [results.md](results.md) has drifted. [tests/test_check.py](../tests/test_check.py) shows it failing on a changed result.

## 2. The figures and the paper

```sh
pip install matplotlib numpy
python figures/heldout.py && python figures/table.py && python figures/tail.py && python figures/hero.py
cd paper && pdflatex main && pdflatex main
```

Each figure script reads [results/](../results/) through [scripts/summary.py](../scripts/summary.py). The paper takes its numbers from `paper/numbers.tex`, which [scripts/typeset.py](../scripts/typeset.py) writes.

## 3. The result files, from replays

The per-window files are computed from baseline replays, one parquet file a trace, which are not in this repository: they carry the traces ([NOTICE.md](../NOTICE.md)). To make them again:

1. **Traces.** The UMass Financial and WebSearch traces and the MSR Cambridge volumes for the held-out set ([scripts/heldout/normalise_sources.py](../scripts/heldout/normalise_sources.py), then [prepare_traces.py](../scripts/heldout/prepare_traces.py), which writes each cut and its SHA-256; ours are in [results/heldout/traces.json](../results/heldout/traces.json)). Heimdall's artefact for the twelve device traces and its replay logs.
2. **Replay** each cut with no admission control on a quiet drive: `bench/pair_replayer --policy 0 --cut 1`, with `bench/quietrec` recording the machine beside it ([method.md](method.md)).
3. **Train and score.** `pip install -r requirements.txt`. `admission.protocol.load_trace` builds Heimdall's labels and features from a replay, `admission.train.train_mlps` and `train_bnn` train the networks, `admission.compare.score_all` scores every estimator and the table. Then, per trace:

   ```sh
   python scripts/classwise.py NAME --replay NAME.parquet --models NAME.npz --out results/classwise/heldout
   python scripts/direct.py    NAME --replay NAME.parquet --models NAME.npz --out results/direct/heldout
   ```

4. **Seal.** `python check.py --seal` rewrites the summary, the paper's numbers, the README, [further.md](further.md), [results.md](results.md) and the checksums from the new files.

The drive measurements of [results/routing](../results/routing) need two drives: the same replayer with `--policy 2|3|4`, `--gate`, `--action` and `--timeout` as its header describes, and `--map0` and `--map1` where the two drives are the members of one RAID0. For the identical enterprise drives, [scripts/enterprise_nvme.py](../scripts/enterprise_nvme.py) turns the per-run records (each run's `summary.json`, `reads.npz` of every scored read and `quiet.jsonl.gz` of every second, earlier attempts in `rejected/`) into [results/routing/enterprise_nvme.json](../results/routing/enterprise_nvme.json), and [scripts/rebuild.sh](../scripts/rebuild.sh) runs that, the seal, Figure 3, the paper and the checks in one go.

Training is not bit-reproducible across machines, and a drive is not the same drive twice: a second replay gives somewhat different windows. The traces' checksums and every rejected attempt's count are recorded so that a difference can be told from a mistake.

## Versions

Python 3.12 with NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.0, PyArrow 26.0.0, Numba 0.68.0, scikit-learn 1.9.1 and PyTorch 2.14.1; the C sources with `cc -O2`.
