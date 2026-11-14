# The drivers behind the development results

These scripts produced the result files of the artefact's traces, the simulator and the cost benchmark. They are kept as they were run, so they read and write the layout they were run in; this repository files their outputs under clearer names.

| script | what it does | wrote | filed here as |
|---|---|---|---|
| [fetch_data.sh](fetch_data.sh) | downloads Heimdall's repository and client-level dataset and checks them against [data/SHA256SUMS](../../data/SHA256SUMS) | `data/` | not included ([NOTICE.md](../../NOTICE.md)) |
| [convert_replays.py](convert_replays.py) | turns the artefact's replay logs into parquet files | `data/parquet/` | not included |
| [validate_pipeline.py](validate_pipeline.py) | checks the reimplemented labelling, features and training against the artefact's own files | `results/validation/` | [results/development/pipeline](../../results/development/pipeline) |
| [run_detection.py](run_detection.py) | trains Heimdall's network, the ensemble and the Bayesian network on the first minute and scores every signal on every later read | `results/detection/`, `results/models/` | [results/development/detection](../../results/development/detection) |
| [sim_validate.py](sim_validate.py) | simulates the artefact's four replayed policies and compares with its logs | `results/simulator/` | [results/routing/artefact_replays.json](../../results/routing/artefact_replays.json) |
| [sim_gates.py](sim_gates.py) | simulates every gate and timeout policy | `results/gate/` | [results/routing/simulated](../../results/routing/simulated) |
| [bench_cost.py](bench_cost.py) | builds the input of the C benchmark [bench/gate_cost.c](../../bench/gate_cost.c) and runs it on one pinned core | `results/cost/` | [results/cost](../../results/cost) |

Run them from the repository's root after `pip install -e .` and `pip install -r requirements.txt`. The held-out traces were scored by the same functions ([docs/reproduce.md](../../docs/reproduce.md)).
