# Results

Every number the paper quotes is computed from these files by [scripts/summary.py](../scripts/summary.py) into [summary.json](summary.json). [tables.md](tables.md) shows them as tables, and [METHOD.md](../METHOD.md) says how each was made.

| path | what |
|---|---|
| [heldout/traces.json](heldout/traces.json) | the 60 held-out cuts: source, volume, checksums of the cut and of its replay, the replay's quiet record |
| [heldout/detection/](heldout/detection) | per trace and window, the AUROC of 39 scores for Heimdall's wrong guesses, under both truths (`signals_G2_request`: each read's own latency, `signals`: Heimdall's period labels), with error rates and counts |
| [heldout/table/](heldout/table) | the table one trace taught: every visited cell's entry, reads, slow reads and Heimdall's calls |
| [heldout/hero/](heldout/hero) | binned counts behind the picture at the top of the README |
| [heldout/frozen.sha256](heldout/frozen.sha256) | the pipeline as frozen before the held-out traces were replayed |
| [development/detection/](development/detection) | the same per-window scores for the artefact's twelve device traces and five further replays |
| [development/pipeline/](development/pipeline) | the reimplemented labelling, features and accuracy against the artefact's own files |
| [classwise/](classwise) | the same ranking with the guess held fixed, with its controls, per window ([scripts/classwise.py](../scripts/classwise.py)) |
| [direct/](direct) | four forecasts of slow reads, per window ([scripts/direct.py](../scripts/direct.py)) |
| [routing/artefact_replays.json](routing/artefact_replays.json) | read-latency statistics of the artefact's own replay logs, with the simulator's version of each |
| [routing/enterprise_nvme.json](routing/enterprise_nvme.json) | sixteen policies, nine pairs, in rounds, on two identical enterprise NVMe drives of one machine: every kept run, quiet or shared, and the attempts rejected before it. `H_us`: each pair's hedging timeout per client. Every run holds each client's reads (`clients`, `p50` the drive's median with no admission control) and how many lie above the pooled p99.9, the run with no admission control also `capped_p999` as in the line below |
| [routing/quiet_drives.json](routing/quiet_drives.json) | fifteen policies, five rounds, five pairs on two unlike drives of a quiet machine. `H_us`: each pair's hedging timeout per client, in microseconds, from its training window. In each run with no admission control, `client0` and `client1` hold each drive's own read latencies (`p50` its median), `clients_above_pooled_p999` how many of each client's reads lie above the pooled p99.9, and `capped_p999` the pooled p99.9 with every read capped at its client's H plus the other drive's median ([scripts/quiet_drives_floor.py](../scripts/quiet_drives_floor.py)) |
| [routing/identical_nvme.json](routing/identical_nvme.json) | the policies on two consumer NVMe drives of a busy machine, one run a pair |
| [routing/simulated/](routing/simulated) | 105 simulated policies a pair |
| [retraining/](retraining) | Heimdall's network and the nine estimators refitted during the replay on four schedules (`heimdall_s7`, `minute`, `slide10`, `grow10`), per schedule and trace: the schedule as run (`spec`, `segments`) and per window the AUROCs of the table and the nine, over every read and with the guess fixed, with counts ([scripts/retraining](../scripts/retraining)) |
| [other_decisions/](other_decisions) | the same per-window scores on two other decisions, with controllers of our own |
| [cost/cost.json](cost/cost.json) | nanoseconds a decision |
| [study_reported.json](study_reported.json) | the values Gohil et al. report, and our reading of their Figure 10, marked as an estimate |
| [heimdall_reported.json](heimdall_reported.json) | two statements of Heimdall's paper that the text quotes |
| [SHA256SUMS](SHA256SUMS) | the checksum of every file above |

In the per-window files the table is called `tracker`. `tracker_size_only`, `tracker_no_age`, `tracker_no_quiet` and `tracker_admitted` are its ablations, and `supervised_w0_G2` is a detector fitted on every labelled read of the first minute.
