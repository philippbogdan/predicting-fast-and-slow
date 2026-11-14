# Method

How each number in the paper was made, what the paper has no room for, and how to run it on your own traces. [check.py](check.py) recomputes every number on this page, and [results/tables.md](results/tables.md) lists every trace and every drive run. An AUROC is the median over traces of the worst of four later windows, and a guess is wrong by the read's own latency, unless period labels are named.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/plates/claim-dark.svg">
  <img src="figures/plates/claim-light.svg" width="570" alt="Two panels, one above the other, each with one thin line for each of the {{HeldTraces}} held-out traces across the tenths of its reads, from most suspect on the left to least suspect on the right, in points above or below the trace's own average, and a heavy red line through their medians. Above, ordered by the drive's own table, the lines and the red median fall from well above the average to well below it. Below, ordered by distance from the training data, they stay close to the average.">
</picture>

**Fig. 1.** Sort each trace's reads from the guess a signal finds most suspect to the one it finds least, and cut them into tenths. Under the drive's own table, above, the share of guesses wrong falls from {{ClaimTableFirst}} points over the trace's average to {{ClaimTableLast}} under it, and the most suspect tenth is wrong more often than the least on all {{ClaimTableOrdered}} held-out traces. Under Euclidean distance from the training data, below, it stays within {{ClaimDistWithin}} points of the average, and that order holds on {{ClaimDistOrdered}} traces. One line a trace, the median in red.

## Protocol

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/plates/protocol-dark.svg">
  <img src="figures/plates/protocol-light.svg" width="570" alt="A time axis from 0 to 180 seconds into a replay, with a dash-dot line at 60 seconds through four lanes. Heimdall's network and the nine estimators: a hatched bar over the first minute, lettered fitted, then a dashed line, lettered frozen. T, the knee of these read latencies: a bracket over the first minute, then a dotted line, lettered fixed. The table, every entry from 1/2: a heavy red arrow across all three minutes, lettered learns from every completion. Every later read scored, the worst window counts: the first minute lettered not scored, then four stippled windows of 30 seconds.">
</picture>

**Fig. 2.** The three minutes of one replay. What is fitted is fitted on the first minute and then frozen. Only the table goes on learning.

**Windows.** A replay lasts 180 s. Its first 60 s train everything: the network, every estimator, the thresholds, the timeout. Every read of the remaining 120 s is scored, in four windows of 30 s.

**Retraining.** Freezing the network and the nine estimators after the first minute is how Heimdall's long-term evaluation deploys it, and it is not what the result rests on. Heimdall's paper also retrains: it checks the model's accuracy every minute and, below {{RetrainBelow}}%, refits on the last minute. Refitted that way during the replay, network and estimators alike, the check fires on {{RetrainFired}} of the {{RetrainTraces}} traces and the result stands. Across that schedule and {{RetrainOthers}} others, down to a refit every {{RetrainEvery}} seconds, the table ranks the wrong guesses at a median {{RetrainTableLo}} to {{RetrainTableHi}} and is ahead of the best of the nine on {{RetrainAheadLo}} to {{RetrainAheadHi}} of the {{RetrainTraces}} traces. Heimdall still calls slow a median {{RetrainCallsAbove}} points more often than reads come back slow, because it retrains on the labels that taught it. Among reads it calls fast the tie stays a tie: the table at {{RetrainFastTableLo}} to {{RetrainFastTableHi}}, the best of the nine at {{RetrainFastBestLo}} to {{RetrainFastBestHi}}. This is measured offline, on replays run with no admission control. It scores each arm's guesses and says nothing of the tail ([results/retraining](results/retraining), [scripts/retraining](scripts/retraining)).

**Heimdall.** A 12-128-16-1 network. Its inputs are the read's size, the client's queue length on the drive, and the queue length, latency and throughput of the last three I/Os that completed there. [admission/heimdall.py](admission/heimdall.py) reimplements the artefact's period labelling, noise filtering, features and training recipe, and reproduces its labels and features row for row ([results/development/pipeline](results/development/pipeline)). The artefact orders I/Os with an unstable sort, which NumPy runs differently on processors with AVX-512, so the module carries a port of NumPy's generic sort. At decision time the network is fed what the artefact's replayer feeds it: the last three I/Os that completed, not the last three submitted. Held out it is wrong on {{HeldWrongLo}} to {{HeldWrongHi}}% of later reads by each read's own latency, and on {{HeldWrongPerLo}} to {{HeldWrongPerHi}}% by its own period labels.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/plates/bias-dark.svg">
  <img src="figures/plates/bias-light.svg" width="570" alt="One thin line for each of the {{HeldTraces}} held-out traces through three columns, the share of its scored reads that came back slow, that Heimdall called slow and that its labels marked slow, on an axis from 0 to 100%. A heavy red line joins the medians, lettered {{ReadsSlow}}%, {{CallsSlow}}% and {{PeriodSlow}}%: it rises from the first column to the second and stays nearly level to the third.">
</picture>

**Fig. 3.** Heimdall's calls follow what it was taught, not what the reads did. On the median held-out trace {{ReadsSlow}}% of the scored reads came back slow, Heimdall called {{CallsSlow}}% slow, and its period labels marked {{PeriodSlow}}% slow. It calls slow more often than reads come back slow on {{BiasAbove}} of the {{HeldTraces}} traces. Over all of them its calls sit a median {{BiasGapReads}} points above the reads and within a median {{BiasGapLabels}} points of its labels.

**The score.** A read is slow when its latency exceeds the knee of the replay's read-latency distribution, found by the tangent method of LinnOS and Heimdall, and a guess is wrong when it disagrees. Each estimator, and the table, gives every guess a number, higher for more suspect. Its score on a trace is the AUROC with which it ranks wrong guesses above right ones, in the worst of the four later windows. A window with one class only is left out. The best of nine is picked per trace after the fact, which favours the estimators. Heimdall's period labels are scored beside each read's latency everywhere.

## Traces

**Held out.** Five UMass collections (Financial1, Financial2, WebSearch1 to 3, each storage unit a volume) and MSR Cambridge volumes. Per volume, the three busiest minutes at the recorded pace, and the 24 busiest minutes replayed eight times faster ([scripts/heldout](scripts/heldout)). A cut is kept when it holds at least 3,000 reads: sixty cuts from 39 volumes. Each was replayed once, with no admission control, on a WD_BLACK SN770 1 TB: eight workers in a closed loop as the artefact's replayer runs them, direct I/O on a 16 GiB file of random data ([bench/pair_replayer.c](bench/pair_replayer.c)). A replay counted only if, in every second of it, nothing else used the machine or the drive beyond a criterion fixed from an idle pilot before the first replay ([bench/quietrec.c](bench/quietrec.c)). 166 attempts were rejected and run again. Two cuts are from a volume the table's design had seen and are left out of every summary, and one has no read in its first minute and cannot be scored. That leaves the {{HeldTraces}} from {{HeldVolumes}} volumes ([results/heldout/traces.json](results/heldout/traces.json)).

**Development.** The artefact's twelve device traces are the two sides of its six pairs, from its own baseline replay logs on two NVMe drives. Five further raw traces of the artefact were replayed on a Samsung 990 EVO Plus. Their first third trains and the rest is four equal windows.

## The nine estimators

All see the same twelve inputs as the network, normalised as the network normalises them ([admission/signals.py](admission/signals.py), [admission/compare.py](admission/compare.py)). The first three families are those of Gohil et al., rebuilt from their description.

| estimator | score |
|---|---|
| Euclidean distance | distance of the input from the centroid of the training inputs |
| Mahalanobis distance | the same under the training inputs' covariance |
| conformal, 100 and 1,000 reads | distance-weighted conformal p-value of the label the network did not predict, weights exp(-distance), calibrated on the last 100 or 1,000 reads of the first minute |
| Bayesian network | standard deviation of the predicted probability over 32 samples of a factorised Gaussian posterior (Bayes by backprop) |
| margin | minus the absolute value of the network's logit |
| ensemble of six | standard deviation of the probability across six independently seeded networks |
| deep nearest neighbour | distance to the tenth nearest training point in the last hidden layer |
| trust score | distance to the nearest training point of the predicted class over that of the other class |

- **Given the drive's live state.** Four more variants give the distances and the conformal scores the oldest outstanding request's age, the time since the last completion and the number of requests in flight as extra inputs. Counted among the estimators they change nothing: the table is still ahead of the best of them on 50 of the 57 traces.
- **The distances as Gohil et al. use them.** A distance flags a read beyond the 90th percentile of the training distances: a median {{EuclidFlags}}% of reads for Euclidean and {{MahalFlags}}% for Mahalanobis. A guess the Euclidean one flags is wrong more often than a guess it passes on {{EuclidFlaggedMore}} of the {{HeldTraces}} traces, by a median {{EuclidFlagGap}} points. For Mahalanobis, on {{MahalFlaggedMore}}, by {{MahalFlagGap}} points.
- **What distance does see.** A faint forecast of slowness. On the artefact's traces it leans the right way where Heimdall says fast ({{DevAdmEuclid}}) and the wrong way where Heimdall says slow ({{DevMovEuclid}} and {{DevMovMahal}}). Held out, even that fades ({{AdmEuclid}} and {{AdmMahal}} among admitted reads, {{MovEuclid}} and {{MovMahal}} among moved ones).
- **Period labels.** Judged by them the table is behind the best estimator on {{PerBehind}} of {{HeldTraces}} traces, and the network's own margin scores {{PerMargin}}.

## The table

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/table-dark.svg">
  <img src="figures/table-light.svg" width="100%" alt="Two grids of squares, request size from 512 B up to 64 KiB against the bytes already outstanding on the drive, one while the drive is answering and one after a quiet gap, each square shaded towards red by its entry q, from 0 to 1. The large reads behind a backlog are the reddest, more so after a quiet gap. Square a, 4 KiB with less than 4 KiB outstanding, is barely shaded, and square b, 64 KiB behind 256 KiB after a quiet gap, is deep red. Left, the table's two operations: a read is submitted, look up its cell, and a read completes, update its cell.">
</picture>

**Fig. 4.** The table after the three minutes of one held-out trace, an MSR Cambridge media-server volume, for reads submitted with nothing outstanding for longer than $T$. Sizes and backlogs are classes one power of two wide, named by their lower edge, and blank cells saw fewer than {{TableMinReads}} reads. Slow reads are large reads behind a backlog, more so just after the drive has gone quiet. Heimdall's twelve inputs carry neither the backlog in bytes nor the quiet.

[admission/tracker.py](admission/tracker.py), {{TableLines}} lines, used unchanged for every result.

- A cell is (size class, load class, age class, quiet bit): floor(log2(bytes)) capped at 31, the bit length of the bytes outstanding on the drive divided by 4,096, capped at 15, the age of the oldest outstanding request in units of T, capped at 3, and 1 if the drive has completed something but nothing in the last 16 T.
- Every one of the 4,096 entries starts at 1/2. When a read completes, the entry of the cell it was submitted in moves 1/64 of the way to 1 if its latency exceeded T, and to 0 otherwise. Only completions strictly before a submission count for it.
- T is the knee of the read latencies of the first minute. The score against a fast guess is the entry q, against a slow guess 1 - q.

It was shaped on the artefact's twelve device traces and the five further replays, then frozen, and [check.py](check.py) verifies that [admission/](admission/) is byte for byte the frozen pipeline.

- **Where it was shaped.** On the artefact's twelve traces it scores {{DevTableLo}} to {{DevTableHi}} and the best of the nine {{DevBestLo}} to {{DevBestHi}}.
- **Where it loses.** All {{TableBehind}} held-out traces on which the best of the nine beats the table are UMass Financial volumes. Its threshold can also go stale: on one development replay whose latency drifts upward it scores {{DevWorstRaw}}.
- **One running count.** The table collapsed to a single entry is ahead of each of the nine estimators on {{OneCellAheadLo}} to {{OneCellAheadHi}} of the {{HeldTraces}} traces. With the guess held fixed it is a coin ({{AdmOneCell}} among reads admitted as fast).
- **With the guess held fixed.** [scripts/classwise.py](scripts/classwise.py) scores the reads Heimdall admits as fast (which come back slow?) and the reads it would move (which would have been fast?) separately, so that knowing how often Heimdall calls slow earns nothing. Among admitted reads the table is ahead of each estimator on {{AdmAheadLo}} to {{AdmAheadHi}} traces and of the best of them picked in hindsight on {{AdmAheadBest}}. Among moved reads it scores {{MovTable}} against {{MovEstLo}} to {{MovEstHi}} and is ahead of the hindsight best on {{MovAheadBest}} of {{MovTraces}}. With the read's size fixed as well, it is no better among admitted reads than the network's own margin ({{SameAdmTable}} and {{SameAdmMargin}}). On the artefact's twelve traces it leads the hindsight best on all {{DevAdmAhead}}, both ways ({{DevAdmTable}} and {{DevMovTable}} against {{DevAdmBest}} and {{DevMovBest}}). The script recomputes every score from the stored networks: within 0.003 of the per-window files held out, and on the development traces, where the Bayesian network is resampled on another processor, within 0.12 for it and 0.03 for the rest.
- **A fitted reference.** A classifier fitted on every labelled read of the first minute reaches {{AdmSup}} among admitted reads: on these traces a read's fate is partly unwritten when it is submitted.
- **Two other decisions.** The same protocol with controllers of Heimdall's shape that we trained on each request's own outcome ([admission/transfer.py](admission/transfer.py)). Hedging calls to the four busiest services in ten 3-minute chunks of Alibaba's 2022 microservice traces: the table scores {{RpcTableLo}} to {{RpcTableHi}} against {{RpcMarginLo}} to {{RpcMarginHi}} for the network's own margin. Hedging reads on a storage server with a buffer cache, on the first four shards of 15 January 2024 of three clusters of Google's Thesios traces, whose cache-miss latencies are its own synthesised estimates: {{StoTableLo}} to {{StoTableHi}} against {{StoMarginLo}} to {{StoMarginHi}}. Distance trails there too ({{OtherDistLo}} to {{OtherDistHi}}).

## The table as a forecast

[scripts/direct.py](scripts/direct.py) scores four forecasts that a read will be slow: the table's entry, Heimdall's logit, a network of Heimdall's architecture and recipe trained on the reads submitted and completed inside the first minute, with decision-time inputs and each read's own latency class as the label, and the same network with the bytes outstanding, the oldest outstanding age and the time since the last completion added to its inputs. The table keeps learning. The three networks are frozen after the first minute, as Heimdall is deployed.

- **Against the retrained network.** It scores {{DirNetMed}}: the table is ahead of it on {{DirNetAhead}} of {{HeldTraces}} traces, by a median {{DirNetLead}}. With that live state among its inputs it scores {{DirLiveMed}} and is ahead of the table on {{DirLiveBehind}}.
- **As a call.** Calling slow when q is at least 1/2, the table is wrong on {{WrongTable}}% of held-out reads, Heimdall on {{WrongHeim}}%, the retrained network on {{WrongNet}}% and the one with live state on {{WrongLive}}% (medians over traces). Never calling slow is wrong on {{NeverSlowWrong}}%, so held out the table's call earns next to nothing over that, and Heimdall is wrong more often than never calling slow on {{HeimWorseThanNever}} of the {{HeldTraces}} traces. On the artefact's twelve traces the table is wrong on {{DevWrongTable}}% of reads, never calling slow on {{DevNeverSlowWrong}}% and Heimdall on {{DevWrongHeim}}%.

## Cost

[bench/gate_cost.c](bench/gate_cost.c) times each estimator and the table alone over the 3.68 million later decisions of one device trace, on one pinned core: the table {{TableNsExact}} ns, Euclidean distance {{EuclidNs}} ns, Mahalanobis distance {{MahalNs}} ns, the conformal estimator {{ConfHundredUs}} µs with 100 calibration reads and {{ConfThousandUs}} µs with 1,000. Inside a fixed-point copy of Heimdall's decision path, with its bookkeeping, the table adds {{TableInsituNs}} ns. Gohil et al. report {{StudyBnnUs}} µs for their Bayesian network and about {{StudyConfUs}} µs for their conformal estimator.

## The tail

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/plates/tail-dark.svg">
  <img src="figures/plates/tail-light.svg" width="570" alt="One line for each of the {{TailPairs}} trace pairs across four columns, Euclidean distance, Mahalanobis distance, the table in red, and the oracle, at the pair's p99.9 read latency in ms on a logarithmic axis. Every line is nearly flat: one pair well above the rest, one in the middle, and the others close together near the bottom.">
</picture>

**Fig. 5.** Whoever picks the reads to distrust, the tail barely moves: the p99.9 read latency of the {{TailPairs}} trace pairs on two identical NVMe drives, with hedging on, as the median of {{IdRoundsSix}} rounds. At the median round the better distance ends within {{IdDistWithin}}% of the table on every pair, and an oracle told which guesses were wrong gains at most {{IdOracleGain}}% on the best of the three.

**Policies.** Heimdall alone, Heimdall with a distance (a flagged read goes to both drives) and Heimdall with the table (a read called fast goes to both drives when its entry is at least 1/2, a read called slow stays home when its entry is below 1/2). Each of these with hedging, and hedging alone, with no model. H is the hedging timeout. The first answer is the read's latency, and a drive cannot cancel the losing copy. The oracle is five policies, each with a different action on the wrong guesses and all with hedging: where the paper names one, it is the lowest p99.9 of the five in each round, picked with hindsight.

**Two identical enterprise NVMe drives.** Two Dell Ent NVMe v2 AGN RI U.2 3.84TB, the members of a RAID0 of 512 KiB chunks in a machine with an AMD EPYC 7402P, each client's I/O kept to the chunks of its own member (reading whole chunks of one member's map moved only that member's counters). Nine pairs (P1 to P6 of the artefact, K, S1 and S3) and sixteen policies, in {{IdRounds}} rounds: each round runs every policy once on every pair, in a fresh random order. Per pair, a training replay with no admission control comes first: Heimdall is retrained on its first third, and the distance thresholds, T, H and the oracle's labels come from it too. Every I/O is issued however late, so every policy is scored on the same reads. A run is quiet when no second of it breaks a fixed criterion on other processes' CPU and on foreign traffic to the two drives ([results/routing/enterprise_nvme.json](results/routing/enterprise_nvme.json) holds the thresholds and every kept run). A run that breaks it only through other users' jobs on other cores, with the two drives untouched, is kept and labelled shared. Any other is rejected and run again, up to four attempts. Each ratio is taken within a round and summarised by its median over rounds.

- **The runs.** {{IdKept}} kept, {{IdShared}} of them shared. One is missing: P1's run with no admission control failed the criterion on all four attempts in its fifth round, so P1's ratios to that run rest on four rounds and its comparisons between policies on five. One pair's run with no admission control differs widely between rounds, so ratios to it move with the round, while comparisons between policies are taken within a round.
- **Routing on the table alone.** A read goes to both drives when q is at least 1/2, with no network and no hedging. On the six artefact pairs this has the lower p99.9 than Heimdall and copies {{IdTableOnlyMovedLo}} to {{IdTableOnlyMovedHi}}% of the reads where Heimdall moves {{IdHeimMovedLo}} to {{IdHeimMovedHi}}%. Heimdall keeps the lower mean on {{IdHeimMeanBelow}} of the six, and the lower p99.9 on the three short pairs.
- **The three short pairs.** On K, with hedging, Euclidean distance ends at {{IdKEuclid}} of no admission control and the table at {{IdKTable}}. On S1 and S3, whose p99.9 rests on {{IdSoneTail}} and {{IdSthreeTail}} reads, Heimdall with hedging, checked by a distance, by the table or by nothing, ends at {{IdSGateLo}} to {{IdSGateHi}} and hedging alone at {{IdSoneAlone}} and {{IdSthreeAlone}}.

**Two unlike drives of a quiet machine.** A Samsung SM951 256 GB (M.2, AHCI) and a Crucial MX500 2 TB (SATA). Five pairs (P2, P5 and P6 of the artefact, S1 and S3), fifteen policies, five rounds: 375 runs met the quiet criterion and 161 attempts did not and were run again. The criterion was loosened once, after the first five replays showed the system drive's own log flushes exceeding it, and every run is judged by the loosened one. Five runs of P5 were made up a day or two after their rounds. Every table decision the replayer took agrees with [admission/tracker.py](admission/tracker.py). The SATA drive cannot keep up with the busier client of P2, P5 and P6, so any policy that moves reads off it relieves an overload as well as its slow periods. Without hedging, Heimdall alone, with either distance or with the table leaves the p99.9 of P2 and P5 at {{QuietNoLo}} to {{QuietNoHi}} of no admission control. With hedging every policy on the three artefact pairs reaches {{QuietWithLo}} to {{QuietWithHi}}, and hedging alone {{QuietAloneLo}} to {{QuietAloneHi}}.

**The floor of hedging.** A read that hedging rescues returns no sooner than its client's H plus one read on the other drive. The floor is that H plus the other drive's median read latency, over the p99.9 of the run with no admission control, for the client that makes the tail. Against the SATA drive's {{QuietTailMs}} ms tail it is {{FloorQuietLo}} to {{FloorQuietHi}}. On the identical drives, whose tails end at {{IdTailLoMs}} to {{IdTailHiMs}} ms, it is {{FloorIdLo}} to {{FloorIdHi}}. Hedging alone ends at or above it on every long pair, and it is no bound on S1 and S3.

**What Gohil et al. report.** Their two distances flag {{StudyFlagLo}} and {{StudyFlagHi}}% of requests. Rebuilt from their description, the better one cuts Heimdall's p99.9 by {{QuietCutLo}} to {{QuietCutHi}}% on the unlike pair and by {{SimCutLo}} to {{SimCutHi}}% in a simulator. Every value of theirs and of Heimdall's paper is transcribed from the text ([results/study_reported.json](results/study_reported.json), [results/heimdall_reported.json](results/heimdall_reported.json)).

**Heimdall's own logs.** The artefact ships per-I/O replay logs of no admission control, random routing, Heimdall and hedging for its six pairs ([results/routing/artefact_replays.json](results/routing/artefact_replays.json)). Over the later reads its hedging baseline ends at {{ArtHedgeLo}} to {{ArtHedgeHi}} of the p99.9 of no admission control and Heimdall at {{ArtHeimLo}} to {{ArtHeimHi}}. Heimdall's mean is {{ArtHeimMeanLo}} to {{ArtHeimMeanHi}} and hedging's {{ArtHedgeMeanLo}} to {{ArtHedgeMeanHi}}.

**Two consumer NVMe drives and the simulator.** Two WD_BLACK SN850X 4 TB of a busy machine, one or two runs a pair with no record of what else the machine was doing. And [admission/sim.py](admission/sim.py), which models the drives as processor-sharing servers and reproduces the artefact's replays to within 15% at p50 and p99.9, except Heimdall's own (its p99.9 is 76% and 89% too high on two pairs). Both are in [results/tables.md](results/tables.md), and nothing above rests on either alone.

## Run it

**Every number.** `python check.py` fails if a result file, the summary, the paper's numbers, either page or the tables have drifted, and [tests/test_check.py](tests/test_check.py) shows it failing on a changed result.

**The figures and the paper.** Each figure script reads [results/](results/) and writes into `paper/figures/`, and [figures/table.py](figures/table.py) also writes the two versions of Fig. 4 beside itself. The paper takes its numbers from `paper/numbers.tex`, which [scripts/typeset.py](scripts/typeset.py) writes.

```sh
pip install numpy matplotlib
python figures/heldout.py && python figures/table.py && python figures/tail.py
python figures/plates/make.py      # this page's plates, into figures/plates/
cd paper && pdflatex main && pdflatex main
```

**Your own traces.** All the pipeline needs is a per-I/O log of a drive with no admission control, as a parquet file of four columns: `ts_submit` in milliseconds, `latency` in microseconds, `size` in bytes and `io_type`, 1 for a read. After `pip install -r requirements.txt`, from the repository's root:

```python
import sys; sys.path.insert(0, "scripts")
from admission.protocol import load_trace
from admission.train import train_mlps, train_bnn, save_models
from direct import Replay

D = load_trace(Replay("mine", "mine.parquet", duration_s=180.0))       # its first third trains
save_models("mine.npz", train_mlps(D["train_X"], D["train_y"], seeds=list(range(6))),
            train_bnn(D["train_X"], D["train_y"], seed=0))               # the network, the ensemble, the Bayesian one
```

```sh
python scripts/classwise.py mine --replay mine.parquet --models mine.npz --out mine/classwise
python scripts/direct.py    mine --replay mine.parquet --models mine.npz --out mine/direct
```

The first writes how the table and each estimator rank Heimdall's wrong guesses, window by window, and the second how the table, Heimdall and the retrained networks forecast slow reads. Both take `--duration` for a log that is not 180 s long.

**Our result files.** They were made the same way, from replays that are not in this repository because a replay log carries its trace ([NOTICE.md](NOTICE.md)). [scripts/heldout](scripts/heldout) cuts the held-out traces and records each cut's SHA-256, `bench/pair_replayer --policy 0 --cut 1` replays a cut with `bench/quietrec` recording the machine beside it, and [scripts/pipeline](scripts/pipeline) fetches Heimdall's artefact and scores its traces. The drive measurements need two drives and the same replayer, with the options its header describes. `python check.py --seal` then rewrites the summary, the paper's numbers, the two pages, the tables and the checksums, and [scripts/rebuild.sh](scripts/rebuild.sh) does all of it for the identical drives in one go.

Training is not bit-reproducible across machines, and a drive is not the same drive twice: a second replay gives somewhat different windows. The traces' checksums and every rejected attempt's count are recorded so that a difference can be told from a mistake. The versions the results were made with are pinned in [requirements.txt](requirements.txt), on Python 3.12, and the C sources build with `cc -O2`.
