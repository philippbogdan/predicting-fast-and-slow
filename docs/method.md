# Method

How every number was made. The tables are in [results.md](results.md), the measurements the paper has no room for in [further.md](further.md), and how to make them again in [reproduce.md](reproduce.md).

## The decision

Heimdall puts a 12-128-16-1 network in front of each SSD of a replicated pair. Before a read is submitted to its drive, the network sees the read's size, the client's queue length there, and the queue length, latency and throughput of the last three I/Os that completed there, and guesses fast or slow. A slow guess sends the read to the other drive. Gohil et al. add an uncertainty estimator that scores each guess, and send a distrusted read to both drives at once.

We ask three things: what picks out the wrong guesses, whether the table needs the network at all, and what cuts the tail.

## Protocol

**Windows.** A replay lasts 180 s. Its first 60 s train everything: the network, every estimator's fitted state, the thresholds, the timeout. Every read of the remaining 120 s is scored, in four windows of 30 s. (The five further development replays have other lengths: their first third trains and the rest is four equal windows.) Heimdall's own long-term evaluation deploys a model trained once on the first minutes of a trace in the same way.

**Heimdall.** [admission/heimdall.py](../admission/heimdall.py) reimplements the artefact's period labelling, noise filtering, training features and training recipe. On the twelve device traces of the artefact it reproduces the artefact's labels and features row for row and its reported accuracy to within 0.5 points ([results/development/pipeline](../results/development/pipeline)). One trap is fixed on the way: the artefact orders I/Os with an unstable sort, which NumPy runs differently on processors with AVX-512, so the module carries a port of NumPy's generic sort. At decision time the network is fed what the artefact's replayer feeds it: the last three I/Os that completed, not the last three submitted.

**Which guesses are wrong.** A read is slow when its own latency exceeds the knee of the replay's read-latency distribution, found by the tangent method LinnOS and Heimdall use. A guess is wrong when it disagrees. Heimdall's period labels, which mark whole stretches of a trace slow, are scored beside this everywhere and reported as such.

**The score.** Each estimator, and the table, gives every guess a number, higher for more suspect. Its score on a trace is the AUROC with which it ranks wrong guesses above right ones, in the worst of the four later windows. A window with one class only is left out. The best of nine is the best of the nine estimators on that trace, picked after the fact, which favours the estimators.

## The nine estimators

All see the same twelve inputs as the network, normalised as the network normalises them ([admission/signals.py](../admission/signals.py), [admission/compare.py](../admission/compare.py)).

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

The first three families are those of Gohil et al., rebuilt from their description. Used as they use it, a distance flags a read when it exceeds the 90th percentile of the training distances. Four more variants give the distances and the conformal scores the drive's live state as extra inputs. Counted among the estimators they change nothing: the table is still ahead of the best of them on 50 of the 57 held-out traces.

## The table

[admission/tracker.py](../admission/tracker.py), 143 lines, used unchanged for every result.

- A cell is (size class, load class, age class, quiet bit): floor(log2(bytes)) capped at 31, the bit length of the bytes outstanding on the drive divided by 4,096, capped at 15, the age of the oldest outstanding request in units of T, capped at 3, and 1 if the drive has completed something but nothing in the last 16 T.
- Every one of the 4,096 entries starts at 1/2. When a read completes, the entry of the cell it was submitted in moves 1/64 of the way to 1 if its latency exceeded T, and to 0 otherwise. Only completions strictly before a submission count for it.
- T is the knee of the read latencies of the first minute. The score against a fast guess is the entry q, against a slow guess 1 - q.

It was shaped on the artefact's twelve device traces and five further replays, then frozen, and [check.py](../check.py) verifies that [admission/](../admission/) is byte for byte the frozen pipeline.

## With the guess held fixed

Heimdall calls slow far more often than reads are slow, so most of its wrong guesses are slow calls on reads that come back fast, and a score that knows the base rate ranks them first without telling one slow call from another. [scripts/classwise.py](../scripts/classwise.py) removes that effect: it recomputes every estimator and the table from the stored networks and scores, per window, the reads Heimdall admits as fast (which come back slow?) and the reads it would move (which would have been fast?) separately. It also scores:

- two controls for the direction alone: the guess itself as the score, and the table collapsed to one running count of slow reads,
- each of the above again on reads of one size only, the trace's commonest size class,
- the two distances as Gohil et al. use them: how often a flagged guess is wrong, and an unflagged one,
- Euclidean distance on unnormalised inputs, in case Gohil et al. meant that.

Its every-read scores reproduce the per-window files of [results/heldout/detection](../results/heldout/detection) to within 0.003. On the development traces the Bayesian network is resampled on another processor and moves by up to 0.12 in a window, and the other estimators by at most 0.03. The table and the distances are deterministic and identical.

## The table as a forecast

[scripts/direct.py](../scripts/direct.py) scores four forecasts that a read will be slow: the table's entry, Heimdall's logit, a network of Heimdall's architecture and recipe trained on the reads submitted and completed inside the first minute, with decision-time inputs and each read's own latency class as the label, and the same network with the bytes outstanding, the oldest outstanding age and the time since the last completion added to its inputs. The table keeps learning. The three networks are frozen after the first minute, as Heimdall is deployed. A forecast's call is slow when the entry is at least 1/2, or the logit is positive.

## Traces

**Held out.** Five UMass collections (Financial1, Financial2, WebSearch1 to 3, each storage unit a volume) and MSR Cambridge volumes. Per volume, the three busiest minutes at the recorded pace, and the 24 busiest minutes replayed eight times faster, so that both last 180 s ([scripts/heldout](../scripts/heldout)). A cut is kept when it holds at least 3,000 reads. Sixty cuts from 39 volumes qualify. Each was replayed once, with no admission control, on a WD_BLACK SN770 1 TB: eight workers in a closed loop as the artefact's replayer runs them, direct I/O on a 16 GiB file of random data ([bench/pair_replayer.c](../bench/pair_replayer.c)). A replay counted only if, in every second of it, nothing else used the machine or the drive beyond a criterion fixed from an idle pilot before the first replay ([bench/quietrec.c](../bench/quietrec.c), 166 attempts were rejected and run again). Two cuts are from a volume the table's design had seen and are left out of every summary, and one has no read in its first minute and cannot be scored. That leaves 57 ([results/heldout/traces.json](../results/heldout/traces.json)).

**Development.** The twelve device traces are the two sides of the six pairs in the artefact's client-level dataset, from its own baseline replay logs on two NVMe drives. Five further raw traces of the artefact were replayed on a Samsung 990 EVO Plus.

## Routing

**Policies.** Heimdall alone. Heimdall with a distance (a flagged read goes to both drives). Heimdall with the table (a read Heimdall calls fast goes to both drives when its entry is at least 1/2, a read Heimdall calls slow stays home when its entry is below 1/2). Each of these with hedging, and hedging alone, with no model. Routing on the table alone, with no network and no hedging, sends a read to both drives when its entry is at least 1/2. Hedging sends a copy to the other drive when a read sent to one drive is still unanswered after a timeout H, the 98th percentile of the first minute's read latencies. The first answer is the read's latency. A drive cannot cancel the losing copy. The drive files also hold five oracle policies, which distrust exactly the wrong guesses, each with a different action and all with hedging. A guess is wrong for the oracle when it disagrees with the read's latency class in an earlier replay of the pair with no admission control. Where the paper names one oracle, it is the lowest p99.9 of the five in each round, picked with hindsight.

**The floor of hedging.** A read of one client still unanswered after that client's H is copied to the other drive, so with hedging alone it returns no sooner than H plus one read there. Per round, the client that makes the tail is the one with more of its reads above the pooled p99.9 of the run with no admission control, and the floor is its H plus the other drive's median read latency in that run, over the pooled p99.9. [further.md](further.md) quotes the median over rounds. The capped p99.9 beside it caps every read of that run at its own client's H plus the other drive's median.

**Heimdall's own logs.** The artefact ships per-I/O replay logs of no admission control, random routing, Heimdall and hedging for its six pairs ([results/routing/artefact_replays.json](../results/routing/artefact_replays.json) holds their read-latency statistics over the later reads).

**Two identical enterprise NVMe drives of one machine.** Two Dell Ent NVMe v2 AGN RI U.2 3.84TB (firmware 2.3.0, PCIe 4.0 x4), the two members of a RAID0 of 512 KiB chunks in a machine with an AMD EPYC 7402P. One file of random data lies across both, and each client's I/O is kept to the file's chunks on its own member ([bench/pair_replayer.c](../bench/pair_replayer.c), `--map0` and `--map1`). Before measuring, whole chunks of each map were read and only that member's counters moved. All nine pairs (P1 to P6, K, S1 and S3) and sixteen policies, no admission control and every policy above, in rounds: each round runs every policy once on every pair, the policies of a pair in a fresh random order, with at least 10 s of idle drives between replays. Per pair, a training replay with no admission control comes first: Heimdall is retrained on its first third, and the distance thresholds, T, the timeout H and the oracle's labels come from it too. The scored reads are those of the later two thirds of the pair's duration, and every I/O is issued however late, so every policy is scored on the same reads. A recorder samples the machine every second. A run is quiet when, in every second of it, no other process used more than 60 ticks of CPU, CPU that no process accounts for stayed within 60 ticks over every 5 s, other users' processes within 5 ticks, nobody was seated, and each drive's traffic other than the replay's own stayed within 16 MiB over every 5 s and over the whole replay. Other users' jobs could run on other cores of the machine: a run that fails the criterion only on their processes, with the two drives untouched and our own and the system's processes within the limit, is kept and labelled shared. Any other run is rejected and run again, up to four attempts in all. A round counts for a pair when every policy of the pair has a kept run in it or failed four attempts. Each ratio is taken within a round, against that round's run with no admission control, and summarised by its median over the rounds counted. [results/routing/enterprise_nvme.json](../results/routing/enterprise_nvme.json) holds every kept run's read-latency statistics, recomputed from its per-read latencies by [scripts/enterprise_nvme.py](../scripts/enterprise_nvme.py), with its label and the attempts rejected before it.

**Two unlike drives of a quiet machine.** A Samsung SM951 256 GB (M.2, AHCI) and a Crucial MX500 2 TB (SATA) in one machine that ran nothing else, recorded every second. Five pairs whose busier client asks for at most 1 GB/s (P2, P5, P6 of the artefact, and two small samples, S1 and S3), fifteen policies, five rounds, each round in a fresh random order; 375 runs met the quiet criterion and 161 attempts did not and were run again. The criterion was fixed from an idle pilot and loosened once, after the first five replays showed the system drive's own log flushes exceeding it; every run is judged by the loosened one. Five runs of P5 were made up one to two days after their rounds. The SATA drive cannot keep up with the busier client of P2, P5 and P6, so any policy that moves reads off it relieves an overload as well as its slow periods. Every table decision the replayer took agrees with [admission/tracker.py](../admission/tracker.py).

**Two consumer NVMe drives of a busy machine.** Two WD_BLACK SN850X 4 TB. One run a pair for P1 to P4 (P4 cut short), two for K, S1 and S3, with no record of what else the machine was doing.

**The simulator.** [admission/sim.py](../admission/sim.py) replays two clients against two drives modelled as processor-sharing servers, with each I/O's service work taken from the artefact's baseline replay. It reproduces the artefact's replays of no admission control, random routing and hedging to within 15% at p50 and p99.9, but Heimdall's own replay less well (its p99.9 is 76% and 89% too high on two pairs). [further.md](further.md) quotes it once, beside drive measurements.

**What the two papers report.** The values in [results/study_reported.json](../results/study_reported.json) and [results/heimdall_reported.json](../results/heimdall_reported.json) are transcribed from their text. The latencies read off Figure 10 of Gohil et al. are estimates and marked as such.

## Two other decisions

The same protocol and estimators on two other learned decisions ([admission/transfer.py](../admission/transfer.py)), each with a controller of Heimdall's shape that we trained on each request's own outcome (latency above the knee of the first third): hedging calls to the four busiest downstream services in ten 3-minute chunks of Alibaba's 2022 microservice traces, and hedging reads on a storage server with a buffer cache, on the first four shards of 15 January 2024 of three clusters of Google's Thesios I/O traces. Thesios gives the latency of cache-miss reads as its own synthesised estimate. Alibaba's response times are whole milliseconds; where the knee falls on the 1 ms floor, a call is slow above the smallest recorded value that fewer than half of the calls exceed. These controllers are ours, not published systems, and the paper quotes the result only as the limit of its claim.

## Cost

[bench/gate_cost.c](../bench/gate_cost.c) times each estimator and the table alone over the 3.68 million later decisions of one device trace on one pinned core of an Intel Core i9-13900KF, and the table again inside a fixed-point copy of Heimdall's decision path ([scripts/pipeline/bench_cost.py](../scripts/pipeline/bench_cost.py) builds its input).
