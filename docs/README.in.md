# Predicting, Fast and Slow: SSD Already Knows Which Guess Is Wrong

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/hero-strips-dark.png">
  <img src="figures/hero-strips-light.png" width="100%" alt="Every read of 57 held-out traces, one row a trace, lit where Heimdall guessed wrong. Ordered by the drive's own table, the wrong guesses gather at the suspect end of each row. Ordered by distance from the training data, they are scattered everywhere.">
</picture>

**Heimdall, a learned controller, calls slow on {{CallsSlow}}% of an SSD's reads when {{ReadsSlow}}% are slow. Uncertainty estimators, the remedy prescribed for its wrong guesses, cannot find them: nine of them do no better than a coin. The drive can.**

On one trace, {{CalmSlow}}% of the 4 KiB reads that found nothing outstanding on the drive came back slow, and Heimdall called {{CalmCalledSlow}}% of them slow. A 4,096-entry table of how the drive's recent reads came back, untrained and {{TableNs}} ns a decision, ranks the wrong guesses at a median AUROC of {{TableMed}} on {{HeldTraces}} held-out traces, where the nine sit at {{EstMedLo}} to {{EstMedHi}}, and it predicts slow reads better than Heimdall itself.

[Read the paper](paper/main.pdf) · [How the table works](#the-drive-already-knows) · [Check every number](#check-it) · [Cite](#cite)

## The estimators flip a coin

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/heldout-dark.svg">
  <img src="figures/heldout-light.svg" width="100%" alt="AUROC for ranking Heimdall's wrong guesses on 57 held-out traces, one dot a trace. Every read: the table has median {{TableMed}}, one running count {{OneCellMed}}, and the nine uncertainty estimators {{EstMedLo}} to {{EstMedHi}}. Reads admitted as fast: the table {{AdmTable}}, one running count {{AdmOneCell}}, the estimators {{AdmEstLo}} to {{AdmEstHi}}.">
</picture>

Heimdall (EuroSys 2025) is a small neural network that guesses, before every read, whether an SSD will serve it fast or slow. Gohil et al. (MLSys 2026) add an uncertainty estimator that scores each guess, and at microsecond budgets they prescribe distance from the training data: "a distance-based estimator must be selected, as it is the only viable option". Whether distance picks out Heimdall's wrong guesses, they did not measure.

None of nine does. Each dot is one held-out trace: Heimdall's own pipeline trains the network on the first minute, and every read of the next two is scored. A guess is wrong when the read comes back on the other side of the knee of the replay's read latencies. An estimator's score is the AUROC with which it ranks wrong guesses above right ones, in the worst of four windows, and a coin scores 0.5. The nine have medians of {{EstMedLo}} to {{EstMedHi}}. On the median trace, where {{WrongHeim}}% of Heimdall's guesses are wrong, a guess that Euclidean distance flags is wrong only {{EuclidFlagGap}} points more often than a guess it passes.

## The wrong guesses are a bias

Doubt is the wrong thing to look for. Heimdall is biased: on the median trace {{SlowCallShare}}% of its wrong guesses are slow calls on reads that come back fast. An estimator that asks how unfamiliar an input is, or how torn the network is, cannot see a bias. One number can: a single running count of how often reads come back slow, blind to the read in front of it, outranks all nine estimators (median {{OneCellMed}}).

The bias is taught. Heimdall trains on period labels, which mark whole stretches of a trace slow: {{PeriodSlow}}% of the scored reads, where {{ReadsSlow}}% come back slow.

None of this is a weak copy of Heimdall. Its artefact reports {{ArtAccLo}} to {{ArtAccHi}}% accuracy on its twelve traces, and our reimplementation of its pipeline reproduces that to within {{ArtAccDiff}} points. That test filters {{FilteredLo}} to {{FilteredHi}}% of a trace's rows as noise, splits the rest at random into training and test halves, and describes each I/O by the three submitted before it, finished or not. Deployed, the network sees the last three I/Os that completed, and a future it was not trained on: on the same twelve traces it is then wrong on {{DevWrongPerLo}} to {{DevWrongPerHi}}% of reads by its own period labels.

## The drive already knows

The table is a notebook the system keeps about its own drive. Every read falls into one of 4,096 situations: how big it is, how many bytes the drive already has outstanding, how long the oldest unfinished request has been waiting, and whether the drive has just been quiet. For each situation the notebook holds one number, q: how often reads in that situation have lately come back slow. When a read finishes, its situation's number moves one sixty-fourth of the way towards 1 if it was slow and towards 0 if it was not. If Heimdall says fast where reads have been coming back slow, that guess is suspect.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/table-dark.svg">
  <img src="figures/table-light.svg" width="100%" alt="The table after three minutes of one held-out trace. For 4 KiB reads with nothing outstanding its entry is {{CalmQ}}: {{CalmSlow}}% of them came back slow, and Heimdall called {{CalmCalledSlow}}% of them slow. For 64 KiB reads behind 256 KiB after a quiet gap its entry is {{StormQ}}: {{StormSlow}}% of them came back slow, and Heimdall called {{StormCalledFast}}% of them fast.">
</picture>

That is the whole mechanism:

```python
# a read is submitted
cell = (size_class, outstanding_bytes_class, oldest_age_class, quiet_bit)
q = table[cell]                                  # how often reads like this one came back slow

# the read completes
table[cell] += ((latency > T) - table[cell]) / 64
```

T is the knee of the first minute's read latencies, so the table needs no training and no label but each read's own latency. The table keeps learning from every completion. The network and the nine estimators are fitted to the first minute and then frozen. None of the machinery is new: processors have assigned confidence to branch predictions this way since 1996. The reference implementation is the {{TableLines}} lines of [admission/tracker.py](admission/tracker.py).

- **It finds the wrong guesses.** Shaped on traces from Heimdall's artefact and frozen before the held-out traces were replayed, it ranks them at a median AUROC of {{TableMed}} and beats the best of the nine, picked per trace with hindsight, on {{TableAhead}} of the {{HeldTraces}} traces. The level is modest: {{TableEighty}} traces reach 0.80. And most of the lead is the bias: among reads Heimdall admits as fast the table scores {{AdmTable}} against {{AdmEstLo}} to {{AdmEstHi}}, level with the best of the nine picked in hindsight.
- **Under Heimdall's period labels it does not.** There the table is a coin ({{PerTable}}) and the best of the nine reaches {{PerEstHi}}: the table learns what reads did, not what the labeller says. The two distances fail under both ({{PerEuclid}} and {{PerMahal}}).
- **It costs what distance costs.** {{TableNsExact}} ns a decision on one core, against {{EuclidNs}} ns for Euclidean distance and {{HeimdallNs}} ns for a fixed-point copy of Heimdall's inference, in 16 KiB.
- **It predicts slow reads better than the network it checks.** Taken as the forecast that a read will be slow, q has a median AUROC of {{DirTableMed}} against {{DirHeimMed}} for Heimdall's output, and is ahead on {{DirHeimAhead}} of the {{HeldTraces}} traces. The fault is not the architecture: trained on each read's own latency, with the inputs it sees when deployed, the same network draws level with the table.

## The tail cannot referee

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/tail-dark.svg">
  <img src="figures/tail-light.svg" width="100%" alt="p99.9 read latency of six trace pairs on two identical NVMe drives, one line a pair, when the guesses to distrust are chosen by Euclidean distance, by Mahalanobis distance, by the table and by an oracle told which were wrong. Every line is close to flat.">
</picture>

The evidence offered for the estimators was a {{StudyCutLo}} to {{StudyCutHi}}% cut in p99.9 latency, and the tail does not care who picks the reads. On two identical enterprise NVMe drives, with hedging on (a read still unanswered after a timeout gets a copy on the other drive), the better of the two distances leaves the p99.9 of every one of the artefact's six trace pairs within {{IdDistWithin}}% of where the table leaves it, at the median round, and even an oracle, told which guesses were wrong in an earlier replay, gains at most {{IdOracleGain}}% on whichever of them does best.

Rebuilt from their description and run without hedging, the better of the two distances cuts Heimdall's p99.9 by {{IdCutLo}} to {{IdCutHi}}% on these drives, pair by pair at the median round. We do not reproduce the reported cut and cannot say which difference of drives, traces or training accounts for the gap.

And part of any cut needs no model. In the replay logs that Heimdall's artefact ships, its own hedging baseline, with no model, has the lower p99.9 on all six pairs, by {{ArtRatioLo}} to {{ArtRatioHi}} times, and Heimdall the lower mean on all six.

## Check it

```sh
python check.py
```

needs only Python 3.10 or later. It recomputes every number the paper and these pages quote from the per-window result files in [results/](results/), and fails if any of them has drifted.

| where | what |
|---|---|
| [paper/](paper/) | the paper: LaTeX source and PDF |
| [docs/further.md](docs/further.md) | every measurement the paper has no room for, each in a sentence |
| [docs/results.md](docs/results.md) | every trace and every drive measurement as tables, including those the table loses |
| [docs/method.md](docs/method.md) | exactly how each was made, and [how to make them again](docs/reproduce.md) |
| [admission/](admission/) | the table, Heimdall's pipeline reimplemented, the nine estimators, the simulator |
| [scripts/](scripts/) | the analyses behind each section, and the one script every number comes from |
| [results/](results/) | per-window results for every trace, every drive measurement, the values the two papers report |
| [figures/](figures/) | every figure with the script that draws it |
| [bench/](bench/) | the replayer, the quiet-machine recorder and the cost benchmark, in C |
| [tests/](tests/) | the table on cases worked by hand, and the check failing on a changed result |

## Cite

```bibtex
@misc{bogdan2026predicting,
  author = {Philipp Bogdan},
  title  = {Predicting, Fast and Slow: {SSD} Already Knows Which Guess Is Wrong},
  year   = {2026}
}
```

[CITATION.cff](CITATION.cff) carries the same.

## Licence and credits

Our code, results, figures and text are released under [CC0 1.0](LICENSE). What belongs to others, and what is deliberately not included, is in [NOTICE.md](NOTICE.md).

- Kurniawan et al. [Heimdall: Optimizing Storage I/O Admission with Extensive Machine Learning Pipeline](https://doi.org/10.1145/3689031.3717496). EuroSys 2025.
- Gohil et al. [When Machine Learning Isn't Sure: Building Resilient ML-Based Computer Systems by Embracing Uncertainty](https://proceedings.mlsys.org/paper_files/paper/2026/hash/96aca14d6c4dcd3adf54bc2c5ad7f138-Abstract-Conference.html). MLSys 2026.
- Dean and Barroso. [The Tail at Scale](https://doi.org/10.1145/2408776.2408794). Communications of the ACM, 2013.
- Jacobsen, Rotenberg and Smith. [Assigning Confidence to Conditional Branch Predictions](https://doi.org/10.1109/MICRO.1996.566457). MICRO 1996.
- UMass Trace Repository. [Storage traces: Financial and WebSearch](https://traces.cs.umass.edu/docs/traces/storage/), from the Storage Performance Council, contributed by Ken Bates and Bruce McNutt.
- Narayanan, Donnelly and Rowstron. [Write Off-Loading: Practical Power Management for Enterprise Storage](https://www.usenix.org/conference/fast-08/write-loading-practical-power-management-enterprise-storage). FAST 2008. The MSR Cambridge traces.
- Luo et al. [The Power of Prediction: Microservice Auto Scaling via Workload Learning](https://doi.org/10.1145/3542929.3563477). SoCC 2022. Alibaba's 2022 microservice traces.
- Phothilimthana et al. [Thesios: Synthesizing Accurate Counterfactual I/O Traces from I/O Samples](https://doi.org/10.1145/3620666.3651337). ASPLOS 2024. Google's Thesios traces, CC BY 4.0.
