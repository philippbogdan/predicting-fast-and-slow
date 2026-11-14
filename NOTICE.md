# What belongs to others

Everything in this repository is released under [CC0 1.0](LICENSE), except as stated here.

## Not included

- **Heimdall's artefact** ([github.com/ucare-uchicago/Heimdall](https://github.com/ucare-uchicago/Heimdall) and its client-level dataset) carries no licence. None of its code, traces, replay logs or trained weights is here. [admission/heimdall.py](admission/heimdall.py) is our own reimplementation of its published pipeline, checked against the artefact's files; the result files hold per-window statistics derived from its traces, not the traces.
- **The UMass storage traces** (Financial1, Financial2, WebSearch1 to 3; [traces.cs.umass.edu](https://traces.cs.umass.edu/docs/traces/storage/)) are made available courtesy of Ken Bates, Bruce McNutt and the Storage Performance Council. None is redistributed here.
- **The MSR Cambridge traces** (Narayanan, Donnelly and Rowstron, FAST 2008) were read from a copy converted to MQSim's format ([github.com/Tina0113/trace](https://github.com/Tina0113/trace)), which states neither how it was converted nor a licence. None is redistributed here.
- **Alibaba's microservice traces** ([cluster-trace-microservices-v2022](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022)) state no licence and ask to be used "for study or research purposes"; only per-window statistics derived from them are here ([results/other_decisions/rpc_hedging](results/other_decisions/rpc_hedging)).
- **Google's Thesios I/O traces** ([github.com/google-research-datasets/thesios](https://github.com/google-research-datasets/thesios), Phothilimthana et al., ASPLOS 2024) are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the statistics derived from them in [results/other_decisions/storage_server_hedging](results/other_decisions/storage_server_hedging) credit them here and in the [README](README.md#licence-and-credits).
- **Our replays** of those traces, and the networks trained on them, are not included either: a replay log carries its trace. What is here is derived from them: per-window scores and counts, latency percentiles, the entries of one learnt table, and the binned counts behind the two pictures in [results/heldout/hero](results/heldout/hero).

## Included

The functions `_aheapsort` and `generic_argsort` in [admission/heimdall.py](admission/heimdall.py) port NumPy's generic argsort, so that I/Os with equal timestamps are ordered as they were when the artefact's files were made. NumPy's licence:

```
Copyright (c) 2005-2026, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The values in [results/study_reported.json](results/study_reported.json) are quoted from Gohil et al., MLSys 2026, for comparison.
