"""Fig. 5 of METHOD.md, the tail. One column wide.

The paper's Figure 3 as an ink plate. Six trace pairs of Heimdall's artefact (P1 to P6) on two identical
enterprise NVMe drives, every policy with hedging on. One line a pair. Each mark is the p99.9 read latency
of the policy, pooled over both clients, the median over the pair's whole rounds, when the guesses to
distrust are picked by Euclidean distance (euclid+timeout), Mahalanobis distance (mahal+timeout), the table
(tracker-mixed+timeout) or the oracle (in each round the lowest p99.9 of its five policies,
oracle-*+timeout). The values are those of the paper's own figures/tail.py, function p999_ms(), which reads
results/routing/enterprise_nvme.json (pairs.<P>.rounds.<r>.<policy>.pooled["p99.9"], microseconds) through
scripts/summary.py. This script calls it unchanged.

Medium: one pair. Circles: its p99.9 under each picker. Red: the table's.
"""
import math
import sys
from pathlib import Path

import numpy as np

import plate

ROOT = Path(__file__).resolve().parents[2]
PICKERS = (("Euclidean", "distance"), ("Mahalanobis", "distance"), ("the", "table"), ("oracle", ""))
TABLE = 2


def data():
    sys.path.insert(0, str(ROOT / "figures"))
    import tail as T
    vals = T.p999_ms()
    flat = [v for row in vals.values() for v in row]
    if min(flat) < YLO or max(flat) > YHI:
        raise RuntimeError("a value lies outside the drawn range")
    return vals


W = 237.6                                    # one column, 3.3 in
LEFT, TOP, PW, PH = 27.0, 16.0, 206.0, 150.0
YLO, YHI = 0.55, 2.1


def draw(vals):
    fig = plate.Figure(W, TOP + PH + 32.0)
    ax = fig.axes((LEFT, TOP, PW, PH), x=(-0.36, 3.36), y=(math.log(YLO), math.log(YHI)))
    xs = np.arange(4)
    for pair, row in vals.items():
        ly = np.log(row)
        ax.line(xs, ly, weight="medium")
        ax.mark([x for x in xs if x != TABLE], [v for x, v in zip(xs, ly) if x != TABLE], kind="circle")
        ax.mark([TABLE], [ly[TABLE]], kind="circle", colour="red")
        ax.mark([TABLE], [ly[TABLE]], kind="dot", colour="red")
    ax.frame("box")
    major = [0.6, 0.8, 1.0, 1.5, 2.0]
    minor = [0.7, 0.9, 1.2, 1.4, 1.6, 1.8]
    ax.yticks([math.log(v) for v in major], [math.log(v) for v in minor], labels=[f"{v:g}" for v in major])
    ax.xticks(list(xs), labels=False, top=True)
    ax.ylabel("p99.9 read latency, ms")
    for x, (l1, l2) in zip(xs, PICKERS):
        col = "red" if x == TABLE else "ink"
        X = float(ax.page(x, math.log(YLO))[0])
        fig.text(X, TOP + PH + 5.0, l1, anchor="n", gap=0, colour=col, knockout=False)
        if l2:
            fig.text(X, TOP + PH + 5.0 + 12.0, l2, anchor="n", gap=0, colour=col, knockout=False)
    return fig


def plates():
    return {"tail": draw(data())}
