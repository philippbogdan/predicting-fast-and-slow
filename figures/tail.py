#!/usr/bin/env python3
"""Figure 3: on two identical drives, with hedging in place, the tail lands in the same place whoever picks the
reads to distrust.

Two identical enterprise NVMe drives, the six artefact pairs (results/routing/enterprise_nvme.json), every
picker given the same timeout.  One line a pair; each mark is the picker's own p99.9 read latency, the median
over the pair's whole rounds.  Left to right: Euclidean distance, Mahalanobis distance, the table, and the oracle
(in each round the lowest p99.9 of its five actions).  The one hue marks the table.
"""
import logging
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "figures"))
import style  # noqa: E402
import summary as S  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)     # the paper's font list names fallbacks

PICKERS = [("euclid+timeout", "Euclidean\ndistance"), ("mahal+timeout", "Mahalanobis\ndistance"),
           (S.RULE, "The\ntable"), ("oracle", "Oracle")]
TABLE = 2                                   # the position of the table among the pickers


def p999_ms():
    """{pair: [median over whole rounds of each picker's own p99.9, in ms]}, the pickers in PICKERS' order.
    A picker missing from a round gives no value there; the oracle needs all five of its actions."""
    e = S.load(S.RES / "routing" / "enterprise_nvme.json")
    out = {}
    for pair in S.ARTEFACT_PAIRS:
        rounds = [e["pairs"][pair]["rounds"][r] for r in S.whole_rounds(e, pair)]

        def val(rnd, p):
            x = rnd.get(p)
            return x["pooled"]["p99.9"] / 1000 if x and x["kind"] != "missing" else None

        row = []
        for key, _ in PICKERS:
            keys = S.ORACLES if key == "oracle" else [key]
            per = [[val(rnd, k) for k in keys] for rnd in rounds]
            row.append(st.median(min(v) for v in per if None not in v))
        out[pair] = row
    return out


def draw(theme):
    t = style.use(theme)
    hue = t["outcome"]
    vals = p999_ms()
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    fig.subplots_adjust(left=0.165, right=0.975, top=0.97, bottom=0.2)
    xs = range(len(PICKERS))
    for row in vals.values():
        ax.plot(xs, row, color=t["input"], lw=1.0, zorder=2, solid_capstyle="round")
        ax.scatter([x for x in xs if x != TABLE], [v for x, v in zip(xs, row) if x != TABLE], s=22,
                   color=t["input"], edgecolor=t["bg"], lw=0.8, zorder=3, clip_on=False)
        ax.scatter([TABLE], [row[TABLE]], s=30, color=hue, edgecolor=t["bg"], lw=0.8, zorder=4, clip_on=False)

    ax.set_yscale("log")
    flat = [v for row in vals.values() for v in row]
    lo, hi = min(0.5, min(flat) / 1.15), max(2.0, max(flat) * 1.15)        # always shows 0.5 to 2 ms
    ax.set_ylim(lo, hi)
    ticks = [v for v in (0.2, 0.3, 0.5, 1, 2, 3, 5, 10) if lo <= v <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks], fontsize=8.5)
    ax.minorticks_off()
    ax.set_ylabel("p99.9 read latency, ms", fontsize=8.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab in PICKERS], fontsize=8.5, linespacing=1.0)
    for k, lab in enumerate(ax.get_xticklabels()):
        lab.set_color(hue if k == TABLE else t["fg"])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.35, len(PICKERS) - 0.65)
    ax.spines["bottom"].set_visible(False)
    style.save(fig, "tail", theme, t)
    plt.close(fig)


if __name__ == "__main__":
    for theme in ("paper", "light", "dark"):
        draw(theme)
