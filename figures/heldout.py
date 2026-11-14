#!/usr/bin/env python3
"""Figure 1: which signal picks out Heimdall's wrong guesses on the 57 held-out traces.

One row per signal, one dot per trace: the signal's worst later-window AUROC.
Left: every read (results/heldout/detection; the one running count from results/classwise).
Right: only the reads Heimdall admits as fast (results/classwise), where the direction of the guess
no longer helps: the one running count falls back to a coin.
"""
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

ROWS = [("tracker", "The table, 4,096 entries"), ("one_cell", "One running count"),
        ("euclid", "Euclidean distance"), ("mahal", "Mahalanobis distance"),
        ("conformal100", "Conformal, 100 reads"), ("conformal1000", "Conformal, 1,000 reads"),
        ("bnn", "Bayesian network"), ("margin", "Margin"), ("ensemble", "Ensemble of six"),
        ("knn", "Deep nearest neighbour"), ("trust", "Trust score")]
OUTCOME = ("tracker", "one_cell")


def swarm(xs, r, step):
    """Deterministic beeswarm: offsets so that dots closer than r in x do not overlap."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    placed, ys = [], [0.0] * len(xs)
    for i in order:
        k = 0
        while True:
            y = ((k + 1) // 2) * (1 if k % 2 else -1) * step
            if all(abs(xs[i] - px) >= r or abs(y - py) >= step for px, py in placed):
                break
            k += 1
        ys[i] = y
        placed.append((xs[i], y))
    return ys


def values(part):
    """Worst later-window AUROC of every row's signal on every scored held-out trace."""
    _, _, _, detection = S.heldout_records()
    recs = S.classwise_records("heldout")
    out = {}
    for key, _ in ROWS:
        v = []
        for n in sorted(recs):
            if part == "pooled" and key != "one_cell":        # the detection files, as the text quotes them
                a = S.worst(detection[n], key)
            else:
                w = [x["own_latency"]["signals"][key][part] for x in S.later(recs[n])]
                w = [a for a in w if S.finite(a)]
                a = min(w) if w else float("nan")
            if S.finite(a):
                v.append(a)
        out[key] = v
    return out


def panel(ax, t, vals, labels, lo):
    ax.axvspan(lo, 0.5, color=t["grid"], lw=0, zorder=0)
    ax.axvline(0.5, color=t["muted"], lw=0.7, zorder=1)
    n = len(ROWS)
    for i, (key, label) in enumerate(ROWS):
        y0 = n - 1 - i - (0.35 if key not in OUTCOME else 0)
        xs = vals[key]
        col = t["outcome"] if key in OUTCOME else t["input"]
        ys = swarm(xs, 0.0105 * (1.0 - lo) / 0.9, 0.095)
        ax.scatter(xs, [y0 + y for y in ys], s=6.5, color=col, lw=0, zorder=3,
                   alpha=1.0 if key == "tracker" else 0.85)
        med = sorted(xs)[len(xs) // 2]
        ax.plot([med, med], [y0 - 0.36, y0 + 0.36], color=t["fg"], lw=1.3, zorder=4, solid_capstyle="butt")
        ax.text(1.0 + 0.013 * (1.0 - lo) / 0.9, y0, f"{med:.2f}", va="center", ha="left", fontsize=8.5,
                color=t["outcome"] if key in OUTCOME else t["fg"])
        if labels:
            ax.text(lo - 0.012, y0, label, va="center", ha="right", fontsize=8.5,
                    color=t["outcome"] if key in OUTCOME else t["fg"])
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(-0.95, n - 0.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.2", "0.4", "0.5", "0.6", "0.8", "1.0"], fontsize=8)


def draw():
    t = style.use()
    fig = plt.figure(figsize=(7.0, 3.05))
    a = fig.add_axes([0.232, 0.185, 0.325, 0.72])
    b = fig.add_axes([0.632, 0.185, 0.325, 0.72])
    panel(a, t, values("pooled"), True, 0.1)
    panel(b, t, values("called_fast"), False, 0.1)
    a.set_title("every read: is the guess wrong?", fontsize=8.5, color=t["fg"], pad=7)
    b.set_title("reads admitted as fast: which come back slow?", fontsize=8.5, color=t["fg"], pad=7)
    for ax in (a, b):
        ax.text(0.5, -0.105, "coin", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5,
                color=t["muted"])
    fig.text(0.595, 0.035, "AUROC in the worst of four later windows. Each dot is one held-out trace.", ha="center",
             va="center", fontsize=8.5, color=t["fg"])
    style.save(fig, "heldout")
    plt.close(fig)


if __name__ == "__main__":
    draw()
