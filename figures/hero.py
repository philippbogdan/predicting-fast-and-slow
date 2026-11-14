#!/usr/bin/env python3
"""The picture at the top of the README (results/heldout/hero, from scripts/hero_data.py), light and dark.

strips  Every later read of all 57 held-out traces, one row a trace.  Each row is the trace's reads put
        in order, most suspect first, and cut into 240 equal runs.  A run is red by the share of
        Heimdall's guesses in it that were wrong.  Left: ordered by the table.  Right: ordered by
        Euclidean distance from the training data.  A signal that knows which guess is wrong reddens the
        left end of its rows and leaves the right end pale.
rain    A second picture, drawn only when asked for.  Two minutes of one drive: every later read of one
        held-out trace by submission time and latency.  Left: red where the table said slow.  Right: red
        where the Euclidean gate flagged the read.  Reads above the line came back slow.

Usage: hero.py [strips] [rain]   (strips by default)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "figures"))
import style  # noqa: E402
import summary as S  # noqa: E402

HERO = S.RES / "heldout" / "hero"


def mix(other, w):
    """The figures' one hue at weight w, the rest the colour other: its tints and shades for the ramps."""
    from matplotlib.colors import to_hex, to_rgb
    return to_hex([w * a + (1 - w) * b for a, b in zip(to_rgb(style.RED), to_rgb(other))])


RAIN_TRACE = "msr.mds_0.orig"
LOOK = {
    "light": {"bg": "#ffffff", "ink": "#1f2328", "muted": "#59636e",
              "ramp": ["#f1efec", mix("#ffffff", 0.25), mix("#ffffff", 0.55), style.RED, mix("#000000", 0.62)],
              "dust": "#8c959f", "line": "#1f2328"},
    "dark": {"bg": "#0d1117", "ink": "#f0f6fc", "muted": "#9198a1",
             "ramp": ["#141a21", mix("#0d1117", 0.2), mix("#0d1117", 0.55), style.RED, mix("#ffffff", 0.3)],
             "dust": "#7d8590", "line": "#f0f6fc"},
}
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def strips(theme):
    L = LOOK[theme]
    _, _, _, scored = S.heldout_records()
    names = sorted(scored, key=lambda n: -S.worst(scored[n], "tracker"))
    Z = {n: np.load(HERO / f"{n}.npz") for n in names}
    cmap = LinearSegmentedColormap.from_list("lit", L["ramp"])
    plt.rcParams.update({"font.family": FONT})
    fig = plt.figure(figsize=(12, 5.6), dpi=200, facecolor=L["bg"])
    for k, (key, title) in enumerate((("strip_table", "ordered by the drive's own table"),
                                      ("strip_euclid", "ordered by distance from the training data"))):
        ax = fig.add_axes([0.035 + k * 0.49, 0.125, 0.445, 0.775])
        img = np.array([Z[n][key] for n in names])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for y in range(1, len(names)):                    # hairlines between traces
            ax.axhline(y - 0.5, color=L["bg"], lw=1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0, -1.6, title, fontsize=13, color=L["ink"], ha="left", va="bottom")
        ax.text(0, len(names) + 0.4, "most suspect", fontsize=10.5, color=L["muted"], ha="left", va="top")
        ax.text(239.5, len(names) + 0.4, "least suspect", fontsize=10.5, color=L["muted"], ha="right", va="top")
    fig.text(0.0195, 0.51, "57 traces, every read", fontsize=10.5, color=L["muted"], rotation=90, ha="center",
             va="center")
    # key: how lit is how wrong
    kax = fig.add_axes([0.465, 0.038, 0.07, 0.018])
    kax.imshow(np.linspace(0, 1, 200)[None, :], cmap=cmap, aspect="auto")
    kax.set_xticks([])
    kax.set_yticks([])
    for sp in kax.spines.values():
        sp.set_visible(False)
    fig.text(0.46, 0.047, "guesses right", fontsize=10.5, color=L["muted"], ha="right", va="center")
    fig.text(0.54, 0.047, "guesses wrong", fontsize=10.5, color=L["muted"], ha="left", va="center")
    fig.savefig(Path(__file__).resolve().parent / f"hero-strips-{theme}.png", facecolor=L["bg"],
                metadata={"Software": None})
    plt.close(fig)


def soften(a, sigma=0.9):
    """Spread each grid cell over its neighbours (a small Gaussian), so that single reads show."""
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def glow(count, lit, L):
    """An image of the reads: brightness from how many reads fell near a cell, colour from the share that is lit."""
    from matplotlib.colors import to_rgb
    count, lit = soften(count), soften(lit)
    dens = np.clip(np.log1p(4 * count) / np.log1p(4 * np.percentile(count[count > 0.05], 98)), 0, 1)
    share = np.clip(np.divide(lit, count, out=np.zeros_like(lit), where=count > 1e-9), 0, 1)
    bg, dust, hot = (np.array(to_rgb(c)) for c in (L["bg"], L["dust"], L["ramp"][3]))
    colour = dust[None, None, :] * (1 - share[..., None]) + hot[None, None, :] * share[..., None]
    alpha = (dens * (0.5 + 0.5 * share ** 0.7))[..., None]
    return bg[None, None, :] * (1 - alpha) + colour * alpha


def rain(theme):
    L = LOOK[theme]
    z = np.load(HERO / f"{RAIN_TRACE}.npz")
    count = z["rain_reads"].astype(float)
    lo, hi = z["rain_latency_us"]
    knee = float(z["knee_us"])
    plt.rcParams.update({"font.family": FONT})
    fig = plt.figure(figsize=(12, 5.2), dpi=200, facecolor=L["bg"])
    for k, (key, title) in enumerate((("rain_table_says_slow", "lit: reads the drive's own table called slow"),
                                      ("rain_gate_flags", "lit: reads the distance gate flagged"))):
        ax = fig.add_axes([0.05 + k * 0.485, 0.09, 0.455, 0.78])
        ax.imshow(glow(count, z[key].astype(float), L), origin="lower", aspect="auto", interpolation="bilinear",
                  extent=(0, 1, np.log(lo), np.log(hi)))
        ax.axhline(np.log(knee), color=L["line"], lw=0.8, alpha=0.75)
        ax.set_xticks([])
        ax.set_yticks([np.log(v) for v in (50, 100, 200, 500, 1000)] if k == 0 else [])
        ax.set_yticklabels(["50 µs", "100", "200", "500", "1 ms"] if k == 0 else [], fontsize=10, color=L["muted"])
        ax.tick_params(length=0, pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0, np.log(hi) + 0.06, title, fontsize=13, color=L["ink"], ha="left", va="bottom")
        if k == 0:
            ax.text(0.995, np.log(knee) + 0.04, "slow above this line", fontsize=10.5, color=L["ink"], ha="right",
                    va="bottom")
        ax.text(0, np.log(lo) - 0.07, "two minutes of one drive, every read", fontsize=10.5, color=L["muted"],
                ha="left", va="top") if k == 0 else None
    fig.savefig(Path(__file__).resolve().parent / f"hero-rain-{theme}.png", facecolor=L["bg"],
                metadata={"Software": None})
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["strips"]
    for theme in ("light", "dark"):
        for w in which:
            {"strips": strips, "rain": rain}[w](theme)
