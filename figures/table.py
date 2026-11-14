#!/usr/bin/env python3
"""Figure 2: how the table works, on what one held-out replay taught it.

Left: the table's two operations, on two of its cells.  Right: its entries at the end of the replay of
msr.mds_0.orig (results/heldout/table), for reads submitted with no request older than T outstanding:
the slow-read frequency q by request size and bytes already outstanding, while the drive is answering
and after it has completed nothing for 16 T.  Every number on the figure is read from that file.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "figures"))
import style  # noqa: E402
import summary as S  # noqa: E402

TRACE = "msr.mds_0.orig"
SIZES = list(range(9, 17))            # 512 B to 64 KiB
LOADS = list(range(0, 9))             # nothing outstanding to 512 KiB and more
SIZE_LABEL = {9: "512 B", 10: "1 KiB", 11: "2 KiB", 12: "4 KiB", 13: "8 KiB", 14: "16 KiB", 15: "32 KiB", 16: "64 KiB"}
LOAD_LABEL = ["0", "", "8K", "", "32K", "", "128K", "", "512K"]
MIN_READS = 20
CALM = (12, 0, 0)                     # 4 KiB, nothing outstanding, drive answering
STORM = (16, 7, 1)                    # 64 KiB, 256 KiB or more outstanding, after a quiet gap
X0, X1, Y0, W, H = 0.468, 0.728, 0.2, 0.235, 0.66       # the two heat maps, in figure coordinates


def draw():
    t = style.use()
    d = S.load(S.RES / "heldout" / "table" / f"{TRACE}.json")
    cells = {(c["size_class"], c["load_class"], c["quiet"]): c for c in d["cells"] if c["age_class"] == 0}
    cmap = LinearSegmentedColormap.from_list("q", [t["bg"], t["outcome"]])
    fig = plt.figure(figsize=(7.0, 2.6))
    axes = [fig.add_axes([X0, Y0, W, H]), fig.add_axes([X1, Y0, W, H])]
    for z, (ax, title) in enumerate(zip(axes, ["drive answering", "after a quiet gap"])):
        for i, s in enumerate(SIZES):
            for j, b in enumerate(LOADS):
                c = cells.get((s, b, z))
                if c is None or c["reads"] < MIN_READS:
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none", edgecolor=t["grid"], lw=0.5))
                else:
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cmap(c["q"]), edgecolor=t["bg"], lw=0.6))
        ax.set_xlim(0, len(LOADS))
        ax.set_ylim(0, len(SIZES))
        ax.set_xticks([j + 0.5 for j in range(len(LOADS))])
        ax.set_xticklabels(LOAD_LABEL, fontsize=7)
        ax.set_yticks([i + 0.5 for i in range(len(SIZES))])
        ax.set_yticklabels([SIZE_LABEL[s] for s in SIZES] if z == 0 else [], fontsize=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=8.5, color=t["fg"], pad=4)
    fig.text(X0 - 0.062, Y0 + H / 2, "request size", ha="center", va="center", rotation=90, fontsize=8,
             color=t["fg"])
    fig.text((X0 + X1 + W) / 2, 0.05, "bytes already outstanding on the drive", ha="center", va="center",
             fontsize=8, color=t["fg"])
    # the two cells the text is about
    for key, ax, tag, ink in ((CALM, axes[0], "a", t["fg"]), (STORM, axes[1], "b", "white")):
        s, b, z = key
        i, j = SIZES.index(s), LOADS.index(b)
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none", edgecolor=t["fg"], lw=1.1, zorder=5))
        ax.text(j + 0.5, i + 0.47, tag, ha="center", va="center", fontsize=8, color=ink, fontweight="bold",
                zorder=6)
    # colour key
    cax = fig.add_axes([X1 + W + 0.008, Y0, 0.008, H])
    for k in range(100):
        cax.add_patch(Rectangle((0, k / 100), 1, 0.011, facecolor=cmap(k / 99), lw=0))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([0, 1])
    cax.set_yticklabels(["0", "1"], fontsize=7)
    cax.tick_params(length=0, pad=1.5)
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.text(X1 + W + 0.012, Y0 + H + 0.045, "q", ha="center", va="center", fontsize=8.5, color=t["fg"],
             style="italic")

    # left: the two operations
    ax = fig.add_axes([0.0, 0.0, 0.385, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def say(y, text, x=0.03, **kw):
        kw.setdefault("fontsize", 8)
        kw.setdefault("color", t["fg"])
        ax.text(x, y, text, va="center", **kw)

    def tag(y, letter):
        ax.add_patch(Rectangle((0.03, y - 0.036), 0.052, 0.072, facecolor="none", edgecolor=t["fg"], lw=0.9))
        ax.text(0.056, y - 0.004, letter, ha="center", va="center", fontsize=8, color=t["fg"], fontweight="bold")

    calm, storm = cells[CALM], cells[STORM]
    say(0.945, "A read is submitted: look up its cell", fontweight="bold", fontsize=8.5)
    tag(0.84, "a")
    say(0.84, "4 KiB, nothing outstanding", x=0.115)
    ax.text(0.99, 0.735, f"q = {calm['q']:.2f}", fontsize=9, va="center", ha="right", color=t["outcome"],
            fontweight="bold")
    say(0.77, f"{100 * calm['slow'] / calm['reads']:.0f}% of these came back slow.", x=0.115)
    say(0.70, f"Heimdall called {100 * calm['heimdall_called_slow'] / calm['reads']:.0f}% of them slow.", x=0.115)
    tag(0.585, "b")
    say(0.585, "64 KiB behind 256 KiB, after a quiet gap", x=0.115)
    say(0.515, f"{100 * storm['slow'] / storm['reads']:.0f}% of these came back slow.", x=0.115)
    say(0.445, f"Heimdall called {100 * (1 - storm['heimdall_called_slow'] / storm['reads']):.0f}% of them fast.",
        x=0.115)
    ax.text(0.99, 0.48, f"q = {storm['q']:.2f}", fontsize=9, va="center", ha="right", color=t["outcome"],
            fontweight="bold")
    say(0.305, "A read completes: update its cell", fontweight="bold", fontsize=8.5)
    say(0.205, "Slower than T: 1/64 of the way to 1.")
    say(0.135, "Faster: 1/64 of the way to 0.")
    style.save(fig, "table")
    plt.close(fig)


if __name__ == "__main__":
    draw()
