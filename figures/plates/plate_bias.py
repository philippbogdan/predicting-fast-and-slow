"""Fig. 3 of METHOD.md, the bias. One column wide.

For each held-out trace, three shares of the reads scored (the four later windows), from
scripts/plate_data.py's bias(): the share that came back slow (by each read's own latency), the share
Heimdall called slow, and the share Heimdall's period labels mark slow, all from
results/classwise/heldout/<trace>.json as summary.classwise_block sums them. Their medians are
summary.json's classwise.heldout.reads_slow_median, .heimdall_calls_slow_median and
classwise.heldout_period_labels.reads_slow_median, and data() checks that they agree.

Thin, partly transparent (as the claim plate's trace lines): one trace, through its three shares.
Red, heavy: the medians of the three shares.
"""
import json
import statistics as st

import numpy as np

import place
import plate
import plate_data as D

ROWS = (("slow", "came back slow"), ("calls", "Heimdall called slow"), ("labels", "period labels marked slow"))


def data():
    out = D.bias()
    med = {k: st.median(v[k] for v in out.values()) for k, _ in ROWS}
    summ = json.loads((D.S.RES / "summary.json").read_text())["classwise"]
    want = {"slow": summ["heldout"]["reads_slow_median"], "calls": summ["heldout"]["heimdall_calls_slow_median"],
            "labels": summ["heldout_period_labels"]["reads_slow_median"]}
    for k in med:
        if abs(med[k] - want[k]) > 1e-12:
            raise RuntimeError(f"median {k}: {med[k]} here, {want[k]} in summary.json")
    return out, med


W = 237.6                                          # one column, 3.3 in
LEFT, TOP, PW, PH = 34.0, 17.0, 198.0, 168.0
XL = (-0.42, 2.42)
ALPHA = {"light": 0.35, "dark": 0.30}             # as the claim plate's trace lines
COLUMNS = (("slow", "came back", "slow"), ("calls", "Heimdall", "called slow"), ("labels", "its labels", "marked slow"))


def draw(traces, med):
    fig = plate.Figure(W, TOP + PH + 32.0)
    ax = fig.axes((LEFT, TOP, PW, PH), x=XL, y=(0, 1))
    frame = (LEFT, TOP, LEFT + PW, TOP + PH)
    room = place.Room()
    xs = [0, 1, 2]
    for n in sorted(traces):
        ys = [traces[n][k] for k, _, _ in COLUMNS]
        ax.line(xs, ys, weight="fine", alpha=ALPHA)
        room.line(ax, xs, ys)
    my = [med[k] for k, _, _ in COLUMNS]
    ax.line(xs, my, weight="heavy", colour="red", halo=True)
    room.line(ax, xs, my)
    # the medians, lettered in red beside their points: outside the fan where it can be
    for x, y, anchors in ((0, my[0], ("e", "se", "ne")), (1, my[1], ("s", "n", "sw", "nw", "se", "ne")),
                          (2, my[2], ("w", "nw", "sw"))):
        X, Y = (float(v) for v in ax.page(x, y))
        s = f"{100 * y:.0f}%"
        for g in (3.0, 4.5, 6.0):
            done = False
            for a in anchors:
                b = place.box_at(X, Y, s, a, gap=g)
                if place.inside(b, frame) and room.clear(b, margin=0.6):
                    ax.text(x, y, s, anchor=a, gap=g, colour="red")
                    room.box(b)
                    done = True
                    break
            if done:
                break
        else:
            ax.text(x, y, s, anchor=anchors[0], gap=3.0, colour="red")     # over the lines, knocked out
    # what a thin line is: lettered over paper at the top left, a fine leader down to the highest line
    label, xt = "one line a trace", 0.12
    lw, lh = plate.measure(label)
    pts = 1.0 / PH                                         # axis units to a printed point
    rows = [[traces[n][k] for k, _, _ in COLUMNS] for n in traces]
    for ytop in np.arange(0.985, 0.6, -0.01):
        X, Y = (float(v) for v in ax.page(XL[0], ytop))
        b = place.box_at(X + 4.0, Y, label, "nw", gap=0)
        if room.clear(b, margin=1.5):
            break
    else:
        raise RuntimeError("no paper for the label")
    hit = max(np.interp(xt, xs, r) for r in rows)
    fig.text(b[0], b[1], label, anchor="nw", gap=0, knockout=False)
    ax.line([xt, xt], [hit + 1.0 * pts, ytop - (lh + 2.0) * pts], weight="fine")
    ax.frame("box")
    t = [0, 0.25, 0.5, 0.75, 1]
    ax.yticks(t, [0.125, 0.375, 0.625, 0.875], labels=["0", "25%", "50%", "75%", "100%"])
    ax.xticks(xs, labels=False)
    ax.ylabel("share of the scored reads")
    for x, (_, l1, l2) in zip(xs, COLUMNS):
        X = float(ax.page(x, 0)[0])
        fig.text(X, TOP + PH + 5.0, l1, anchor="n", gap=0, knockout=False)
        fig.text(X, TOP + PH + 17.0, l2, anchor="n", gap=0, knockout=False)
    return fig


def plates():
    return {"bias": draw(*data())}
