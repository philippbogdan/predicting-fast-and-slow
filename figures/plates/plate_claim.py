"""Fig. 1 of METHOD.md, the claim: all the held-out traces, each one thin line across the tenths of its reads,
sorted by how suspect a signal finds Heimdall's guess. One column wide.

The values are scripts/plate_data.py's claim(): every trace's later reads, sorted most suspect first and cut
into 240 runs of equal size by scripts/hero_data.py (results/heldout/hero/<trace>.npz, strip_table and
strip_euclid), summed into tenths of the reads, 24 runs each. A tenth's height is how much more often its
guesses are wrong than the trace's guesses as a whole, in percentage points: the share wrong in the tenth
minus the share wrong over all the trace's later reads. Zero is the trace's own average, so traces wrong on
7% and on 86% of their reads share one axis, and a signal that knows nothing leaves every tenth near zero.

Thin: one trace, its ten tenths joined by straight lines, each line partly transparent (ALPHA: 0.35 on the
light page, 0.30 on the dark page, where light ink spreads to the eye), so that the ink builds up where traces
run together and a trace that runs alone stays faint. Red, heavy: the median of the traces in each tenth,
joined the same way, solid. Dotted: zero, each trace's own average. The frame, the dotted line and the
lettering are solid.
"""
import numpy as np

import plate
import plate_data as D

PANELS = (("table", "the drive's own table"), ("euclid", "distance from the training data"))
TENTHS = D.TENTHS
W, LEFT, RIGHT = 237.6, 39.0, 5.0                # one column of the paper, 3.3 in
WIDE = W - LEFT - RIGHT
Y = (-75.0, 75.0)                                # points from the trace's average
ALPHA = {"light": 0.35, "dark": 0.30}            # the trace lines' opacity, judged by eye at printed size


def data():
    """{signal: array (traces, tenths)}: points by which each tenth of a trace's reads, most suspect
    first, is wrong more often than the trace as a whole."""
    m = {key: np.array(v) for key, v in D.claim().items()}
    for key, v in m.items():
        if v.min() < Y[0] or v.max() > Y[1]:
            raise RuntimeError(f"{key}: {v.min():.1f} to {v.max():.1f} points does not fit the axis {Y}")
    return m


def draw(M, alpha=ALPHA, weight="fine"):
    strip, gap, top = 84.0, 12.0, 17.0
    fig = plate.Figure(W, top + 2 * strip + gap + 17.0)
    xc = np.arange(TENTHS) + 0.5
    for i, (key, name) in enumerate(PANELS):
        box = (LEFT, top + i * (strip + gap), WIDE, strip)
        ax = fig.axes(box, x=(0, TENTHS), y=Y)
        m = M[key]
        ax.line([0, TENTHS], [0, 0], weight="fine", dash="dotted")
        for row in m:
            ax.line(xc, row, weight=weight, alpha=alpha)
        ax.line(xc, np.median(m, axis=0), weight="heavy", colour="red", halo=True)
        ax.frame("box")
        ax.yticks([-40, 0, 40], [-60, -20, 20, 60], labels=["−40", "average", "+40"])
        if i == 0:
            # what a thin line is: lettered under the fan, clear of every line it spans, with a fine
            # leader straight up to the lowest line, which it touches without crossing another
            pts = (Y[1] - Y[0]) / strip                          # points of the axis to a printed point
            label = "one line a trace"
            xt, xl = 1.0, 0.85
            lw, lh = plate.measure(label)
            span = np.linspace(xl, xl + lw / WIDE * TENTHS, 60)
            top_y = min(min(np.interp(x, xc, row) for row in m) for x in span) - 4.0 * pts
            if top_y - (lh + 2.0) * pts < Y[0]:
                raise RuntimeError("no paper under the fan for its label")
            hit = min(np.interp(xt, xc, row) for row in m)
            ax.line([xt, xt], [hit, top_y + 1.5 * pts], weight="fine")
            ax.text(xl, top_y, label, anchor="nw", gap=0, knockout=False)
        ax.xticks([], list(range(1, TENTHS)), labels=False)
        # the signal's name, at the top, over paper: the place nearest the right where no line reaches it
        w, h = plate.measure(name)
        half = 0.5 * w / WIDE * TENTHS
        floor = Y[1] - (h + 5.0) / strip * (Y[1] - Y[0])
        edge = 4.5 / WIDE * TENTHS                       # 4.5 pt clear of the frame's side
        for x in np.linspace(TENTHS - edge - half, edge + half, 61):
            lo, hi = x - half, x + half
            seg = np.linspace(lo, hi, 50)
            reach = max(np.interp(seg, xc, row).max() for row in m)
            if reach < floor:
                break
        else:
            raise RuntimeError(f"{key}: no paper at the top for its name")
        ax.text(x, Y[1] - 3.0 / strip * (Y[1] - Y[0]), name, anchor="n", gap=0, knockout=False)
        if i == 1:
            base = box[1] + strip + 3.5 + plate._NUMERAL * plate.SIZES["small"]
            fig.text(LEFT, base, "most suspect", anchor="w", gap=0, size="small", baseline=True, knockout=False)
            fig.text(LEFT + WIDE, base, "least suspect", anchor="e", gap=0, size="small", baseline=True,
                     knockout=False)
        else:
            fig.text(LEFT, box[1] - 4.5, "share of guesses wrong, in points", anchor="sw", gap=0, knockout=False)
    return fig


def plates():
    return {"claim": draw(data())}
