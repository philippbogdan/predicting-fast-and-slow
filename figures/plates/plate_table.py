"""Fig. 4 of METHOD.md, the table. Full width.

Right: the table's entries at the end of one held-out replay, msr.mds_0.orig, from scripts/plate_data.py's
table() (results/heldout/table/msr.mds_0.orig.json, written by scripts/table_state.py), for reads submitted
with no request outstanding for longer than T (age_class 0): one cell per size class (512 B to 64 KiB) and
load class (bytes outstanding, 0 to 512 KiB), while the drive is answering (quiet 0) and after it has
completed nothing for 16 T (quiet 1). Each cell is hatched up to its entry q ("q"), the level drawn exactly.
Cells with fewer than plate_data.MIN_READS reads ("reads") are left blank, as in the paper's Figure 2. Every
visited age-0 cell with that many reads or more lies inside the drawn ranges, and data() checks it.

The two cells the lettering works through are summary.json's example_table.small_read_idle_drive (size 12,
load 0, quiet 0) and .large_read_backlog_after_quiet (size 16, load 7, quiet 1): their shares are
"slow" / "reads" and "heimdall_called_slow" / "reads" of the same records, checked against summary.json.

Left: the two operations, on a magnified cell drawn without data (its level is a schematic q).
"""
import json

import plate
import plate_data as D

TRACE = D.TABLE_TRACE
SIZES = list(range(9, 17))
LOADS = list(range(0, 9))
SIZE_LABEL = {9: "512 B", 10: "1 KiB", 11: "2 KiB", 12: "4 KiB", 13: "8 KiB", 14: "16 KiB", 15: "32 KiB",
              16: "64 KiB"}
LOAD_LABEL = {0: "0", 2: "8K", 4: "32K", 6: "128K", 8: "512K"}
MIN_READS = D.MIN_READS
A, B = (12, 0, 0), (16, 7, 1)              # (size, load, quiet)


def data():
    d, cells = D.table()
    for (s, b, z), c in cells.items():
        if c["reads"] >= MIN_READS and (s not in SIZES or b not in LOADS):
            raise RuntimeError(f"cell {(s, b, z)} has {c['reads']} reads and is outside the drawing")
    ex = json.loads((D.S.RES / "summary.json").read_text())["example_table"]
    for key, name in ((A, "small_read_idle_drive"), (B, "large_read_backlog_after_quiet")):
        c, e = cells[key], ex[name]
        if not (c["q"] == e["q"] and c["reads"] == e["reads"] and c["slow"] / c["reads"] == e["slow_share"]
                and c["heimdall_called_slow"] / c["reads"] == e["heimdall_called_slow_share"]):
            raise RuntimeError(f"cell {key} disagrees with summary.json's {name}")
    return d, cells


W = 496.8                                   # the paper's text width, 6.9 in
CW, CH = 13.8, 18.0                         # a cell, as printed
GRID_TOP = 18.0
GAP = 12.0
GRID_B_LEFT = W - 12.0 - 9 * CW
GRID_A_LEFT = GRID_B_LEFT - GAP - 9 * CW
H = GRID_TOP + 8 * CH + 34.0
TEXT_RIGHT = GRID_A_LEFT - 34.0             # the left block ends here, clear of the size labels
LINE = 12.5
LEVEL = "medium"
INSET = 1.3                                 # paper between neighbouring cells, points


def pct(a, b):
    return f"{100 * a / b:.0f}%"


def vessel(v, x0, x1, y0, y1, q):
    """A cell hatched up to its entry q, its level a line."""
    yq = y0 + q * (y1 - y0)
    if q > 0.02:                             # below this the hatch would be a sliver under the level
        v.fill([x0, x1, x1, x0], [y0, y0, yq, yq], tone="hatch")
    v.line([x0, x1, x1, x0], [y0, y0, y1, y1], weight="fine", closed=True)
    v.line([x0, x1], [yq, yq], weight=LEVEL)


def grid(fig, left, z, cells, title):
    v = fig.view((left, GRID_TOP, 9 * CW, 8 * CH), x=(0, 9), y=(0, 8))
    for i, s in enumerate(SIZES):
        for j, b in enumerate(LOADS):
            c = cells.get((s, b, z))
            if c is not None and c["reads"] >= MIN_READS:
                vessel(v, j + INSET / CW, j + 1 - INSET / CW, i + INSET / CH, i + 1 - INSET / CH, c["q"])
    v.text(4.5, 8, title, anchor="s", gap=5.0, knockout=False)
    for j, lab in LOAD_LABEL.items():
        v.text(j + 0.5, 0, lab, anchor="n", gap=4.0, size="small", knockout=False)
    return v


def mark(v, key, letter):
    s, b, z = key
    j, i = LOADS.index(b), SIZES.index(s)
    v.line([j, j + 1, j + 1, j], [i, i, i + 1, i + 1], weight="heavy", colour="red", closed=True)
    v.text(j + 0.5, i + 0.5, letter, anchor="c", colour="red")


def draw(cells):
    fig = plate.Figure(W, H)
    va = grid(fig, GRID_A_LEFT, 0, cells, "drive answering")
    vb = grid(fig, GRID_B_LEFT, 1, cells, "after a quiet gap")
    for i, s in enumerate(SIZES):
        va.text(0, i + 0.5, SIZE_LABEL[s], anchor="e", gap=4.0, size="small", knockout=False)
    fig.text(0.5 * (GRID_A_LEFT + GRID_B_LEFT + 9 * CW), GRID_TOP + 8 * CH + 18.0, "bytes already outstanding on the drive",
             anchor="n", gap=0, knockout=False)
    mark(va, A, "a")
    mark(vb, B, "b")

    # left: a read is submitted, and the two cells the text works through
    a, b = cells[A], cells[B]
    y = GRID_TOP - 12.0
    fig.text(0, y, "A read is submitted: look up its cell.", anchor="nw", gap=0, knockout=False)
    y += 20.0
    for c, letter, words in (
            (a, "a", ["4 KiB, less than 4 KiB outstanding:",
                      f"{pct(a['slow'], a['reads'])} came back slow, and Heimdall",
                      f"called {pct(a['heimdall_called_slow'], a['reads'])} of them slow."]),
            (b, "b", ["64 KiB behind 256 KiB, after a quiet gap:",
                      f"{pct(b['slow'], b['reads'])} came back slow, and Heimdall",
                      f"called {pct(b['reads'] - b['heimdall_called_slow'], b['reads'])} of them fast."])):
        fig.text(0, y, letter, anchor="nw", gap=0, colour="red", knockout=False)
        for k, w in enumerate(words):
            fig.text(9.0, y + k * LINE, w, anchor="nw", gap=0, knockout=False)
        fig.text(TEXT_RIGHT, y + 2 * LINE, f"$q$ = {c['q']:.2f}", anchor="ne", gap=0, colour="red", knockout=False)
        y += 3 * LINE + 8.0

    # a read completes: its cell's level moves, shown on one cell drawn large, without data
    y += 3.0
    fig.text(0, y, "A read completes: update its cell.", anchor="nw", gap=0, knockout=False)
    top, height, width, left = y + 19.0, 50.0, 22.0, 12.0
    g = fig.view((left, top, width, height), x=(0, 1), y=(0, 1))
    level = 0.45
    vessel(g, 0, 1, 0, 1, level)
    g.text(0, 0, "0", anchor="e", gap=3.0, size="small", knockout=False)
    g.text(0, 1, "1", anchor="e", gap=3.0, size="small", knockout=False)
    g.text(1, level, "$q$", anchor="w", gap=3.0)
    ax = left + width + 18.0
    fig.arrow(ax, top + (1 - level) * height - 2.0, ax, top + 2.0, weight="medium")
    fig.arrow(ax, top + (1 - level) * height + 2.0, ax, top + height - 2.0, weight="medium")
    fig.line([ax - 4.0, ax + 4.0], [top + (1 - level) * height] * 2, weight="fine")
    tx = ax + 7.0
    fig.text(tx, top, "slower than $T$:", anchor="nw", gap=0, knockout=False)
    fig.text(tx, top + LINE, "1/64 of the way to 1", anchor="nw", gap=0, knockout=False)
    fig.text(tx, top + height - LINE, "faster:", anchor="sw", gap=0, knockout=False)
    fig.text(tx, top + height, "1/64 of the way to 0", anchor="sw", gap=0, knockout=False)
    return fig


def plates():
    return {"table": draw(data()[1])}
