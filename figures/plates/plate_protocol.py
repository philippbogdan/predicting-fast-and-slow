"""Fig. 2 of METHOD.md, the protocol. One column wide. A schematic, drawn without data.

One time axis of a 180 s replay, as admission/protocol.py cuts it (window_edges: the training window
[0, 60) s, then [60, 90), [90, 120), [120, 150), [150, 180] s) and as METHOD.md describes it: Heimdall's
network (trained by its own pipeline on the first minute, its period labels) and the nine estimators are
fitted on the first minute and then frozen, T is the knee of the first minute's read latencies
(protocol.load_trace: T_read_w0), the table (admission/tracker.py) starts every entry at 1/2 and moves an
entry at every completion, from the start of the replay to its end, and every read of the last two minutes
is scored, an AUROC in each window, the worst of the four being the trace's score.

Hatched: fitted to the first minute. Dashed: frozen. Stippled: scored. Red: the table, learning throughout.
"""
import plate

W = 237.6                                   # one column, 3.3 in
X0, X1 = 7.0, W - 8.0                       # 0 s and 180 s
BAR = 12.0                                  # height of a bar
NAME = 13.0                                 # from a lane's name to its bar
LANE = 35.0                                # from lane to lane


def draw():
    lanes = [14.0 + NAME + i * LANE for i in range(4)]        # top of each bar
    axis_y = lanes[-1] + BAR + 12.0
    fig = plate.Figure(W, axis_y + 30.0)
    t = fig.view((X0, 0, X1 - X0, fig.height), x=(0, 180), y=(fig.height, 0))     # x: seconds, y: points down

    def name(y, s, at=0.0, colour="ink"):
        t.text(at, y - 4.0, s, anchor="sw", gap=0, colour=colour)

    # the end of the first minute, through every lane
    t.line([60, 60], [lanes[0] - NAME - 2.0, axis_y], weight="fine", dash="dashdot")

    # 1. the network and the nine estimators: fitted, then frozen
    y = lanes[0]
    name(y, "Heimdall's network and the nine estimators")
    t.fill([0, 60, 60, 0], [y, y, y + BAR, y + BAR], tone="hatch")
    t.line([0, 60, 60, 0], [y, y, y + BAR, y + BAR], weight="medium", closed=True)
    t.line([60, 180], [y + BAR / 2] * 2, weight="medium", dash="dashed")
    t.text(30, y + BAR / 2, "fitted", anchor="c")
    t.text(120, y + BAR / 2, "frozen", anchor="c")

    # 2. T, from the first minute's read latencies
    y = lanes[1]
    name(y, "$T$, the knee of these read latencies")
    t.line([0, 0, 60, 60], [y + BAR, y + BAR / 2, y + BAR / 2, y + BAR], weight="medium")
    t.line([60, 180], [y + BAR / 2] * 2, weight="fine", dash="dotted")
    t.text(120, y + BAR / 2, "fixed", anchor="c")

    # 3. the table: every entry from 1/2, moved at every completion, to the end
    y = lanes[2]
    name(y, "the table, every entry from 1/2")
    t.line([0, 180], [y + BAR / 2] * 2, weight="heavy", colour="red", arrow="end")
    t.text(120, y + BAR / 2, "learns from every completion", anchor="c")

    # 4. every later read scored, in four windows
    y = lanes[3]
    name(y, "every later read scored, the worst window counts")
    for a in (60, 90, 120, 150):
        b = a + 30
        t.fill([a, b, b, a], [y, y, y + BAR, y + BAR], tone="stipple")
        t.line([a, b, b, a], [y, y, y + BAR, y + BAR], weight="medium", closed=True)
    t.text(30, y + BAR / 2, "not scored", anchor="c")

    # the time axis
    t.line([0, 180], [axis_y, axis_y], weight="medium")
    for s in range(0, 181, 30):
        t.line([s, s], [axis_y, axis_y + (3.5 if s % 60 == 0 else 2.0)], weight="medium")
        t.text(s, axis_y + 5.0, f"{s}", anchor="n", gap=0, size="small", knockout=False)
    t.text(90, axis_y + 16.0, "seconds into the replay", anchor="n", gap=0, knockout=False)
    return fig


def plates():
    return {"protocol": draw()}
