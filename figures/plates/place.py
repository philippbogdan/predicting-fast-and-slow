"""Where a label can go without covering a mark: a few lines on top of plate.py.

Obstacles are kept in page points (from the top left): marks as discs, lines as polylines. A label's
box is tested against all of them, with a margin, before it is accepted.
"""
from __future__ import annotations

import numpy as np

import plate


class Room:
    def __init__(self):
        self.discs = []       # (x, y, r)
        self.lines = []       # arrays of (x, y)
        self.boxes = []       # (l, t, r, b)

    def disc(self, view, x, y, r=2.2):
        X, Y = view.page(np.atleast_1d(x), np.atleast_1d(y))
        for a, b in zip(np.ravel(X), np.ravel(Y)):
            self.discs.append((float(a), float(b), r))

    def line(self, view, x, y):
        X, Y = view.page(np.asarray(x, float), np.asarray(y, float))
        self.lines.append(np.column_stack([np.ravel(X), np.ravel(Y)]))

    def box(self, b):
        self.boxes.append(tuple(b))

    def clear(self, b, margin=1.2):
        l, t, r, bt = b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin
        for x, y, rad in self.discs:
            cx, cy = min(max(x, l), r), min(max(y, t), bt)
            if (cx - x) ** 2 + (cy - y) ** 2 < rad * rad:
                return False
        for p in self.lines:
            for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
                if _seg_hits(x0, y0, x1, y1, l, t, r, bt):
                    return False
        for q in self.boxes:
            if not (q[2] < l or q[0] > r or q[3] < t or q[1] > bt):
                return False
        return True


def _seg_hits(x0, y0, x1, y1, l, t, r, b):
    """Does the segment cross the rectangle? (Liang and Barsky.)"""
    dx, dy = x1 - x0, y1 - y0
    u0, u1 = 0.0, 1.0
    for p, q in ((-dx, x0 - l), (dx, r - x0), (-dy, y0 - t), (dy, b - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            u = q / p
            if p < 0:
                u0 = max(u0, u)
            else:
                u1 = min(u1, u)
            if u0 > u1:
                return False
    return True


def box_at(X, Y, s, anchor, size="normal", gap=None):
    """The ink box a label would have, as Figure._letter places it (page points)."""
    items, (bx0, by0, bx1, by1), _ = plate._layout(s)
    k = plate.SIZES[size] / 1000
    if gap is None:
        gap = 0.0 if anchor == "c" else plate._GAP
    l, r, t, b = bx0 * k, bx1 * k, -by1 * k, -by0 * k
    hx = l if "w" in anchor else r if "e" in anchor else 0.5 * (l + r)
    vy = t if anchor[0] == "n" else b if anchor[0] == "s" else 0.5 * (t + b)
    dx = gap if "w" in anchor else -gap if "e" in anchor else 0.0
    dy = gap if anchor[0] == "n" else -gap if anchor[0] == "s" else 0.0
    ox, oy = X + dx - hx, Y + dy - vy
    return (ox + l, oy + t, ox + r, oy + b)


def inside(b, frame, margin=1.5):
    return b[0] >= frame[0] + margin and b[1] >= frame[1] + margin and b[2] <= frame[2] - margin \
        and b[3] <= frame[3] - margin
