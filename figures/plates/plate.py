r"""Ink plates: the drawing style of torux's method page, lettered in the paper's own face, with one red.

The figures of "Predicting, Fast and Slow" are drawn with this module and nothing else, so that all of
them look inked by one hand for one book: black on white (light ink on GitHub's dark page), a few
weights of line, dashes, hatching and stipple, upright lettering in XCharter, the face the paper is
set in, and exactly one colour beside ink and paper, red #E73121, for the one thing a figure is about.
It is torux's method/figures/style.py (CC0) with three changes: the unit is the printer's point at
the size the figure is printed, the face is XCharter instead of Computer Modern, and there is red.

Running
-------
Needs numpy, matplotlib (for its path tools) and fontTools (which matplotlib brings), and the faces
XCharter-Roman.otf and XCharter-Italic.otf: in fonts/ beside this file, in a directory named by the
environment variable PLATE_FONTS, or wherever kpsewhich finds them (TeX Live's xcharter package).
rsvg-convert (Homebrew librsvg) writes the PDF and the previews.

    fig = plate.Figure(width=237.6, height=160)          # points, as printed: 3.3 in is one column
    ax = fig.axes(box=(40, 20, 190, 100), x=(0, 1), y=(0, 1))
    ax.fill(xs, ys, tone="solid", colour="red")
    ax.line([0, 1], [0.4, 0.4], weight="fine", dash="dotted")
    ax.text(0.98, 0.95, "the drive's own table", anchor="ne")
    fig.save("name", out_dir)

save() writes name-light.svg and name-dark.svg for GitHub, at px pixels to the point, and name.pdf
for the paper, at its printed size, and refuses a file with anything in it but ink, paper and red.

The vocabulary
--------------
    weight  fine        hatching, leaders, rules, a family of many curves
            medium      the ordinary line: outlines, frames, curves (the default)
            heavy       the one thing the figure is about, if red does not already carry it
    dash    solid, dashed (hidden, a level), dotted (a reference), dashdot (an axis), dashdotdot
    tone    solid       small shapes, and the red regions
            stipple     light: a region to notice      hatch   middle: a region that matters
            paper       blank, hides what is under it
    colour  ink         everything (the default)
            red         the one thing the figure is about, and nothing else
    alpha   a number    a family of many thin ink lines, each partly transparent, so that the ink
                        builds up where they run together. Never on red,
                        frames, references or lettering, which stay solid.
    size    normal      every label (10 pt)            small   numbers on axes (9 pt)

No size is smaller than the paper's caption type, 9 pt. Words are upright and $...$ is set in
italic, for symbols. No legend, no title, no note at the bottom: a label sits where its thing is.
"""

from __future__ import annotations

import glob
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from matplotlib.path import Path as _MPath

HERE = Path(__file__).resolve().parent
RED = "#E73121"
WEIGHTS = ("fine", "medium", "heavy")
DASHES = ("solid", "dashed", "dotted", "dashdot", "dashdotdot")
TONES = ("solid", "stipple", "hatch", "paper")
COLOURS = ("ink", "red")
MARKS = ("dot", "circle", "cross")
SIZES = {"normal": 10.0, "small": 9.0}
ANCHORS = ("c", "n", "s", "e", "w", "ne", "nw", "se", "sw")

# What differs between the pages. Light ink on a dark page spreads to the eye, so the dark page has
# finer values, as torux's has. Lengths are points at the printed size.
PAGES = {
    "light": {"name": "light", "ink": "#000000", "paper": "#ffffff", "red": RED,
              "weight": {"fine": 0.45, "medium": 0.75, "heavy": 1.35},
              "hatch": {"pitch": 4.05, "width": 0.44}, "stipple": {"spacing": 2.65, "dot": 0.84},
              "dot": 0.8},
    "dark": {"name": "dark", "ink": "#f0f6fc", "paper": "#0d1117", "red": RED,
             "weight": {"fine": 0.42, "medium": 0.66, "heavy": 1.15},
             "hatch": {"pitch": 4.05, "width": 0.36}, "stipple": {"spacing": 2.85, "dot": 0.72},
             "dot": 0.72},
}
_DASH = {"dashed": (6.6, 3.1), "dotted": (0.0, 2.4), "dashdot": (10.6, 2.6, 0.0, 2.6),
         "dashdotdot": (10.6, 2.4, 0.0, 2.4, 0.0, 2.4)}
_DASH_SCALE = {"fine": 0.85, "medium": 1.0, "heavy": 1.3}
_ARROW = {"fine": (4.4, 1.2), "medium": (5.5, 1.5), "heavy": (6.8, 1.9)}     # length, half width
_TICK = {"major": 3.5, "minor": 2.0}
_GAP = 2.2        # between a point and the label that names it
_KNOCK = 1.3      # paper kept clear round a label
_HALO = 1.5       # paper cut either side of a haloed line
_MARK = {"dot": 1.45, "circle": 1.9, "cross": 2.4, "end": 0.9}
_NUMERAL = 0.68   # height of an XCharter lining numeral, in ems


def _choose(value, choices, what):
    if value not in choices:
        raise ValueError(f"{what} must be one of {tuple(choices)}, not {value!r}")
    return value


# ------------------------------------------------------------------------------- the face


def _find_font(name):
    places = [HERE / "fonts" / name]
    if os.environ.get("PLATE_FONTS"):
        places.insert(0, Path(os.environ["PLATE_FONTS"]) / name)
    for p in places:
        if p.exists():
            return p
    if shutil.which("kpsewhich"):
        r = subprocess.run(["kpsewhich", name], capture_output=True, text=True)
        if r.stdout.strip():
            return Path(r.stdout.strip())
    found = glob.glob(str(Path.home() / "Library/Caches/Tectonic/bundles/data/*" / name))
    if found:
        return Path(found[0])
    raise FileNotFoundError(f"{name} not found: put it in {HERE / 'fonts'} or name its directory in PLATE_FONTS")


def _gpos_kerning(tt):
    """The pair adjustments of the font's 'kern' feature: a dict of glyph pairs and the class tables."""
    pairs, classes = {}, []
    if "GPOS" not in tt:
        return pairs, classes
    g = tt["GPOS"].table
    idx = sorted({i for fr in g.FeatureList.FeatureRecord if fr.FeatureTag == "kern" for i in fr.Feature.LookupListIndex})
    for li in idx:
        lk = g.LookupList.Lookup[li]
        for st in lk.SubTable:
            if lk.LookupType == 9:
                st = st.ExtSubTable
            if getattr(st, "LookupType", 2) != 2 and lk.LookupType not in (2, 9):
                continue
            if st.Format == 1:
                for i, g1 in enumerate(st.Coverage.glyphs):
                    for pv in st.PairSet[i].PairValueRecord:
                        x = getattr(pv.Value1, "XAdvance", 0) if pv.Value1 is not None else 0
                        pairs.setdefault((g1, pv.SecondGlyph), x or 0)
            elif st.Format == 2:
                classes.append((set(st.Coverage.glyphs), st.ClassDef1.classDefs, st.ClassDef2.classDefs,
                                st.Class1Record))
    return pairs, classes


class _Face:
    def __init__(self, filename):
        path = _find_font(filename)
        self.tt = TTFont(str(path))
        self.key = Path(filename).stem
        self.upm = self.tt["head"].unitsPerEm
        self.cmap = self.tt.getBestCmap()
        self.glyphs = self.tt.getGlyphSet()
        self.hmtx = self.tt["hmtx"]
        self.pairs, self.classes = _gpos_kerning(self.tt)
        self._d, self._b = {}, {}

    def glyph(self, ch):
        g = self.cmap.get(ord(ch))
        if g is None:
            raise ValueError(f"{ch!r} is not in {self.key}")
        return g

    def kern(self, a, b):
        if (a, b) in self.pairs:
            return self.pairs[(a, b)]
        for cov, c1, c2, recs in self.classes:
            if a in cov:
                v = recs[c1.get(a, 0)].Class2Record[c2.get(b, 0)].Value1
                return (getattr(v, "XAdvance", 0) or 0) if v is not None else 0
        return 0

    def outline(self, g):
        if g not in self._d:
            pen = SVGPathPen(self.glyphs)
            self.glyphs[g].draw(pen)
            self._d[g] = pen.getCommands()
        return self._d[g]

    def bounds(self, g):
        if g not in self._b:
            pen = BoundsPen(self.glyphs)
            self.glyphs[g].draw(pen)
            self._b[g] = pen.bounds
        return self._b[g]


_FACES = {}
_LIGATURES = (("ffi", "ﬃ"), ("ffl", "ﬄ"), ("ff", "ﬀ"), ("fi", "ﬁ"), ("fl", "ﬂ"))
_LAYOUTS = {}


def _face(style):
    name = {"roman": "XCharter-Roman.otf", "italic": "XCharter-Italic.otf"}[style]
    if name not in _FACES:
        _FACES[name] = _Face(name)
    return _FACES[name]


def _layout(s):
    """The glyphs of a label, [(face, glyph, x)], and its ink box (x0, y0, x1, y1), in font units
    (1000 to the em) with y up and the baseline at 0. Between dollars is italic."""
    if s in _LAYOUTS:
        return _LAYOUTS[s]
    if "\u2014" in s or "\u2013" in s:
        raise ValueError(f"label {s!r}: no em or en dashes")
    parts = s.split("$")
    if len(parts) % 2 == 0:
        raise ValueError(f"label {s!r}: a $ is not closed")
    items, x, prev = [], 0.0, None
    for i, t in enumerate(parts):
        if not t:
            continue
        f = _face("italic" if i % 2 else "roman")
        for a, b in _LIGATURES:
            t = t.replace(a, b)
        t = t.replace("-", "−") if i % 2 else t
        for ch in t:
            g = f.glyph(ch)
            if prev is not None and prev[0] is f:
                x += f.kern(prev[1], g)
            items.append((f, g, x))
            x += f.hmtx[g][0]
            prev = (f, g)
        prev = None
    boxes = [(gx + b[0], b[1], gx + b[2], b[3]) for f, g, gx in items if (b := f.bounds(g)) is not None]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)) \
        if boxes else (0.0, 0.0, 0.0, 0.0)
    _LAYOUTS[s] = (items, box, x)
    return _LAYOUTS[s]


def measure(s, size="normal"):
    """Width and height, in points, of the ink of the label s."""
    _, (x0, y0, x1, y1), _ = _layout(s)
    k = SIZES[_choose(size, SIZES, "size")] / 1000
    return (x1 - x0) * k, (y1 - y0) * k


# ------------------------------------------------------------------------------- writing numbers


def _num(v, nd):
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def _join(tokens):
    out = []
    for i, t in enumerate(tokens):
        if i and not t.startswith("-"):
            out.append(" ")
        out.append(t)
    return "".join(out)


def _poly_d(p, closed=False, nd=2):
    q = np.round(np.asarray(p, float), nd)
    d = "M" + _join([_num(q[0, 0], nd), _num(q[0, 1], nd)])
    if len(q) > 1:
        d += "l" + _join([_num(v, nd) for v in np.diff(q, axis=0).ravel()])
    return d + ("z" if closed else "")


def _dots_d(pts):
    """Round dots at the points: each a stroke 0.04 long with round caps (WebKit draws a stroke of
    length 0 heavier than the rest)."""
    q = np.round(np.asarray(pts, float), 2)
    d = ["M" + _join([_num(q[0, 0], 2), _num(q[0, 1], 2)]) + "h.04"]
    step = np.diff(q, axis=0) - [0.04, 0.0]
    d += ["m" + _join([_num(a, 2), _num(b, 2)]) + "h.04" for a, b in step]
    return "".join(d)


# ------------------------------------------------------------------------------- geometry


def _pieces(X, Y):
    p = np.column_stack([np.ravel(X), np.ravel(Y)]).astype(float)
    good = np.isfinite(p).all(axis=1)
    out, start = [], None
    for i, g in enumerate(np.append(good, False)):
        if g and start is None:
            start = i
        elif not g and start is not None:
            out.append(p[start:i])
            start = None
    return [q for q in out if len(q) >= 2]


def _dedupe(p):
    keep = np.ones(len(p), bool)
    keep[1:] = np.hypot(*np.diff(p, axis=0).T) > 1e-9
    return p[keep]


def _length(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def _at(p, s):
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = min(max(s, 0.0), cum[-1])
    k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    t = 0.0 if seg[k] == 0 else (s - cum[k]) / seg[k]
    d = p[k + 1] - p[k]
    nd = np.hypot(*d)
    return p[k] + t * d, (d / nd if nd else np.array([1.0, 0.0]))


def _trim(p, s0, s1):
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    a, _ = _at(p, s0)
    b, _ = _at(p, s1)
    return np.vstack([a, p[(cum > s0) & (cum < s1)], b])


def _inside(pts, rings):
    inside = np.zeros(len(pts), bool)
    for r in rings:
        inside ^= _MPath(np.vstack([r, r[:1]]), closed=True).contains_points(pts)
    return inside


def _hatch(rings, pitch, angle):
    """Segments of the lines at `angle` degrees, pitch apart on one grid for the whole page, inside
    the rings by the even-odd rule (torux's)."""
    a = math.radians(angle)
    dvec = np.array([math.cos(a), -math.sin(a)])
    nvec = np.array([-dvec[1], dvec[0]])
    ks, us = [], []
    for r in rings:
        p0, p1 = r, np.roll(r, -1, axis=0)
        s0, s1 = p0 @ nvec, p1 @ nvec
        u0, u1 = p0 @ dvec, p1 @ dvec
        k0 = np.ceil(np.minimum(s0, s1) / pitch).astype(np.int64)
        k1 = np.ceil(np.maximum(s0, s1) / pitch).astype(np.int64)
        cnt = np.maximum(k1 - k0, 0)
        if cnt.sum() == 0:
            continue
        edge = np.repeat(np.arange(len(r)), cnt)
        k = np.repeat(k0, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        t = (k * pitch - s0[edge]) / (s1[edge] - s0[edge])
        ks.append(k)
        us.append(u0[edge] + t * (u1[edge] - u0[edge]))
    if not ks:
        return np.zeros((0, 2, 2))
    k, u = np.concatenate(ks), np.concatenate(us)
    order = np.lexsort((u, k))
    k, u = k[order], u[order]
    if len(k) % 2 or not np.array_equal(k[0::2], k[1::2]):
        raise RuntimeError("hatching: a region's outline does not close")
    c, a0, a1 = k[0::2] * pitch, u[0::2], u[1::2]
    keep = a1 - a0 > 0.2
    c, a0, a1 = c[keep], a0[keep], a1[keep]
    return np.stack([c[:, None] * nvec + a0[:, None] * dvec, c[:, None] * nvec + a1[:, None] * dvec], axis=1)


_NOISE = {}


def _blue_noise(width, height, spacing, seed=7):
    """Points no closer than `spacing` that fill the page evenly with no lattice (torux's)."""
    key = (math.ceil(width), math.ceil(height), spacing, seed)
    if key not in _NOISE:
        r = spacing
        h = r / math.sqrt(2)
        nx, ny, P = int(math.ceil(key[0] / h)) + 1, int(math.ceil(key[1] / h)) + 1, 2
        gx = np.full((ny + 2 * P, nx + 2 * P), np.nan)
        gy = np.full_like(gx, np.nan)
        rng = np.random.default_rng(seed)
        near = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if (i, j) != (0, 0)]
        for _ in range(16):
            for pj in range(3):
                for pi in range(3):
                    jj, ii = np.meshgrid(np.arange(pj, ny, 3), np.arange(pi, nx, 3), indexing="ij")
                    jj, ii = jj.ravel(), ii.ravel()
                    empty = np.isnan(gx[jj + P, ii + P])
                    jj, ii = jj[empty], ii[empty]
                    x = (ii + rng.random(len(ii))) * h
                    y = (jj + rng.random(len(jj))) * h
                    ok = np.ones(len(ii), bool)
                    for di, dj in near:
                        qx, qy = gx[jj + P + dj, ii + P + di], gy[jj + P + dj, ii + P + di]
                        with np.errstate(invalid="ignore"):
                            ok &= ~((qx - x) ** 2 + (qy - y) ** 2 < r * r)
                    gx[jj[ok] + P, ii[ok] + P] = x[ok]
                    gy[jj[ok] + P, ii[ok] + P] = y[ok]
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        _NOISE[key] = pts[np.isfinite(pts[:, 0])]
    return _NOISE[key]


def _point(p, cum, s):
    k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(p) - 2))
    seg = cum[k + 1] - cum[k]
    t = 0.0 if seg == 0 else (s - cum[k]) / seg
    return p[k] + t * (p[k + 1] - p[k])


def _dash_geometry(p, closed, dash, weight, w, dot):
    """Dashes (polylines) and dots (centres) of a patterned line, the pattern stretched so that an
    open line begins and ends on its first element (torux's)."""
    if closed:
        p = np.vstack([p, p[:1]])
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    L = cum[-1]
    pat = np.array(_DASH[dash]) * _DASH_SCALE[weight]
    elems = [("dot", 0.0) if v == 0 else ("dash", max(v - w, 0.2)) for v in pat[0::2]]
    caps = [0.5 * (dot if kind == "dot" else w) for kind, _ in elems]
    gaps = [pat[2 * i + 1] + caps[i] + caps[(i + 1) % len(elems)] for i in range(len(elems))]
    T = sum(length for _, length in elems) + sum(gaps)
    first = elems[0][1]
    if closed:
        n = max(2, int(round(L / T)))
        scale, tail = L / (n * T), False
    elif L >= T + first or (elems[0][0] == "dot" and L >= T):
        n = max(1, int(round((L - first) / T)))
        scale, tail = L / (n * T + first), True
    else:
        return [p], []

    def piece(a, b):
        inner = p[(cum > a) & (cum < b)]
        return np.vstack([_point(p, cum, a), inner, _point(p, cum, b)])

    dashes, dots, s = [], [], 0.0
    for k in range(n + (1 if tail else 0)):
        for i, (kind, length) in enumerate(elems):
            if tail and k == n and i > 0:
                break
            if kind == "dot":
                dots.append(_point(p, cum, s))
            else:
                dashes.append(piece(s, min(s + length * scale, L)))
            s += (length + gaps[i]) * scale
    return dashes, dots


# ------------------------------------------------------------------------------- drawing


class _Drawing:
    _fig: "Figure"
    _clip = None

    def _map(self, x, y):
        raise NotImplementedError

    def _xy(self, x, y):
        X, Y = self._map(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(X, float), np.asarray(Y, float)

    def page(self, x, y):
        """Page coordinates, in points from the top left, of the points (x, y) of this view."""
        return self._xy(x, y)

    def line(self, x, y, weight="medium", dash="solid", colour="ink", closed=False, arrow=None, halo=False,
             alpha=None):
        """alpha: for a family of many ink lines, each drawn partly transparent so that where they run
        together the ink builds up and where one runs alone it stays faint. A number between 0 and 1,
        or {"light": a, "dark": b} for each page its own. Each call is its own path, so lines that
        cross build up and a line never builds up on itself."""
        _choose(weight, WEIGHTS, "weight")
        _choose(dash, DASHES, "dash")
        _choose(colour, COLOURS, "colour")
        if alpha is not None:
            values = alpha.values() if isinstance(alpha, dict) else [alpha]
            if colour != "ink" or dash != "solid" or halo or arrow is not None \
                    or not all(0 < float(v) <= 1 for v in values):
                raise ValueError("alpha is for plain solid ink lines, a number in (0, 1] or one for each page")
        hl = _ARROW[weight][0]
        pieces, heads = [], []
        for p in _pieces(*self._xy(x, y)):
            p = _dedupe(p)
            if len(p) < 2:
                continue
            L = _length(p)
            if isinstance(arrow, str) and not closed and L > hl:
                ends = ("start", "end") if arrow == "both" else (arrow,)
                for end in ends:
                    tip, _ = _at(p, L if end == "end" else 0.0)
                    back, _ = _at(p, L - 0.8 * hl if end == "end" else 0.8 * hl)
                    heads.append((tip, tip - back))
                p = _trim(p, 0.5 * hl if "start" in ends else 0.0, L - 0.5 * hl if "end" in ends else L)
            pieces.append(p)
        if pieces:
            self._fig._ops.append({"kind": "line", "pieces": pieces, "closed": bool(closed), "weight": weight,
                                   "dash": dash, "colour": colour, "halo": bool(halo), "clip": self._clip,
                                   "alpha": alpha})
        for tip, direction in heads:
            self._fig._ops.append({"kind": "head", "tip": tip, "dir": direction, "weight": weight,
                                   "colour": colour, "clip": self._clip})

    def arrow(self, x0, y0, x1, y1, weight="medium", colour="ink"):
        self.line([x0, x1], [y0, y1], weight=weight, colour=colour, arrow="end")

    def fill(self, x, y, tone="solid", colour="ink", flip=False):
        """A tone inside the closed curve (x, y), or inside several, fill([xa, xb], [ya, yb]), by the
        even-odd rule. No edge is drawn."""
        _choose(tone, TONES, "tone")
        _choose(colour, COLOURS, "colour")
        if (isinstance(x, (list, tuple)) and len(x) and np.ndim(x[0]) >= 1) or np.ndim(x) == 2:
            pairs = list(zip(x, y))
        else:
            pairs = [(x, y)]
        rings = []
        for a, b in pairs:
            X, Y = self._xy(a, b)
            r = np.column_stack([X.ravel(), Y.ravel()])
            r = _dedupe(r[np.isfinite(r).all(axis=1)])
            if len(r) > 1 and np.allclose(r[0], r[-1]):
                r = r[:-1]
            if len(r) >= 3:
                rings.append(r)
        if rings:
            self._fig._ops.append({"kind": "fill", "rings": rings, "tone": tone, "colour": colour,
                                   "flip": bool(flip), "clip": self._clip})

    def mark(self, x, y, kind="dot", colour="ink"):
        _choose(kind, MARKS, "mark")
        X, Y = self._xy(np.atleast_1d(x), np.atleast_1d(y))
        pts = np.column_stack([X.ravel(), Y.ravel()])
        self._fig._ops.append({"kind": "mark", "pts": pts[np.isfinite(pts).all(axis=1)], "mark": kind,
                               "colour": colour, "clip": self._clip})

    def text(self, x, y, s, anchor="c", size="normal", gap=None, baseline=False, knockout=True, colour="ink"):
        """Letter s at (x, y). anchor names the point of the label's ink box that sits there ("sw":
        the label lies above and to the right). Returns the ink box (left, top, right, bottom)."""
        X, Y = self._xy(x, y)
        return self._fig._letter(float(X), float(Y), s, anchor, size, gap, baseline, knockout, colour)

    def leader(self, x, y, s, tx, ty, end=None, size="normal"):
        """The label s at (x, y), on the side away from the target, with a fine leader to (tx, ty)."""
        _choose(end, (None, "dot", "arrow"), "end")
        X, Y = (float(v) for v in self._xy(x, y))
        TX, TY = (float(v) for v in self._xy(tx, ty))
        ang = math.degrees(math.atan2(-(TY - Y), TX - X)) % 360
        anchor = ["e", "ne", "n", "nw", "w", "sw", "s", "se"][int(((ang + 22.5) % 360) // 45)]
        box = self._fig._letter(X, Y, s, anchor, size, _GAP, False, True, "ink")
        start, target = np.array([X, Y]), np.array([TX, TY])
        v = target - start
        L = float(np.hypot(*v))
        if L < 0.5:
            return box
        u = v / L
        stop = target - 0.5 * _ARROW["fine"][0] * u if end == "arrow" else target
        self._fig._ops.append({"kind": "line", "pieces": [np.array([start, stop])], "closed": False, "weight": "fine",
                               "dash": "solid", "colour": "ink", "halo": False, "clip": None})
        if end == "arrow":
            self._fig._ops.append({"kind": "head", "tip": target, "dir": u, "weight": "fine", "colour": "ink",
                                   "clip": None})
        elif end == "dot":
            self._fig._ops.append({"kind": "mark", "pts": target[None, :], "mark": "end", "colour": "ink",
                                   "clip": None})
        return box


class View(_Drawing):
    """Data coordinates mapped onto a box of the page, x to the right and y up."""

    def __init__(self, fig, box, x, y, clip=False):
        self._fig = fig
        self.box = tuple(float(v) for v in box)
        self.lim = (float(x[0]), float(x[1]), float(y[0]), float(y[1]))
        self._clip = fig._clip_id(self.box) if clip else None

    def _map(self, x, y):
        x0, x1, y0, y1 = self.lim
        left, top, width, height = self.box
        return left + (x - x0) / (x1 - x0) * width, top + (y1 - y) / (y1 - y0) * height


class Axes(View):
    """A View with a frame, inward ticks and names, put on over the drawing when the figure is saved."""

    def __init__(self, fig, box, x, y):
        super().__init__(fig, box, x, y, clip=True)
        self._frame, self._ticks, self._names, self._done = "box", {}, {}, False
        fig._axes.append(self)

    def frame(self, kind="box"):
        self._frame = _choose(kind, ("box", "open", "bottom", "none"), "frame")

    def xticks(self, major=(), minor=(), labels=None, top=True):
        self._ticks["x"] = (list(major), list(minor), labels, top)

    def yticks(self, major=(), minor=(), labels=None, right=True):
        self._ticks["y"] = (list(major), list(minor), labels, right)

    def xlabel(self, s, at="centre"):
        self._names["x"] = (s, at)

    def ylabel(self, s, at="top"):
        self._names["y"] = (s, at)

    def _line(self, pieces, butt=False):
        self._fig._late.append({"kind": "line", "pieces": [np.asarray(p, float) for p in pieces], "closed": False,
                                "weight": "medium", "dash": "solid", "colour": "ink", "halo": False, "clip": None,
                                "butt": butt, "mitre": not butt})

    def _finish(self):
        L, T, w, h = self.box
        R, B = L + w, T + h
        if self._frame == "box":
            self._line([[[L, T], [R, T], [R, B], [L, B], [L, T]]])
        elif self._frame == "open":
            self._line([[[L, T], [L, B], [R, B]]])
        elif self._frame == "bottom":
            self._line([[[L, B], [R, B]]])
        numeral = _NUMERAL * SIZES["small"]
        for axis, (major, minor, labels, both) in self._ticks.items():
            segs = []
            for values, length in ((major, _TICK["major"]), (minor, _TICK["minor"])):
                for v in values:
                    if axis == "x":
                        q = float(self._xy(v, self.lim[2])[0])
                        segs.append([[q, B], [q, B - length]])
                        if both and self._frame == "box":
                            segs.append([[q, T], [q, T + length]])
                    else:
                        q = float(self._xy(self.lim[0], v)[1])
                        segs.append([[L, q], [L + length, q]])
                        if both and self._frame == "box":
                            segs.append([[R, q], [R - length, q]])
            if segs:
                self._line(segs, butt=True)
            if labels is False:
                continue
            if labels is None:
                labels = [f"{v:g}" for v in major]
            for v, s in zip(major, labels):
                if not s:
                    continue
                if axis == "x":
                    q = float(self._xy(v, self.lim[2])[0])
                    self._fig._letter(q, B + 3.6 + numeral, s, "c", "small", 0.0, True, False, "ink")
                else:
                    q = float(self._xy(self.lim[0], v)[1])
                    self._fig._letter(L - 3.6, q + 0.5 * numeral, s, "e", "small", 0.0, True, False, "ink")
        if "x" in self._names:
            s, at = self._names["x"]
            if at == "centre":
                self._fig._letter(L + 0.5 * w, B + 3.6 + numeral + 4.0, s, "n", "normal", 0.0, False, False, "ink")
            else:
                self._fig._letter(R + 4.0, B, s, "w", "normal", 0.0, False, False, "ink")
        if "y" in self._names:
            s, at = self._names["y"]
            self._fig._letter(L, T - 4.5, s, "sw" if at == "top" else "s", "normal", 0.0, False, False, "ink")


class Figure(_Drawing):
    """A figure `width` by `height` points as printed. Its own calls take points from the top left."""

    def __init__(self, width, height):
        self.width, self.height = float(width), float(height)
        self._fig = self
        self._ops, self._late, self._letters, self._clips, self._axes = [], [], [], {}, []

    def _map(self, x, y):
        return x, y

    def view(self, box, x, y, clip=False):
        return View(self, box, x, y, clip=clip)

    def axes(self, box, x, y):
        return Axes(self, box, x, y)

    def _clip_id(self, box):
        key = tuple(round(v, 3) for v in box)
        return self._clips.setdefault(key, f"k{len(self._clips)}")

    def _letter(self, X, Y, s, anchor, size, gap, baseline, knockout, colour):
        _choose(anchor, ANCHORS, "anchor")
        _choose(colour, COLOURS, "colour")
        items, (bx0, by0, bx1, by1), _ = _layout(s)
        k = SIZES[_choose(size, SIZES, "size")] / 1000
        if gap is None:
            gap = 0.0 if anchor == "c" else _GAP
        l, r, t, b = bx0 * k, bx1 * k, -by1 * k, -by0 * k
        hx = l if "w" in anchor else r if "e" in anchor else 0.5 * (l + r)
        vy = t if anchor[0] == "n" else b if anchor[0] == "s" else 0.5 * (t + b)
        dx = gap if "w" in anchor else -gap if "e" in anchor else 0.0
        dy = gap if anchor[0] == "n" else -gap if anchor[0] == "s" else 0.0
        ox = X + dx - hx
        oy = Y if baseline else Y + dy - vy
        box = (ox + l, oy + t, ox + r, oy + b)
        self._letters.append({"items": items, "o": (ox, oy), "k": k, "box": box, "knockout": bool(knockout),
                              "colour": colour})
        return box

    # --------------------------------------------------------------- writing

    def svg(self, page, px=None):
        """The figure as SVG text for the page "light" or "dark": px pixels to the point for a screen,
        or None for a file sized in points, for print."""
        P = PAGES[_choose(page, PAGES, "page")]
        for ax in self._axes:
            if not ax._done:
                ax._finish()
                ax._done = True
        defs, body = [], []
        for (l, t, w, h), cid in self._clips.items():
            defs.append(f'<clipPath id="{cid}"><rect x="{_num(l, 2)}" y="{_num(t, 2)}" width="{_num(w, 2)}" '
                        f'height="{_num(h, 2)}"/></clipPath>')
        gid = {}
        for L in self._letters:
            for f, g, _ in L["items"]:
                if (f.key, g) not in gid:
                    gid[(f.key, g)] = f"g{len(gid)}"
                    d = f.outline(g)
                    defs.append(f'<path id="{gid[(f.key, g)]}" d="{d}"/>')
        for op in self._ops + self._late:
            items = self._render(op, P)
            if items and op.get("clip"):
                items = [f'<g clip-path="url(#{op["clip"]})">' + "".join(items) + "</g>"]
            body.extend(items)
        knock, letters = [], []
        for L in self._letters:
            if L["knockout"]:
                x0, y0, x1, y1 = L["box"]
                m = _KNOCK
                knock.append(f'<rect x="{_num(x0 - m, 2)}" y="{_num(y0 - m, 2)}" width="{_num(x1 - x0 + 2 * m, 2)}" '
                             f'height="{_num(y1 - y0 + 2 * m, 2)}" fill="{P["paper"]}" stroke="none"/>')
            k, (ox, oy) = L["k"], L["o"]
            uses = "".join(f'<use href="#{gid[(f.key, g)]}" x="{_num(x, 1)}"/>' for f, g, x in L["items"]
                           if f.bounds(g) is not None)
            letters.append(f'<g transform="matrix({_num(k, 6)} 0 0 {_num(-k, 6)} {_num(ox, 3)} {_num(oy, 3)})" '
                           f'fill="{P["red"] if L["colour"] == "red" else P["ink"]}" stroke="none">{uses}</g>')
        body += knock + letters
        W, H = _num(self.width, 3), _num(self.height, 3)
        size = (f'width="{_num(self.width * px, 1)}" height="{_num(self.height * px, 1)}"' if px
                else f'width="{W}pt" height="{H}pt"')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" {size} viewBox="0 0 {W} {H}">\n'
                + (f"<defs>{''.join(defs)}</defs>\n" if defs else "")
                + f'<rect width="{W}" height="{H}" fill="{P["paper"]}"/>\n'
                + f'<g fill="none" stroke="{P["ink"]}" stroke-linecap="round" stroke-linejoin="round">\n'
                + "\n".join(body) + "\n</g>\n</svg>\n")

    def _render(self, op, P):
        ink, paper = P["ink"], P["paper"]
        col = P["red"] if op.get("colour") == "red" else ink
        sc = f' stroke="{col}"' if col != ink else ""
        kind, out = op["kind"], []
        if kind == "line":
            w = P["weight"][op["weight"]]
            closed = op["closed"]
            if op["halo"]:
                d = "".join(_poly_d(p, closed) for p in op["pieces"])
                out.append(f'<path d="{d}" stroke="{paper}" stroke-width="{_num(w + 2 * _HALO, 3)}"/>')
            if op.get("butt"):
                d = "".join(_poly_d(p) for p in op["pieces"])
                out.append(f'<path d="{d}" stroke-width="{_num(w, 3)}" stroke-linecap="butt"{sc}/>')
            elif op["dash"] == "solid":
                d = "".join(_poly_d(p, closed) for p in op["pieces"])
                join = ' stroke-linejoin="miter" stroke-linecap="square"' if op.get("mitre") else ""
                a = op.get("alpha")
                a = a[P["name"]] if isinstance(a, dict) else a
                tone = f' stroke-opacity="{float(a):.3f}"' if a is not None and float(a) < 1 else ""
                out.append(f'<path d="{d}" stroke-width="{_num(w, 3)}"{join}{sc}{tone}/>')
            else:
                strokes, dots, dot = [], [], max(w, P["dot"])
                for p in op["pieces"]:
                    dashes, centres = _dash_geometry(p, closed, op["dash"], op["weight"], w, dot)
                    strokes += dashes
                    dots += centres
                if strokes:
                    out.append(f'<path d="{"".join(_poly_d(q) for q in strokes)}" stroke-width="{_num(w, 3)}"{sc}/>')
                if dots:
                    out.append(f'<path d="{_dots_d(np.array(dots))}" stroke-width="{_num(dot, 3)}"{sc}/>')
        elif kind == "head":
            hl, hw = _ARROW[op["weight"]]
            u = np.asarray(op["dir"], float)
            u = u / np.hypot(*u)
            n = np.array([-u[1], u[0]])
            tip = np.asarray(op["tip"], float)
            pts = np.array([tip, tip - hl * u + hw * n, tip - 0.78 * hl * u, tip - hl * u - hw * n])
            out.append(f'<path d="{_poly_d(pts, True)}" fill="{col}" stroke="none"/>')
        elif kind == "mark":
            r, mk, w = _MARK[op["mark"]], op["mark"], P["weight"]["medium"]
            for x, y in op["pts"]:
                cx, cy = _num(x, 2), _num(y, 2)
                if mk in ("dot", "end"):
                    out.append(f'<circle cx="{cx}" cy="{cy}" r="{_num(r, 2)}" fill="{col}" stroke="none"/>')
                elif mk == "circle":
                    out.append(f'<circle cx="{cx}" cy="{cy}" r="{_num(r, 2)}" fill="{paper}" '
                               f'stroke-width="{_num(w, 3)}"{sc}/>')
                else:
                    a = r / math.sqrt(2)
                    d = (f"M{_num(x - a, 2)} {_num(y - a, 2)}l{_num(2 * a, 2)} {_num(2 * a, 2)}"
                         f"M{_num(x - a, 2)} {_num(y + a, 2)}l{_num(2 * a, 2)} {_num(-2 * a, 2)}")
                    out.append(f'<path d="{d}" stroke-width="{_num(w, 3)}"{sc}/>')
        elif kind == "fill":
            tone, rings = op["tone"], op["rings"]
            if tone in ("solid", "paper"):
                d = "".join(_poly_d(r, True) for r in rings)
                out.append(f'<path d="{d}" fill="{col if tone == "solid" else paper}" fill-rule="evenodd" '
                           f'stroke="none"/>')
            elif tone == "hatch":
                spec = P["hatch"]
                segs = _hatch(rings, spec["pitch"], -45.0 if op["flip"] else 45.0)
                if len(segs):
                    q = np.round(np.array(segs), 2)
                    d = "".join("M" + _join([_num(s[0, 0], 2), _num(s[0, 1], 2)]) + "l"
                                + _join([_num(s[1, 0] - s[0, 0], 2), _num(s[1, 1] - s[0, 1], 2)]) for s in q)
                    out.append(f'<path d="{d}" stroke-width="{_num(spec["width"], 3)}" stroke-linecap="butt"{sc}/>')
            else:
                spec = P["stipple"]
                lo = np.min([r.min(axis=0) for r in rings], axis=0)
                hi = np.max([r.max(axis=0) for r in rings], axis=0)
                pts = _blue_noise(max(self.width, hi[0]) + 5, max(self.height, hi[1]) + 5, spec["spacing"])
                pts = pts[((pts >= lo) & (pts <= hi)).all(axis=1)]
                pts = pts[_inside(pts, rings)]
                if len(pts):
                    out.append(f'<path d="{_dots_d(pts[np.lexsort((pts[:, 0], pts[:, 1]))])}" '
                               f'stroke-width="{_num(spec["dot"], 3)}"{sc}/>')
        return out

    def save(self, name, out, px=2.4, previews=True):
        """Write <name>-light.svg and <name>-dark.svg (px pixels to the point) and <name>.pdf (in points,
        light) into the directory out, after checking each is ink, paper and red alone. With previews,
        also PNGs of the screen files at their own width and at 324 px, and of the PDF at 300 dpi."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for page in ("light", "dark"):
            s = self.svg(page, px)
            _verify(s, PAGES[page])
            p = out / f"{name}-{page}.svg"
            p.write_text(s)
            written.append(p)
        tool = shutil.which("rsvg-convert")
        if tool is None:
            raise RuntimeError("the PDF needs rsvg-convert (brew install librsvg)")
        tmp = out / f".{name}-print.svg"
        s = self.svg("light", None)
        _verify(s, PAGES["light"])
        tmp.write_text(s)
        pdf = out / f"{name}.pdf"
        env = dict(os.environ, SOURCE_DATE_EPOCH="0")     # no creation date: the same bytes every run
        subprocess.run([tool, "-f", "pdf", "-o", str(pdf), str(tmp)], check=True, env=env)
        tmp.unlink()
        written.append(pdf)
        if previews:
            pv = out / "previews"
            pv.mkdir(exist_ok=True)
            for p in written[:2]:
                for w in (round(self.width * px), 324):
                    subprocess.run([tool, "-w", str(w), "-o", str(pv / f"{p.stem}-{w}.png"), str(p)], check=True)
            subprocess.run([tool, "-w", str(round(self.width / 72 * 300)), "-o", str(pv / f"{name}-print-300dpi.png"),
                            str(written[0])], check=True)
        print(", ".join(f"{p.name} {p.stat().st_size / 1024:.0f} KB" for p in written))
        return written


def _verify(svg, page):
    """Refuse a file with anything but ink, paper and red, and any transparency but the stroke
    opacity of an ink line (alpha)."""
    colours = {c.lower() for c in re.findall(r"#[0-9a-fA-F]{3,8}\b", svg)}
    extra = colours - {page["ink"].lower(), page["paper"].lower(), page["red"].lower()}
    if extra:
        raise RuntimeError(f"colours other than ink, paper and red: {sorted(extra)}")
    for path in re.findall(r"<path [^>]*stroke-opacity[^>]*>", svg):
        if re.search(r' stroke="(?!' + re.escape(page["ink"]) + ')', path) or "fill=" in path:
            raise RuntimeError(f"alpha on something other than an ink line: {path[:120]}")
    for v in re.findall(r' stroke-opacity="([^"]*)"', svg):
        if not 0 < float(v) < 1:
            raise RuntimeError(f"stroke-opacity {v} is not between 0 and 1")
    bare = re.sub(r' stroke-opacity="[0-9.]+"', "", svg)
    for word in ("opacity", "gradient", "pattern", "filter", "mask", "rgb(", "<image", "<text", "font"):
        if word in bare:
            raise RuntimeError(f"not allowed in an ink plate: {word}")
