#!/usr/bin/env python3
"""What the plates of METHOD.md draw and their captions quote, beyond results/summary.json.

Standard library only, like summary.py, which it reads through. figures/plates/ draws the plates from these
functions, typeset.py sets macros() into METHOD.md beside the summary's own numbers (not into
paper/numbers.tex: the paper quotes none of them), and check.py recomputes them and tests statements(), what
the captions say in words.

  claim()  Fig. 1. Every scored held-out trace's later reads, in order from the guess a signal finds most
           suspect to the one it finds least, as scripts/hero_data.py stored them (results/heldout/hero: 240
           runs of equal size and the share of Heimdall's guesses wrong in each), cut into tenths. A tenth's
           value is the share wrong in it less the share wrong over the whole trace, in percentage points.
  bias()   Fig. 3. Each held-out trace's share of scored reads that came back slow, that Heimdall called slow
           and that its period labels mark slow (results/classwise/heldout), summed over the later windows
           as summary.classwise_block sums them.
  table()  Fig. 4. The learnt table of one held-out replay (results/heldout/table, from scripts/table_state.py).

Fig. 2 is drawn without data, and Fig. 5's values are those of figures/tail.py.
"""
import ast
import functools
import math
import statistics as st
import struct
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summary as S  # noqa: E402

SIGNALS = ("table", "euclid")      # Fig. 1: the drive's own table, and Euclidean distance
TENTHS = 10
TABLE_TRACE = "msr.mds_0.orig"     # Fig. 4
MIN_READS = 20                     # Fig. 4 leaves a cell with fewer reads blank
WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
_NPY = {"<f4": "f", "<f8": "d", "<i8": "q"}


def npz(path, names):
    """The named arrays of an .npz file NumPy wrote (little-endian, C order), each a list, or a number if 0-d."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in names:
            b = z.read(f"{name}.npy")
            if b[:6] != b"\x93NUMPY":
                raise ValueError(f"{path}: {name} is not a NumPy array")
            size, start = (struct.unpack("<H", b[8:10])[0], 10) if b[6] == 1 else (struct.unpack("<I", b[8:12])[0], 12)
            head = ast.literal_eval(b[start:start + size].decode("latin1"))
            if head["fortran_order"] or head["descr"] not in _NPY:
                raise ValueError(f"{path}: {name} is {head}, not read here")
            code, n = _NPY[head["descr"]], math.prod(head["shape"])
            v = list(struct.unpack(f"<{n}{code}", b[start + size:start + size + n * struct.calcsize(code)]))
            out[name] = v if head["shape"] else v[0]
    return out


def runs(z, key):
    """A stored strip's runs: their sizes, as np.diff(np.linspace(0, reads, runs + 1).round()), and their counts
    of wrong guesses, recovered exactly from each run's share and checked against the file's total."""
    n, k = z["reads"], z["runs"]
    step = n / k
    edges = [round(i * step) for i in range(k)] + [n]
    size = [b - a for a, b in zip(edges, edges[1:])]
    share = z[f"strip_{key}"]
    count = [round(s * m) for s, m in zip(share, size)]
    if any(abs(s * m - c) > 1e-3 + 1e-5 * abs(c) for s, m, c in zip(share, size, count)) or sum(count) != z["wrong"]:
        raise RuntimeError(f"{key}: the runs do not add up to the file's {z['wrong']} wrong guesses")
    return size, count


@functools.lru_cache(maxsize=None)
def claim():
    """{signal: ((points by tenth, most suspect first) for each scored held-out trace, in order of name)}."""
    _, _, _, scored = S.heldout_records()
    out = {key: [] for key in SIGNALS}
    for name in sorted(scored):
        z = npz(S.RES / "heldout" / "hero" / f"{name}.npz",
                ["reads", "wrong", "runs"] + [f"strip_{k}" for k in SIGNALS])
        for key in SIGNALS:
            size, count = runs(z, key)
            k = len(size) // TENTHS
            whole = sum(count) / sum(size)
            out[key].append(tuple(100 * (sum(count[i * k:(i + 1) * k]) / sum(size[i * k:(i + 1) * k]) - whole)
                                  for i in range(TENTHS)))
    return {key: tuple(v) for key, v in out.items()}


@functools.lru_cache(maxsize=None)
def bias():
    """{trace: {"slow", "calls", "labels": shares of its scored reads, "reads"}} for the held-out traces."""
    out = {}
    for name, r in sorted(S.classwise_records("heldout").items()):
        W = S.later(r)
        n = sum(w["n"] for w in W)
        out[name] = {"slow": sum(w["own_latency"]["slow"] for w in W) / n,
                     "calls": sum(w["called_slow"] for w in W) / n,
                     "labels": sum(w["period_labels"]["slow"] for w in W) / n, "reads": n}
    return out


@functools.lru_cache(maxsize=None)
def table():
    """The stored table of TABLE_TRACE, and its cells for reads with nothing outstanding for longer than T
    (age class 0), by (size class, load class, quiet bit)."""
    d = S.load(S.RES / "heldout" / "table" / f"{TABLE_TRACE}.json")
    return d, {(c["size_class"], c["load_class"], c["quiet"]): c for c in d["cells"] if c["age_class"] == 0}


def medians(rows):
    return [st.median(col) for col in zip(*rows)]


def macros(s):
    """The plates' captions' numbers that the paper does not quote, by the names METHOD.in.md uses."""
    c, b, (d, _) = claim(), bias(), table()
    tm, em = medians(c["table"]), medians(c["euclid"])
    return {
        # Fig. 1: points above or below each trace's own average, the median over traces
        "ClaimTableFirst": f"{tm[0]:.0f}", "ClaimTableLast": f"{-tm[-1]:.0f}",
        "ClaimTableOrdered": sum(r[0] > r[-1] for r in c["table"]),
        "ClaimDistWithin": math.ceil(max(abs(v) for v in em) - 1e-9),         # a bound, so rounded up
        "ClaimDistOrdered": sum(r[0] > r[-1] for r in c["euclid"]),
        # Fig. 3: Heimdall's calls against the reads and against its own labels
        "BiasAbove": sum(t["calls"] > t["slow"] for t in b.values()),
        "BiasGapReads": f"{100 * st.median(t['calls'] - t['slow'] for t in b.values()):.0f}",
        "BiasGapLabels": f"{100 * st.median(abs(t['calls'] - t['labels']) for t in b.values()):.0f}",
        # Fig. 4: the knee of the replay's first minute, and what the plate leaves blank
        "TableKneeUs": f"{d['T_us']:.0f}", "TableMinReads": MIN_READS,
        # Fig. 5: the artefact's pairs
        "TailPairs": WORDS[len(S.ARTEFACT_PAIRS)],
    }


def statements(s):
    """What the plates' captions say in words, each as a test on the recomputed numbers."""
    c, b = claim(), bias()
    tm = medians(c["table"])
    cw, traces = s["classwise"], s["heldout"]["traces"]
    gap = st.median(t["calls"] - t["slow"] for t in b.values())
    return [
        ("Fig. 1 draws every one of the held-out traces, and under the table the median falls from above each "
         "trace's average to below it, the most suspect tenth wrong more often than the least on all of them",
         len(c["table"]) == len(c["euclid"]) == traces and tm[0] > 0 > tm[-1]
         and sum(r[0] > r[-1] for r in c["table"]) == traces),
        ("Fig. 3 draws every one of the held-out traces, its red medians are the summary's, and "
         "Heimdall calls slow more often than reads come back slow by a median above zero",
         len(b) == traces and gap > 0
         and abs(st.median(t["slow"] for t in b.values()) - cw["heldout"]["reads_slow_median"]) < 1e-12
         and abs(st.median(t["calls"] for t in b.values()) - cw["heldout"]["heimdall_calls_slow_median"]) < 1e-12
         and abs(st.median(t["labels"] for t in b.values())
                 - cw["heldout_period_labels"]["reads_slow_median"]) < 1e-12),
    ]
