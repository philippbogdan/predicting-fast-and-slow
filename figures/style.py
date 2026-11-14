"""One look for every figure of the paper: one hue, red, for what the drive's outcomes say, greys for the rest.

The paper's theme draws into paper/figures/. Figure 2 (figures/table.py) is also drawn light and dark, in the
paper's face, into figures/ for the pages.
"""
import logging
from pathlib import Path

import matplotlib as mpl

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)     # the font list names fallbacks

RED = "#E73121"
PAPER = {"bg": "white", "fg": "#1a1a1a", "muted": "#6b6b6b", "faint": "#d9d9d9", "grid": "#ececec",
         "outcome": RED, "input": "#8a8f98", "input_dark": "#565b63",
         "font": ["Charter", "XCharter", "STIX Two Text", "DejaVu Serif"]}
THEMES = {
    "paper": PAPER,
    "light": {**PAPER, "fg": "#1f2328", "muted": "#59636e", "faint": "#d1d9e0", "grid": "#eef1f4"},
    "dark": {**PAPER, "bg": "#0d1117", "fg": "#f0f6fc", "muted": "#9198a1", "faint": "#3d444d", "grid": "#21262d",
             "input": "#8b949e", "input_dark": "#b1bac4"},
}
HERE = Path(__file__).resolve().parent


def use(theme="paper"):
    t = THEMES[theme]
    mpl.rcParams.update({
        "font.family": t["font"], "font.size": 8.5, "axes.edgecolor": t["muted"], "axes.labelcolor": t["fg"],
        "text.color": t["fg"], "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "pdf.fonttype": 42, "mathtext.fontset": "stix",
        "svg.fonttype": "path", "svg.hashsalt": "predicting",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return t


def svg(fig):
    """A page's version of a figure, as SVG text: transparent, its letters as paths, no date."""
    import io
    out = io.StringIO()
    fig.savefig(out, format="svg", transparent=True, metadata={"Date": None, "Creator": None})
    return out.getvalue()


def save(fig, stem, theme="paper"):
    """paper/figures/<stem>.pdf, with no creation date or tool name in the file: the same bytes every run.
    For the light and dark themes, figures/<stem>-<theme>.svg."""
    if theme != "paper":
        (HERE / f"{stem}-{theme}.svg").write_text(svg(fig))
        return
    out = HERE.parent / "paper" / "figures" / f"{stem}.pdf"
    fig.savefig(out, metadata={"CreationDate": None, "Creator": None, "Producer": None})
