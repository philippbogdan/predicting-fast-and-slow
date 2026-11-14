"""One look for every figure of the paper: one hue, red, for what the drive's outcomes say, greys for the rest."""
from pathlib import Path

import matplotlib as mpl

RED = "#E73121"
PAPER = {"bg": "white", "fg": "#1a1a1a", "muted": "#6b6b6b", "faint": "#d9d9d9", "grid": "#ececec",
         "outcome": RED, "input": "#8a8f98", "input_dark": "#565b63",
         "font": ["Charter", "XCharter", "STIX Two Text", "DejaVu Serif"]}


def use():
    t = PAPER
    mpl.rcParams.update({
        "font.family": t["font"], "font.size": 8.5, "axes.edgecolor": t["muted"], "axes.labelcolor": t["fg"],
        "text.color": t["fg"], "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "pdf.fonttype": 42, "mathtext.fontset": "stix",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return t


def save(fig, stem):
    """paper/figures/<stem>.pdf, with no creation date or tool name in the file: the same bytes every run."""
    out = Path(__file__).resolve().parents[1] / "paper" / "figures" / f"{stem}.pdf"
    fig.savefig(out, metadata={"CreationDate": None, "Creator": None, "Producer": None})
