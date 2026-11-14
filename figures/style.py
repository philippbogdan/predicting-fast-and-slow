"""One look for every figure: one hue, red, for what the drive's outcomes say, greys for the rest."""
import matplotlib as mpl

RED = "#E73121"

THEMES = {
    "paper": {"bg": "white", "fg": "#1a1a1a", "muted": "#6b6b6b", "faint": "#d9d9d9", "grid": "#ececec",
              "outcome": RED, "input": "#8a8f98", "input_dark": "#565b63",
              "font": ["Charter", "XCharter", "STIX Two Text", "DejaVu Serif"], "transparent": False},
    "light": {"bg": "white", "fg": "#1f2328", "muted": "#59636e", "faint": "#d1d9e0", "grid": "#eef1f4",
              "outcome": RED, "input": "#8a8f98", "input_dark": "#565b63",
              "font": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "transparent": True},
    "dark": {"bg": "#0d1117", "fg": "#f0f6fc", "muted": "#9198a1", "faint": "#3d444d", "grid": "#21262d",
             "outcome": RED, "input": "#8b949e", "input_dark": "#b1bac4",
             "font": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "transparent": True},
}


def use(theme):
    t = THEMES[theme]
    mpl.rcParams.update({
        "font.family": t["font"], "font.size": 8.5, "axes.edgecolor": t["muted"], "axes.labelcolor": t["fg"],
        "text.color": t["fg"], "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "pdf.fonttype": 42, "svg.fonttype": "path", "svg.hashsalt": "predicting", "mathtext.fontset": "stix",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return t


def save(fig, stem, theme, t):
    """figures/<stem>.pdf for the paper; <stem>-light.svg and <stem>-dark.svg for the README."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    if theme == "paper":                      # no creation date or tool name in the files: the same bytes every run
        fig.savefig(here / f"{stem}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    else:
        fig.savefig(here / f"{stem}-{theme}.svg", transparent=t["transparent"],
                    metadata={"Date": None, "Creator": None})
