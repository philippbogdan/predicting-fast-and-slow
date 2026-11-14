"""Per-decision signals that flag a learned I/O admission controller's mistakes.

Evaluated on the public traces of Heimdall (EuroSys 2025) and on two transfer
tasks.  The situation tracker is in admission.tracker.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
