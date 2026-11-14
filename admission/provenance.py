"""Every result file a script writes is stamped with the commit of this repository it ran from."""
from __future__ import annotations

import subprocess

from . import ROOT


def revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def stamp(d: dict) -> dict:
    d = dict(d)
    d["revision"] = revision()
    return d
