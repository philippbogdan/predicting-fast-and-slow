#!/bin/sh
# Rebuild everything that rests on the identical enterprise drives' runs, from their per-run records:
#   results/routing/enterprise_nvme.json   (scripts/enterprise_nvme.py)
#   results/summary.json, results/SHA256SUMS, paper/numbers.tex, README.md, docs/further.md, docs/results.md   (check.py --seal)
#   figures/tail.pdf and its light and dark SVGs, paper/figures/tail.pdf   (figures/tail.py)
#   paper/main.pdf   (pdflatex, or tectonic where there is no pdflatex)
# then runs check.py and the tests.  Stops at the first failure.
#
# Usage: scripts/rebuild.sh RUNS_ROOT      (the folder holding runs/, inputs/, train/ and quiet_criterion.json)
#        PYTHON=/path/to/python scripts/rebuild.sh RUNS_ROOT   (a Python with numpy, matplotlib and pytest)
set -eu
[ $# -eq 1 ] || { sed -n '2,10p' "$0"; exit 2; }
SRC=$1
PY=${PYTHON:-python3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

"$PY" -I scripts/enterprise_nvme.py "$SRC"
"$PY" -I check.py --seal
"$PY" -I figures/tail.py
cp figures/tail.pdf paper/figures/tail.pdf

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cp paper/main.tex paper/numbers.tex "$TMP"/
cp -R paper/figures "$TMP"/figures
if command -v pdflatex >/dev/null 2>&1; then
    (cd "$TMP" && for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || {
        tail -30 main.log; exit 1; }; done)
    LOG=$TMP/main.log
else
    # tectonic is XeTeX underneath: hyperref is told so, and the page is set as pdfLaTeX sets it
    printf '%s\n' '\PassOptionsToPackage{xetex}{hyperref}' '\input{main.tex}' > "$TMP/wrap.tex"
    (cd "$TMP" && tectonic --keep-logs wrap.tex >tectonic.out 2>&1) || { tail -30 "$TMP/tectonic.out"; exit 1; }
    mv "$TMP/wrap.pdf" "$TMP/main.pdf"
    LOG=$TMP/wrap.log
fi
cp "$TMP/main.pdf" paper/main.pdf
PAGES=$(pdfinfo paper/main.pdf 2>/dev/null | awk '/^Pages:/ {print $2}')
echo "paper/main.pdf: ${PAGES:-?} pages"
grep -E '^Overfull|undefined|Undefined' "$LOG" || echo "no overfull box, no undefined reference or macro"
[ "${PAGES:-5}" -le 4 ] || echo "NOTE: the paper is ${PAGES:-?} pages, more than 4"

"$PY" -I check.py
"$PY" -m pytest -q tests
