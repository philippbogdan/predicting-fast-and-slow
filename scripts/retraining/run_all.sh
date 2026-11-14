#!/bin/sh
# The four retraining schedules on the 57 held-out traces, then their result files and the summary.
#
# Usage: scripts/retraining/run_all.sh REPLAYS [WORK] [LANES]
#   REPLAYS  the held-out replay logs, <trace>.parquet, which the repository does not hold (retrain.py says how
#            to make them)
#   WORK     where the per-read files and the networks go, never the repository (default data/retraining)
#   LANES    parallel jobs, each taking every LANES-th trace (default 3)
# PYTHON=/path/to/python picks a Python with requirements.txt installed.
set -eu
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 2; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
REPLAYS=$1
WORK=${2:-$REPO/data/retraining}
LANES=${3:-3}
PY=${PYTHON:-python3}
cd "$REPO"
mkdir -p "$WORK/logs"
"$PY" -c 'import json; print("\n".join(t["name"] for t in json.load(open("results/heldout/traces.json"))["traces"]
                                       if t["scored"] and not t["development_exposed"]))' > "$WORK/traces.txt"
i=0
while [ "$i" -lt "$LANES" ]; do
  awk -v l="$LANES" -v i="$i" 'NR % l == i' "$WORK/traces.txt" > "$WORK/logs/lane$i.txt"
  ( while read -r t; do "$PY" scripts/retraining/retrain.py "$t" --replays "$REPLAYS" --work "$WORK"; done \
      < "$WORK/logs/lane$i.txt" ) > "$WORK/logs/lane$i.log" 2>&1 &
  i=$((i + 1))
done
wait
"$PY" scripts/retraining/metrics.py --work "$WORK"
"$PY" check.py --seal
