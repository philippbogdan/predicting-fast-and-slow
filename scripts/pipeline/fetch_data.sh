#!/usr/bin/env bash
# Download the public inputs: Heimdall's code at the commit used here (it holds the five raw traces) and its
# client-level dataset (3.5 GB, six trace pairs with per-I/O replay logs and trained weights), then check
# them against scripts/pipeline/SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/../.."

HEIMDALL_COMMIT=dd0e0d1903e9c9a2c5c62c8056035d18860c40e1
if [ ! -d data/heimdall_repo/src/.git ]; then
  git clone https://github.com/ucare-uchicago/Heimdall.git data/heimdall_repo/src
fi
git -C data/heimdall_repo/src checkout --quiet "$HEIMDALL_COMMIT"

# the client-level dataset, linked from Heimdall's documentation/3_client_level_integration.md
mkdir -p data/client_level_gdrive/extracted
if [ ! -f data/client_level_gdrive/data.tar.gz ]; then
  ${PY:-python} -m gdown "https://drive.google.com/uc?id=149SkDXwpAEchrBTRL1zx8IxUo4E7Gwq6" -O data/client_level_gdrive/data.tar.gz
fi
shasum -a 256 -c scripts/pipeline/SHA256SUMS
if [ ! -d data/client_level_gdrive/extracted/data ]; then
  tar -xzf data/client_level_gdrive/data.tar.gz -C data/client_level_gdrive/extracted
fi
echo "data ready"
