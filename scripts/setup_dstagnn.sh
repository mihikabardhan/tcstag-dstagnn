#!/usr/bin/env bash
# Clones the authors' DSTAGNN code (we do not redistribute it) and drops our model file into it.
# Usage: bash scripts/setup_dstagnn.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/third_party/DSTAGNN"
if [ ! -d "$TARGET" ]; then
  mkdir -p "$ROOT/third_party"
  git clone --depth 1 https://github.com/SYLan2019/DSTAGNN.git "$TARGET"
fi
cp "$ROOT/src/tcstag.py" "$TARGET/model/tcstag.py"
echo "Done. Authors' repo: $TARGET (our model copied to model/tcstag.py)"
echo "Next: put PEMS04.npz and PEMS04.csv in $TARGET/data/PEMS04/, then prepare the data (see README)."
