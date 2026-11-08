#!/usr/bin/env bash
# Task 22: regenerate every offline table twice into two directories and check they are byte-identical (determinism),
# writing manifest.json (git sha, versions, config hash, checkpoint + output hashes). Usage: ./reproduce.sh [--jsonl your.jsonl]
set -euo pipefail
cd "$(dirname "$0")"
python scripts/run_all_offline.py --out eval --quick "$@"
python scripts/run_all_offline.py --out eval_check --quick "$@" >/dev/null
for f in table2.json 2x2.json; do
  if cmp -s "eval/$f" "eval_check/$f"; then echo "reproducible: $f identical across runs"; else echo "NOT reproducible: $f differs"; exit 1; fi
done
