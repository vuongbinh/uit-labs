#!/usr/bin/env bash
# Reproduce every benchmark number reported for HYPE-13.
#
# Usage:
#   bash scripts/run_text_feature_benchmark.sh [data-dir]
#
# Downloads the shards it needs into <data-dir> (default: data/) and writes
# JSON + markdown reports under reports/text_features/.
#
# Train and test are *different time periods* (shard_0000 = 2025-06,
# shard_0009 = 2026-03), so every reported delta is out-of-time.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${1:-data}"
PYTHON="${PYTHON:-.venv/bin/python}"
PHOBERT_ROWS="${PHOBERT_ROWS:-8000}"
TRAIN_SHARD="$DATA_DIR/shard_0000.parquet"
TEST_SHARD="$DATA_DIR/shard_0009.parquet"

mkdir -p "$DATA_DIR" reports/text_features

# Headline result: word-level TF-IDF, 128 SVD components, 2 seeds.
"$PYTHON" -m real_estate.text.benchmark \
  --train-shard "$TRAIN_SHARD" --test-shard "$TEST_SHARD" --cache-dir "$DATA_DIR" \
  --n-train 100000 --n-test 40000 --n-seeds 2 \
  --tfidf-mode word --tfidf-components 128 \
  --out-dir reports/text_features/main

# TF-IDF analyzer comparison. All three modes run at the *same* sample size and
# seed so they share an identical tabular baseline and are directly comparable.
for MODE in word char both; do
  # `word` at this scale is the matched comparison arm; the headline run above
  # already covers word at full scale.
  if [ "$MODE" = word ]; then
    SUFFIX="word_matched"
  else
    SUFFIX="$MODE"
  fi
  "$PYTHON" -m real_estate.text.benchmark \
    --train-shard "$TRAIN_SHARD" --test-shard "$TEST_SHARD" --cache-dir "$DATA_DIR" \
    --n-train 60000 --n-test 25000 --n-seeds 1 \
    --tfidf-mode "$MODE" --tfidf-components 128 \
    --out-dir "reports/text_features/tfidf_$SUFFIX"
done

# Optional PhoBERT arm. Skipped unless torch/transformers are installed.
if "$PYTHON" -c "import torch, transformers" >/dev/null 2>&1; then
  "$PYTHON" -m real_estate.text.benchmark \
    --train-shard "$TRAIN_SHARD" --test-shard "$TEST_SHARD" --cache-dir "$DATA_DIR" \
    --n-train "$PHOBERT_ROWS" --n-test "$((PHOBERT_ROWS / 2))" --n-seeds 1 \
    --tfidf-mode word --tfidf-components 128 --use-phobert \
    --out-dir reports/text_features/phobert
else
  echo "skipping PhoBERT arm: torch/transformers not installed" >&2
fi

echo "done" >&2
