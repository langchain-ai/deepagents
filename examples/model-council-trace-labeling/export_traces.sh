#!/usr/bin/env bash
set -euo pipefail

PROJECT=${1:?"usage: export_traces.sh PROJECT [OUTPUT_DIR] [LIMIT]"}
OUTPUT_DIR=${2:-.workspace/traces}
LIMIT=${3:-10}

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
langsmith trace export "$OUTPUT_DIR" \
  --project "$PROJECT" \
  --limit "$LIMIT" \
  --full

echo "Exported up to $LIMIT traces from $PROJECT into $OUTPUT_DIR"
