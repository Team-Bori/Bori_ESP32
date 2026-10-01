#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <model_url>"
  echo "Example: $0 /dev/ttyUSB0 https://github.com/USER/REPO/blob/main/models/mlp/default_model.bin"
  echo "GitHub file links (github.com/.../blob/...) are converted to raw links automatically."
  exit 1
fi

source "$(dirname "$0")/common.sh"

PORT="$1"
URL="$2"
OUT="/tmp/mlp_model.bin"

load_config
require_tools curl python3
find_esptool

download "$URL" "$OUT" "$MAX_MODEL_BYTES"
check_size "$OUT" "$MAX_MODEL_BYTES" "Model"

echo "Validating downloaded model..."
CHECKSUM="$(validate_model "$OUT")" || exit 1

echo "Flashing model partition at $MODEL_OFFSET ..."
flash_image "$PORT" "$MODEL_OFFSET" "$OUT"

verify_boot "$PORT" --expect-checksum "$CHECKSUM"
echo "Model deploy complete (checksum $CHECKSUM)."
