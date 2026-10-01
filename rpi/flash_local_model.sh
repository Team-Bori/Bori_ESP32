#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <model_file>"
  echo "Example: $0 /dev/ttyUSB0 /home/pi/uploaded/model.bin"
  exit 1
fi

source "$(dirname "$0")/common.sh"

PORT="$1"
MODEL="$2"

load_config
require_tools python3
find_esptool

[[ -f "$MODEL" ]] || die "Model file not found: $MODEL"
check_size "$MODEL" "$MAX_MODEL_BYTES" "Model"

CHECKSUM="$(validate_model "$MODEL")" || exit 1

echo "Flashing validated model to $PORT at $MODEL_OFFSET ..."
flash_image "$PORT" "$MODEL_OFFSET" "$MODEL"

verify_boot "$PORT" --expect-checksum "$CHECKSUM"
echo "Model deploy complete (checksum $CHECKSUM)."
