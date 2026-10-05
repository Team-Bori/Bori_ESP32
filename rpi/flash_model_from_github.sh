#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <model_url>"
  echo "Example: $0 /dev/ttyUSB0 https://github.com/USER/REPO/blob/main/models/sine_regression/default_model.bin"
  echo "GitHub file links (github.com/.../blob/...) are converted to raw links automatically."
  echo "MLP1 models need the mlp firmware, BTF1 packages need tflm_runtime (checked before writing)."
  exit 1
fi

source "$(dirname "$0")/common.sh"

PORT="$1"
URL="$2"
OUT="/tmp/esp32_model.bin"

load_config
require_tools curl python3
find_esptool

download "$URL" "$OUT" "$MAX_MODEL_BYTES"
check_size "$OUT" "$MAX_MODEL_BYTES" "Model"

echo "Validating downloaded model..."
validate_model "$OUT"

BOARD_FW="$(board_firmware "$PORT")" \
  || die "the board on $PORT did not answer; flash a firmware first (flash_firmware_from_github.sh)"
require_compatible "$MODEL_FIRMWARE" "$BOARD_FW"

echo "Flashing model partition at $MODEL_OFFSET ..."
flash_image "$PORT" "$MODEL_OFFSET" "$OUT"

verify_boot "$PORT" --expect-checksum "$MODEL_CHECKSUM"
echo "Model deploy complete (checksum $MODEL_CHECKSUM)."
