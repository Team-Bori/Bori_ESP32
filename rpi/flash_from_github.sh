#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <serial_port> <firmware_url> <model_url>"
  echo "Example:"
  echo "  $0 /dev/ttyUSB0 \\"
  echo "    https://github.com/USER/REPO/blob/main/firmware/tflm_runtime.bin \\"
  echo "    https://github.com/USER/REPO/blob/main/models/sine_regression/default_model.bin"
  exit 1
fi

source "$(dirname "$0")/common.sh"

PORT="$1"
FIRMWARE_URL="$2"
MODEL_URL="$3"
FW_OUT="/tmp/esp32_firmware.bin"
MODEL_OUT="/tmp/esp32_model.bin"

load_config
require_tools curl python3
find_esptool

# Download and check both files before touching the board.
download "$FIRMWARE_URL" "$FW_OUT" "$MAX_FIRMWARE_BYTES"
check_size "$FW_OUT" "$MAX_FIRMWARE_BYTES" "Firmware"
validate_firmware "$FW_OUT"

download "$MODEL_URL" "$MODEL_OUT" "$MAX_MODEL_BYTES"
check_size "$MODEL_OUT" "$MAX_MODEL_BYTES" "Model"
validate_model "$MODEL_OUT"

require_compatible "$MODEL_FIRMWARE" "$FIRMWARE_ID"

echo "Flashing $FIRMWARE_ID firmware to $PORT ..."
flash_image "$PORT" 0x0 "$FW_OUT"
echo "Flashing model partition at $MODEL_OFFSET ..."
flash_image "$PORT" "$MODEL_OFFSET" "$MODEL_OUT"

verify_boot "$PORT" --expect-checksum "$MODEL_CHECKSUM" --expect-firmware "$FIRMWARE_ID"
echo "Firmware + model update complete ($FIRMWARE_ID, checksum $MODEL_CHECKSUM)."
