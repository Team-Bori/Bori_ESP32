#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <firmware_url>"
  echo "Example: $0 /dev/ttyUSB0 https://github.com/USER/REPO/blob/main/firmware/mlp.bin"
  echo "GitHub file links (github.com/.../blob/...) are converted to raw links automatically."
  exit 1
fi

source "$(dirname "$0")/common.sh"

PORT="$1"
URL="$2"
OUT="/tmp/mlp_firmware.bin"

load_config
require_tools curl python3
find_esptool

download "$URL" "$OUT" "$MAX_FIRMWARE_BYTES"
# The merged image is written from 0x0; anything past 0x1E0000 would overwrite the model.
check_size "$OUT" "$MAX_FIRMWARE_BYTES" "Firmware"

echo "Flashing firmware to $PORT ..."
flash_image "$PORT" 0x0 "$OUT"

# Firmware-only update: the model partition is untouched, so a missing model is not an error here.
verify_boot "$PORT" --allow-no-model
echo "Firmware flash complete."
