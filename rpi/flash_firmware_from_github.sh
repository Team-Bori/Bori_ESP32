#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <firmware_url>"
  echo "Example: $0 /dev/ttyUSB0 https://github.com/USER/REPO/blob/main/firmware/tflm_runtime.bin"
  echo "GitHub file links (github.com/.../blob/...) are converted to raw links automatically."
  exit 1
fi

source "$(dirname "$0")/common.sh"

PORT="$1"
URL="$2"
OUT="/tmp/esp32_firmware.bin"

load_config
require_tools curl python3
find_esptool

download "$URL" "$OUT" "$MAX_FIRMWARE_BYTES"
# The merged image is written from 0x0; anything past the model partition start would overwrite it.
check_size "$OUT" "$MAX_FIRMWARE_BYTES" "Firmware"
validate_firmware "$OUT"

echo "Flashing $FIRMWARE_ID firmware to $PORT ..."
flash_image "$PORT" 0x0 "$OUT"

# Firmware-only update: the model partition is untouched and may hold a package for
# another firmware, so a missing model is not an error here.
verify_boot "$PORT" --allow-no-model --expect-firmware "$FIRMWARE_ID"
echo "Firmware flash complete ($FIRMWARE_ID)."
