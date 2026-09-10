#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <firmware_url>"
  echo "Example: $0 /dev/ttyUSB0 https://raw.githubusercontent.com/USER/REPO/main/firmware/mlp.bin"
  exit 1
fi

PORT="$1"
URL="$2"
OUT="/tmp/mlp_firmware.bin"

command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v esptool >/dev/null || { echo "esptool is required"; exit 1; }

curl -fL --retry 3 "$URL" -o "$OUT"

echo "Downloaded firmware: $(stat -c%s "$OUT") bytes"
echo "Flashing firmware to $PORT ..."

esptool --chip esp32 -p "$PORT" -b 921600 \
  write-flash --flash-mode dio --flash-size detect 0x0 "$OUT"

echo "Firmware flash complete."
