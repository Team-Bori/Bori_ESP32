#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <serial_port> <firmware_url> <model_url>"
  echo "Example:"
  echo "  $0 /dev/ttyUSB0 \\\n    https://raw.githubusercontent.com/USER/REPO/main/firmware/mlp.bin \\\n    https://raw.githubusercontent.com/USER/REPO/main/models/mlp/default_model.bin"
  exit 1
fi

PORT="$1"
FIRMWARE_URL="$2"
MODEL_URL="$3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/flash_firmware_from_github.sh" "$PORT" "$FIRMWARE_URL"
"$SCRIPT_DIR/flash_model_from_github.sh" "$PORT" "$MODEL_URL"

echo "Firmware + model update complete."
