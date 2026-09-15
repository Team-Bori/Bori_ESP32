#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <model_url>"
  echo "Example: $0 /dev/ttyUSB0 https://raw.githubusercontent.com/USER/REPO/main/models/mlp/default_model.bin"
  exit 1
fi

PORT="$1"
URL="$2"
OUT="/tmp/mlp_model.bin"
MODEL_OFFSET="0x1E0000"

command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v esptool >/dev/null || { echo "esptool is required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

curl -fL --retry 3 "$URL" -o "$OUT"

echo "Validating downloaded model..."
python3 "$PROJECT_ROOT/pc/validate_model.py" "$OUT"

echo "Downloaded model: $(stat -c%s "$OUT") bytes"
echo "Flashing model partition at $MODEL_OFFSET ..."

esptool --chip esp32 -p "$PORT" -b 921600 \
  write-flash --flash-mode dio --flash-size detect "$MODEL_OFFSET" "$OUT"

echo "Model flash complete. Reboot the ESP32 and run the benchmark."
