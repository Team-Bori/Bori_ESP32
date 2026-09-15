#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <model_file>"
  echo "Example: $0 /dev/ttyUSB0 /home/pi/uploaded/model.bin"
  exit 1
fi

PORT="$1"
MODEL="$2"
MODEL_OFFSET="0x1E0000"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
command -v esptool >/dev/null || { echo "esptool is required"; exit 1; }

[[ -f "$MODEL" ]] || { echo "Model file not found: $MODEL"; exit 1; }

python3 "$PROJECT_ROOT/pc/validate_model.py" "$MODEL"

echo "Flashing validated model to $PORT at $MODEL_OFFSET ..."
esptool --chip esp32 -p "$PORT" -b 921600 \
  write-flash --flash-mode dio --flash-size detect "$MODEL_OFFSET" "$MODEL"

echo "Model flash complete."
