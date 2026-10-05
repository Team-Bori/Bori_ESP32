#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <serial_port> <model_file>"
  echo "Example: $0 /dev/ttyUSB0 /home/pi/uploaded/default_model.bin"
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

validate_model "$MODEL"

BOARD_FW="$(board_firmware "$PORT")" \
  || die "the board on $PORT did not answer; flash a firmware first (flash_firmware_from_github.sh)"
require_compatible "$MODEL_FIRMWARE" "$BOARD_FW"

echo "Flashing validated model to $PORT at $MODEL_OFFSET ..."
flash_image "$PORT" "$MODEL_OFFSET" "$MODEL"

verify_boot "$PORT" --expect-checksum "$MODEL_CHECKSUM"
echo "Model deploy complete (checksum $MODEL_CHECKSUM)."
