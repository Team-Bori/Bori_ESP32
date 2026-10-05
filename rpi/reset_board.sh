#!/usr/bin/env bash
# Session end (spec 1.8): return the board to its default state for the next user.
#   1. erase the whole model partition so no part of the previous user's model remains
#   2. write the default model (DEFAULT_MODEL in esp32.conf)
#   3. verify the board boots the default firmware and runs the default model
#   4. if that fails (e.g. the previous user flashed another or broken firmware),
#      re-flash the default firmware (DEFAULT_FIRMWARE) and try once more
#
# DEFAULT_FIRMWARE and DEFAULT_MODEL must belong together (mlp + MLP1 model, or
# tflm_runtime + BTF1 package); this is checked before anything is written.
set -euo pipefail

usage() {
  echo "Usage: $0 <serial_port> [--force-firmware]"
  echo "  --force-firmware  always re-flash the default firmware as well"
  exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || usage
FORCE_FIRMWARE=0
if [[ $# -eq 2 ]]; then
  [[ "$2" == "--force-firmware" ]] || usage
  FORCE_FIRMWARE=1
fi

source "$(dirname "$0")/common.sh"

PORT="$1"

load_config
require_tools python3
find_esptool

[[ -f "$DEFAULT_MODEL" ]] || die "default model not found: $DEFAULT_MODEL"
[[ -f "$DEFAULT_FIRMWARE" ]] || die "default firmware not found: $DEFAULT_FIRMWARE"
check_size "$DEFAULT_FIRMWARE" "$MAX_FIRMWARE_BYTES" "Default firmware"
validate_firmware "$DEFAULT_FIRMWARE"
validate_model "$DEFAULT_MODEL"
require_compatible "$MODEL_FIRMWARE" "$FIRMWARE_ID"

restore_firmware() {
  echo "Re-flashing default firmware: $DEFAULT_FIRMWARE ($FIRMWARE_ID)"
  flash_image "$PORT" 0x0 "$DEFAULT_FIRMWARE"
}

restore_model() {
  erase_model_partition "$PORT"
  echo "Writing default model: $DEFAULT_MODEL"
  flash_image "$PORT" "$MODEL_OFFSET" "$DEFAULT_MODEL"
}

if (( FORCE_FIRMWARE )); then
  restore_firmware
fi
restore_model

# --expect-firmware catches a board left on another firmware by the previous user.
if check_boot "$PORT" --expect-checksum "$MODEL_CHECKSUM" --expect-firmware "$FIRMWARE_ID"; then
  echo "Board reset complete ($FIRMWARE_ID, default model $MODEL_CHECKSUM)."
  exit 0
fi

if (( FORCE_FIRMWARE )); then
  die "board still fails the boot check with the default firmware; check the hardware"
fi

echo "Boot check failed; recovering with the default firmware ..."
restore_firmware
verify_boot "$PORT" --expect-checksum "$MODEL_CHECKSUM" --expect-firmware "$FIRMWARE_ID"
echo "Board reset complete after firmware recovery ($FIRMWARE_ID, default model $MODEL_CHECKSUM)."
