#!/usr/bin/env bash
# Shared helpers for the ESP32 flash scripts. Source this file; do not run it.

RPI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RPI_DIR/.." && pwd)"

# Partition layout (esp32_code/mlp_esp32/partitions.csv)
MODEL_OFFSET="0x1E0000"
MODEL_PARTITION_BYTES=131072      # 0x20000
FIRMWARE_LIMIT_BYTES=1966080      # 0x1E0000: merged image must end before the model partition

die() {
  echo "ERROR: $*" >&2
  exit 1
}

load_config() {
  local conf="${ESP32_CONF:-$RPI_DIR/esp32.conf}"
  [[ -f "$conf" ]] || die "config not found: $conf"
  # shellcheck source=esp32.conf
  source "$conf"

  # A per-board config may set only some keys; fall back to the hardware limits.
  : "${MAX_MODEL_BYTES:=$MODEL_PARTITION_BYTES}" "${MAX_FIRMWARE_BYTES:=$FIRMWARE_LIMIT_BYTES}"

  # Admin limits can only tighten the hardware limits, never exceed them.
  if (( MAX_MODEL_BYTES > MODEL_PARTITION_BYTES )); then MAX_MODEL_BYTES=$MODEL_PARTITION_BYTES; fi
  if (( MAX_FIRMWARE_BYTES > FIRMWARE_LIMIT_BYTES )); then MAX_FIRMWARE_BYTES=$FIRMWARE_LIMIT_BYTES; fi
  : "${FLASH_BAUD:=921600}" "${BOOT_TIMEOUT:=15}" "${BOOT_SETTLE:=3}"
  : "${DEFAULT_MODEL:=$PROJECT_ROOT/models/mlp/default_model.bin}"
  : "${DEFAULT_FIRMWARE:=$PROJECT_ROOT/firmware/mlp.bin}"
}

require_tools() {
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null || die "$tool is required"
  done
}

# esptool v5 installs `esptool`, v4 installs `esptool.py`.
# The underscore sub-commands/options below are accepted by both.
find_esptool() {
  if command -v esptool >/dev/null; then
    ESPTOOL=(esptool)
  elif command -v esptool.py >/dev/null; then
    ESPTOOL=(esptool.py)
  elif python3 -c "import esptool" 2>/dev/null; then
    ESPTOOL=(python3 -m esptool)
  else
    die "esptool is required (pip install esptool)"
  fi
}

# Turn GitHub page links into direct download links.
#   https://github.com/U/R/blob/BRANCH/path -> https://raw.githubusercontent.com/U/R/BRANCH/path
#   https://github.com/U/R/raw/BRANCH/path  -> same
# raw.githubusercontent.com and release-asset links are returned unchanged.
to_raw_url() {
  local url="$1"
  [[ "$url" =~ ^https:// ]] || die "only https:// links are allowed: $url"

  if [[ "$url" =~ ^https://github\.com/([^/]+)/([^/]+)/(blob|raw)/(.+)$ ]]; then
    local path="${BASH_REMATCH[4]%%\?*}"
    echo "https://raw.githubusercontent.com/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}/${path}"
  elif [[ "$url" =~ ^https://github\.com/[^/]+/[^/]+/?$ ]] || \
       [[ "$url" =~ ^https://github\.com/[^/]+/[^/]+/tree/ ]]; then
    die "this is a repository/folder link, not a file link. Open the .bin file on GitHub and copy that link: $url"
  else
    echo "$url"
  fi
}

file_size() {
  stat -c%s "$1"
}

check_size() {
  local file="$1" max="$2" label="$3"
  local size
  size="$(file_size "$file")"
  (( size > 0 )) || die "$label is empty: $file"
  (( size <= max )) || die "$label is too large: $size bytes (limit $max bytes)"
  echo "$label size: $size bytes (limit $max bytes)"
}

# download <url> <output> <max_bytes>
download() {
  local url out max
  url="$(to_raw_url "$1")"
  out="$2"
  max="$3"
  echo "Downloading $url"
  rm -f "$out"
  # --max-filesize stops early when the server reports the size; check_size covers the rest.
  curl -fL --retry 3 --max-filesize "$max" "$url" -o "$out" \
    || die "download failed (missing file, private repo, or larger than $max bytes): $url"
}

# Validates model.bin and prints its checksum (0x........) on stdout.
validate_model() {
  local model="$1" output
  output="$(python3 "$PROJECT_ROOT/pc/validate_model.py" "$model" 2>&1)" \
    || die "invalid model file: $output"
  echo "$output" >&2
  echo "$output" | sed -n 's/^checksum=//p'
}

flash_image() {
  local port="$1" offset="$2" file="$3"
  "${ESPTOOL[@]}" --chip esp32 -p "$port" -b "$FLASH_BAUD" \
    write_flash --flash_mode dio --flash_size detect "$offset" "$file"
}

erase_model_partition() {
  local port="$1"
  echo "Erasing the whole model partition ($MODEL_OFFSET, $MODEL_PARTITION_BYTES bytes) ..."
  "${ESPTOOL[@]}" --chip esp32 -p "$port" -b "$FLASH_BAUD" \
    erase_region "$MODEL_OFFSET" "$MODEL_PARTITION_BYTES"
}

# check_boot <port> [extra esp_monitor verify args...]; returns non-zero on failure.
check_boot() {
  local port="$1"
  shift
  echo "Verifying boot on $port ..."
  python3 "$RPI_DIR/esp_monitor.py" "$port" verify \
    --timeout "$BOOT_TIMEOUT" --settle "$BOOT_SETTLE" "$@"
}

# verify_boot <port> [extra esp_monitor verify args...]; exits on failure.
verify_boot() {
  check_boot "$@" || die "board did not pass the boot check (see the verify message above)"
  echo "Boot check passed."
}
