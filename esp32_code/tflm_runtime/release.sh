#!/usr/bin/env bash
# Builds tflm_runtime and copies the merged image (written at 0x0) to firmware/.
set -euo pipefail

idf.py build
# Not build/tflm_runtime.bin: that is the app image idf.py build writes (merge-bin reads it).
idf.py merge-bin -o build/tflm_runtime_merged.bin -f raw

mkdir -p ../../firmware
cp build/tflm_runtime_merged.bin ../../firmware/tflm_runtime.bin
sha256sum ../../firmware/tflm_runtime.bin | tee ../../firmware/tflm_runtime.bin.sha256
python ../../pc/validate_firmware.py ../../firmware/tflm_runtime.bin

echo "Firmware release created: ../../firmware/tflm_runtime.bin"
echo "Default package: ../../models/sine_regression/default_model.bin (flash at 0x200000)."
