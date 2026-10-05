#!/usr/bin/env bash
set -euo pipefail

idf.py build
idf.py merge-bin -o build/mlp.bin -f raw

mkdir -p ../../firmware
cp build/mlp.bin ../../firmware/mlp.bin
if [[ -f ../../models/mlp/default_model.bin ]]; then
  cp ../../models/mlp/default_model.bin ../../firmware/model.bin
fi
sha256sum ../../firmware/mlp.bin | tee ../../firmware/mlp.bin.sha256
if [[ -f ../../firmware/model.bin ]]; then
  sha256sum ../../firmware/model.bin | tee ../../firmware/model.bin.sha256
fi

echo "Firmware release created: ../../firmware/mlp.bin"
python ../../pc/validate_firmware.py ../../firmware/mlp.bin
echo "Model image: ../../firmware/model.bin (flash at 0x200000)."
