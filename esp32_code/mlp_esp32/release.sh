#!/usr/bin/env bash
set -euo pipefail

idf.py build
idf.py merge-bin -o build/mlp.bin -f raw

mkdir -p ../firmware
cp build/mlp.bin ../firmware/mlp.bin
sha256sum ../firmware/mlp.bin | tee ../firmware/mlp.bin.sha256

echo "Firmware release created: ../firmware/mlp.bin"
echo "NOTE: model weights are released separately as model.bin."
