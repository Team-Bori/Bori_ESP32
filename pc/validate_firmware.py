#!/usr/bin/env python3
"""Identifies a merged firmware image (written at 0x0) before it is flashed.

Prints:
  VALID
  firmware=<mlp|tflm_runtime>   (from the app's project_name)
  version=<app version>
  model_offset=0x200000 model_bytes=2097152   (from the image's partition table)

Fails if the image is not a merged ESP32 image or its partition table does not
match the layout the deploy scripts expect (rpi/common.sh).
"""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import btf_format as bf  # noqa: E402

BOOTLOADER_OFFSET = 0x1000
PARTITION_TABLE_OFFSET = 0x8000
APP_OFFSET = 0x10000
APP_DESC_MAGIC = 0xABCD5432
PROJECT_TO_FIRMWARE = {"mlp_esp32": "mlp", "tflm_runtime": "tflm_runtime"}


def identify(data: bytes) -> dict:
    if len(data) < APP_OFFSET + 256:
        raise ValueError("image too small to be a merged firmware image (write it at 0x0)")
    if data[BOOTLOADER_OFFSET] != 0xE9 or data[APP_OFFSET] != 0xE9:
        raise ValueError("no ESP image at 0x1000/0x10000 (expected `idf.py merge-bin` output)")

    desc = APP_OFFSET + 24 + 8  # image header + first segment header
    magic, = struct.unpack_from("<I", data, desc)
    if magic != APP_DESC_MAGIC:
        raise ValueError("app description not found")
    version = data[desc + 16:desc + 48].split(b"\0")[0].decode("ascii", "replace")
    project = data[desc + 48:desc + 80].split(b"\0")[0].decode("ascii", "replace")

    model = None
    for i in range(95):
        off = PARTITION_TABLE_OFFSET + 32 * i
        entry = data[off:off + 32]
        if entry[:2] != b"\xaa\x50":
            break
        ptype, subtype, p_off, p_size = struct.unpack_from("<BBII", entry, 2)
        label = entry[12:28].split(b"\0")[0].decode("ascii", "replace")
        if label == "model" and ptype == 1 and subtype == 0x40:
            model = (p_off, p_size)
    if model is None:
        raise ValueError("partition table has no 'model' data partition")
    if model != (bf.MODEL_PARTITION_OFFSET, bf.MODEL_PARTITION_BYTES):
        raise ValueError(f"model partition at 0x{model[0]:X} ({model[1]} bytes); this repository expects "
                         f"0x{bf.MODEL_PARTITION_OFFSET:X} ({bf.MODEL_PARTITION_BYTES} bytes). "
                         "Rebuild the firmware with the current partitions.csv")
    if len(data) > bf.MODEL_PARTITION_OFFSET:
        raise ValueError(f"image is {len(data)} bytes and would overwrite the model partition")
    return {"firmware": PROJECT_TO_FIRMWARE.get(project, project), "project": project,
            "version": version, "model_offset": model[0], "model_bytes": model[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    args = ap.parse_args()
    try:
        info = identify(args.image.read_bytes())
    except (OSError, ValueError) as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    print("VALID")
    print(f"firmware={info['firmware']}")
    print(f"version={info['version']}")
    print(f"model_offset=0x{info['model_offset']:X} model_bytes={info['model_bytes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
