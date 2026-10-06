"""Validates a model file before it is written to the model partition.

Recognises both formats by their magic:
  MLP1  64-16-10 MLP for the `mlp` firmware            (pc/model_format.py)
  BTF1  TFLite Micro package for `tflm_runtime`         (pc/btf_format.py, docs/PACKAGE_FORMAT.md)

Output (parsed by rpi/common.sh): VALID, format=, firmware=, size=, checksum=0x........
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import btf_format as bf  # noqa: E402
from model_format import read_model_bin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    args = ap.parse_args()
    data = args.model.read_bytes()
    magic = data[:4]

    if magic == b"MLP1":
        info = read_model_bin(args.model)
        print("VALID")
        print("format=MLP1")
        print("firmware=mlp")
        print(f"size={info['size']}")
        print(f"payload_size={info['payload_size']}")
        print(f"dims={info['dims'][0]}->{info['dims'][1]}->{info['dims'][2]}")
        print(f"checksum=0x{info['checksum']:08x}")
    elif magic == bf.MAGIC:
        try:
            pkg = bf.parse(data)
        except bf.PackageError as e:
            print(f"INVALID {e.code}: {e}", file=sys.stderr)
            sys.exit(1)
        print("VALID")
        print("format=BTF1")
        print("firmware=tflm_runtime")
        print(f"size={pkg.total_size}")
        print(f"model_id={pkg.model_id}")
        print(f"task={pkg.task}")
        print(f"checksum=0x{pkg.checksum:08x}")
    else:
        print(f"INVALID: unknown model format (magic {magic!r}; expected MLP1 or BTF1)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
