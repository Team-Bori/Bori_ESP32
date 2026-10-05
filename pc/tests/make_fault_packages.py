#!/usr/bin/env python3
"""Writes deliberately broken packages for the board error-handling test.

Output: models/generated/faults/*.bin (git-ignored). Each must make tflm_runtime
report the listed error code and keep running (no reboot):

  bad_checksum.bin     checksum_mismatch    one tflite byte flipped after building
  unsupported_op.bin   unsupported_op       FULLY_CONNECTED patched to SIN (not registered)
  arena_too_small.bin  arena_too_small      arena_bytes = 256; message gives the real need
  arena_too_big.bin    arena_alloc_failed   arena_bytes = 1.5 MB (more than the board has)
  tensor_mismatch.bin  package_invalid      input scale in the package differs from the tflite
  mlp_package.bin      package_invalid      MLP1 package (needs the mlp firmware)
"""
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pc"))

import btf_format as bf  # noqa: E402

OUT = ROOT / "models" / "generated" / "faults"
BASE = ROOT / "models" / "sine_regression" / "default_model.bin"


def rebuild(pkg: bf.Package, **changes) -> bytes:
    for k, v in changes.items():
        setattr(pkg, k, v)
    return bf.build(pkg)


def patch_opcode(tfl: bytes, new_code: int) -> bytes:
    import tflite

    model = tflite.Model.GetRootAsModel(tfl, 0)
    oc = model.OperatorCodes(0)
    out = bytearray(tfl)
    tab = oc._tab
    dep = tab.Offset(4)    # deprecated_builtin_code (int8)
    full = tab.Offset(10)  # builtin_code (int32)
    if not dep and not full:
        raise SystemExit("cannot locate the operator code fields")
    if dep:
        out[tab.Pos + dep] = new_code
    if full:
        struct.pack_into("<i", out, tab.Pos + full, new_code)
    return bytes(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = BASE.read_bytes()

    data = bytearray(base)
    pkg = bf.parse(base)
    tfl_off = next(s["offset"] for s in pkg.sections if s["type"] == "tflite")
    data[tfl_off + 100] ^= 0xFF
    (OUT / "bad_checksum.bin").write_bytes(data)

    sin_code = __import__("tflite").BuiltinOperator.SIN
    (OUT / "unsupported_op.bin").write_bytes(rebuild(bf.parse(base), tflite=patch_opcode(pkg.tflite, sin_code)))
    (OUT / "arena_too_small.bin").write_bytes(rebuild(bf.parse(base), arena_bytes=256))
    (OUT / "arena_too_big.bin").write_bytes(rebuild(bf.parse(base), arena_bytes=1_500_000))
    p = bf.parse(base)
    p.input.scale *= 2
    (OUT / "tensor_mismatch.bin").write_bytes(rebuild(p))
    shutil.copy(ROOT / "models" / "mlp" / "default_model.bin", OUT / "mlp_package.bin")

    for f in sorted(OUT.glob("*.bin")):
        print(f"{f.relative_to(ROOT)}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
