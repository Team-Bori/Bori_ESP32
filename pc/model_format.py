from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"MLP1"
VERSION = 1
INPUT_SIZE = 64
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10

# 4s + I + 4H + 4f + 5I = 52 bytes
HEADER_FORMAT = "<4sIHHHHffffIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def fnv1a32(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def build_model_bin(
    path: str | Path,
    qw1,
    qb1,
    qw2,
    qb2,
    input_scale: float,
    w1_scale: float,
    hidden_scale: float,
    w2_scale: float,
) -> int:
    payload = (
        qw1.astype("<i1").tobytes()
        + qb1.astype("<i4").tobytes()
        + qw2.astype("<i1").tobytes()
        + qb2.astype("<i4").tobytes()
    )

    expected = (
        INPUT_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE * 4
        + HIDDEN_SIZE * OUTPUT_SIZE
        + OUTPUT_SIZE * 4
    )
    if len(payload) != expected:
        raise ValueError(f"Unexpected payload size: {len(payload)} != {expected}")

    checksum = fnv1a32(payload)
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        0,
        float(input_scale),
        float(w1_scale),
        float(hidden_scale),
        float(w2_scale),
        qw1.nbytes,
        qb1.nbytes,
        qw2.nbytes,
        qb2.nbytes,
        checksum,
    )

    if len(header) != HEADER_SIZE:
        raise RuntimeError(f"Header size mismatch: {len(header)} != {HEADER_SIZE}")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(header + payload)
    return len(header) + len(payload)


def read_model_bin(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError("model.bin is smaller than the header")

    fields = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    (
        magic,
        version,
        input_size,
        hidden_size,
        output_size,
        _reserved,
        input_scale,
        w1_scale,
        hidden_scale,
        w2_scale,
        w1_bytes,
        b1_bytes,
        w2_bytes,
        b2_bytes,
        checksum,
    ) = fields

    payload = data[HEADER_SIZE:]
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported model format")
    if (input_size, hidden_size, output_size) != (INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE):
        raise ValueError("Model dimensions do not match 64-16-10")
    expected = w1_bytes + b1_bytes + w2_bytes + b2_bytes
    if len(payload) != expected:
        raise ValueError(f"Unexpected payload length: {len(payload)} != {expected}")
    actual_checksum = fnv1a32(payload)
    if actual_checksum != checksum:
        raise ValueError("Model checksum mismatch")

    return {
        "size": len(data),
        "header_size": HEADER_SIZE,
        "payload_size": len(payload),
        "dims": (input_size, hidden_size, output_size),
        "input_scale": input_scale,
        "w1_scale": w1_scale,
        "hidden_scale": hidden_scale,
        "w2_scale": w2_scale,
        "checksum": checksum,
    }
