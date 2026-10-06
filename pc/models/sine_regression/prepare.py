#!/usr/bin/env python3
"""Prepares models/sine_regression from the TFLite Micro hello_world example.

- downloads hello_world_int8.tflite at a pinned tflite-micro commit and checks its sha256
- writes demo/eval samples (int8 inputs, float sin(x) targets)

Then: python pc/bori_package.py build models/sine_regression/manifest.json
"""
import hashlib
import math
import urllib.request
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import btf_format as bf  # noqa: E402

COMMIT = "22c2469a233981012ac16ad849f59c7105655aa9"
URL = ("https://raw.githubusercontent.com/tensorflow/tflite-micro/"
       f"{COMMIT}/tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite")
SHA256 = "505ee4fae7fa46ab67bea4c08b4969eb3eb8b9114c50595ec4a29d9a27993202"

EVAL_SAMPLES = 64

OUT = Path(__file__).resolve().parents[3] / "models" / "sine_regression"


def quantize(x, t):
    return np.clip(np.round(x / t.scale) + t.zero_point, -128, 127).astype(np.int8)


def dequantize(q, t):
    return (q.astype(np.float32) - t.zero_point) * np.float32(t.scale)


def main():
    tfl = OUT / "model.tflite"
    if not tfl.exists():
        print(f"downloading {URL}")
        tfl.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
    digest = hashlib.sha256(tfl.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch for {tfl}: {digest}")

    t = bf.tflite_info(tfl.read_bytes())["inputs"][0][0]
    samples = OUT / "samples"
    samples.mkdir(exist_ok=True)

    # Demo: x = pi/2 -> sin = 1. Targets use the dequantized x actually fed to the model.
    demo_q = quantize(np.array([[math.pi / 2]], dtype=np.float32), t)
    np.save(samples / "demo_input.npy", demo_q)
    np.save(samples / "demo_expected.npy", np.sin(dequantize(demo_q, t)).reshape(1, 1))

    # Eval: evenly spaced over [0, 2*pi) (the training range of hello_world).
    xs = np.linspace(0, 2 * math.pi, EVAL_SAMPLES, endpoint=False, dtype=np.float32)
    q = quantize(xs, t).reshape(-1, 1, 1)
    np.save(samples / "eval_inputs.npy", q)
    np.save(samples / "eval_expected.npy", np.sin(dequantize(q, t)).reshape(-1, 1))
    print(f"wrote samples to {samples} (eval {EVAL_SAMPLES})")


if __name__ == "__main__":
    main()
