"""Image -> person_detection model input (96 x 96 x 1 int8).

Same steps as the original model's conversion (training_a_model.md, representative_dataset_gen):
  PIL image.resize((96, 96)) -> convert('L') -> x = pixel / 127.5 - 1.0   (float32, -1..1)
then quantized with the model input parameters (scale 0.0078431377, zero_point -1):
  q = clip(round(x / scale) + zero_point, -128, 127)

The whole image is resized (aspect ratio not kept), like the conversion script.
`center_crop=True` instead takes the central 87.5% first, like TF-slim's mobilenet_v1
eval preprocessing (only used for the comparison in evaluate.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

INPUT_SCALE = 0.007843137718737125
INPUT_ZERO_POINT = -1
SIZE = 96


def to_input(path: Path, center_crop: bool = False) -> np.ndarray:
    img = Image.open(path)
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if center_crop:
        w, h = img.size
        cw, ch = round(w * 0.875), round(h * 0.875)
        left, top = (w - cw) // 2, (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))
    img = img.resize((SIZE, SIZE)).convert("L")
    x = np.asarray(img, dtype=np.float32) / np.float32(127.5) - np.float32(1.0)
    q = np.round(x / np.float32(INPUT_SCALE)) + INPUT_ZERO_POINT
    return np.clip(q, -128, 127).astype(np.int8).reshape(SIZE, SIZE, 1)
