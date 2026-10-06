"""Shared settings for the micro_speech scripts (prepare.py, features.py, evaluate.py).

The original example is pinned to one tflite-micro commit; its files are downloaded
from that commit and checked by sha256.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

TFLM_COMMIT = "22c2469a233981012ac16ad849f59c7105655aa9"
EXAMPLE = "tensorflow/lite/micro/examples/micro_speech"
RAW = f"https://raw.githubusercontent.com/tensorflow/tflite-micro/{TFLM_COMMIT}/{EXAMPLE}"

# file -> sha256 at TFLM_COMMIT
ORIGINAL_FILES = {
    "models/micro_speech_quantized.tflite": "09e5e2a9dfb2d8ed78802bf18ce297bff54281a66ca18e0c23d69ca14f822a83",
    "models/audio_preprocessor_int8.tflite": "278949d197166fb8b580c0bdc94e902fb709fec0569dcf5766816b28285440e5",
    "testdata/yes_30ms.wav": None,
    "testdata/no_30ms.wav": None,
    "testdata/yes_1000ms.wav": None,
    "testdata/no_1000ms.wav": None,
    "testdata/silence_1000ms.wav": None,
    "testdata/noise_1000ms.wav": None,
}

TEST_SET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
TEST_SET_SHA256 = "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df"

# Downloads (not committed). Set BORI_CACHE to share one cache between Windows and WSL,
# e.g. BORI_CACHE=/mnt/c/Users/<you>/.cache/bori inside WSL.
CACHE = Path(os.environ.get("BORI_CACHE", Path.home() / ".cache" / "bori"))
ORIGINAL_DIR = CACHE / "micro_speech_original"
TEST_SET_DIR = CACHE / "speech_commands" / "test_v0.02"

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "kws_micro_speech_int8"

# Model categories, in the original order (micro_model_settings.h).
LABELS = ["silence", "unknown", "yes", "no"]

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000          # 1 s clip; shorter test clips are zero padded at the end
WINDOW_SAMPLES = 480          # 30 ms
STRIDE_SAMPLES = 320          # 20 ms
FEATURE_COUNT = 49            # frames per clip
FEATURE_SIZE = 40             # channels per frame


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_original(name: str) -> Path:
    """Downloads (once) a file of the pinned original example and checks its hash."""
    dst = ORIGINAL_DIR / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(urllib.request.urlopen(f"{RAW}/{name}", timeout=60).read())
    want = ORIGINAL_FILES.get(name)
    if want and sha256(dst) != want:
        raise SystemExit(f"sha256 mismatch for {dst}")
    return dst


def folder_label(folder: str) -> int:
    """Speech Commands test-set folder -> model category index."""
    if folder == "_silence_":
        return 0
    if folder == "yes":
        return 2
    if folder == "no":
        return 3
    return 1  # _unknown_ and the other 8 command words
