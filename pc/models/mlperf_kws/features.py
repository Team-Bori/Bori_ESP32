#!/usr/bin/env python3
"""Host MFCC features for the MLPerf Tiny keyword-spotting model (DS-CNN).

Same TensorFlow ops as the reference (mlcommons/tiny benchmark/training/keyword_spotting/
get_dataset.py, feature_type "mfcc", evaluation path):
  int16 audio -> float32 / reduce_max(audio)  (the largest sample value, not the abs max)
  zero pad to 16,000 samples (1 s at 16 kHz); the reference's 2-sample time-shift pad/slice is a no-op
  tf.signal.stft(frame_length 480 = 30 ms, frame_step 320 = 20 ms, fft_length None -> 512, Hann)
  |STFT| -> 40 mel bins 20-4000 Hz (tf.signal.linear_to_mel_weight_matrix) -> log(x + 1e-6)
  tf.signal.mfccs_from_log_mel_spectrograms -> first 10 coefficients -> [49, 10, 1]
Quantization like the reference eval_quantized_model.py / make_bin_files.py:
  np.array(x / scale + zero_point, dtype=np.int8)  -> truncation toward zero
(here clipped to -128..127 first, so out-of-range values do not wrap).

  python features.py --verify     # byte-compare with EEMBC's official kws01 benchmark inputs
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import wave
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

CACHE = Path(os.environ.get("BORI_CACHE", Path.home() / ".cache" / "bori"))
TEST_SET = CACHE / "speech_commands" / "test_v0.02"
EEMBC_KWS01 = CACHE / "eembc_runner" / "datasets" / "kws01"

SAMPLE_RATE = 16000
DESIRED_SAMPLES = 16000
WINDOW = 480
STRIDE = 320
NUM_MEL = 40
DCT = 10
FRAMES = 49

# Model input quantization (kws_ref_model.tflite at the pinned commit).
INPUT_SCALE = 0.5847029089927673
INPUT_ZERO_POINT = 83

# Model output order (get_dataset.py word_labels) and the test-set folder of each.
LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "silence", "unknown"]
FOLDER_TO_LABEL = {f: i for i, f in enumerate(LABELS[:10])}
FOLDER_TO_LABEL.update({"_silence_": 10, "_unknown_": 11})


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: need 16 kHz mono 16-bit PCM")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


class Mfcc:
    def __init__(self):
        import tensorflow as tf

        self.tf = tf
        self.mel = tf.signal.linear_to_mel_weight_matrix(NUM_MEL, 257, SAMPLE_RATE, 20.0, 4000.0)

    def __call__(self, audio_int16: np.ndarray) -> np.ndarray:
        tf = self.tf
        x = tf.cast(tf.constant(audio_int16), tf.float32)
        x = x / tf.reduce_max(x)
        x = tf.pad(x, [[0, DESIRED_SAMPLES - tf.shape(x)[-1]]])
        stfts = tf.signal.stft(x, frame_length=WINDOW, frame_step=STRIDE, fft_length=None,
                               window_fn=tf.signal.hann_window)
        mel = tf.tensordot(tf.abs(stfts), self.mel, 1)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel + 1e-6))[..., :DCT]
        return tf.reshape(mfcc, [FRAMES, DCT, 1]).numpy()


def quantize(mfcc: np.ndarray) -> np.ndarray:
    v = mfcc / INPUT_SCALE + INPUT_ZERO_POINT
    return np.trunc(np.clip(v, -128, 127)).astype(np.int8)


def test_clips():
    files = sorted(TEST_SET.glob("*/*.wav"))
    if not files:
        raise SystemExit(f"no Speech Commands test set in {TEST_SET} (see pc/models/kws_micro_speech/prepare.py)")
    return files


def all_features():
    """Float MFCCs of every test clip, cached: (mfcc [N, 49, 10, 1] float32, folders, names)."""
    files = test_clips()
    names = np.array([f"{p.parent.name}/{p.name}" for p in files])
    cache = CACHE / "speech_commands" / "test_v0.02_mlperf_mfcc.npz"
    if cache.exists():
        z = np.load(cache)
        if z["names"].tolist() == names.tolist():
            return z["mfcc"], np.array([n.split("/")[0] for n in names]), names
    fx = Mfcc()
    mfcc = np.stack([fx(read_wav(p)) for p in files]).astype(np.float32)
    np.savez(cache, mfcc=mfcc, names=names)
    return mfcc, np.array([n.split("/")[0] for n in names]), names


def verify() -> bool:
    """Every EEMBC kws01 input must equal (byte for byte) our features of some test clip
    with the same label."""
    bins = sorted(EEMBC_KWS01.glob("*.bin"))
    if not bins:
        raise SystemExit(f"EEMBC kws01 inputs not found in {EEMBC_KWS01}")
    mfcc, folders, _ = all_features()
    ours = {}
    for m, folder in zip(mfcc, folders):
        q = quantize(m)
        ours.setdefault(hashlib.sha1(q.tobytes()).hexdigest(), set()).add(FOLDER_TO_LABEL.get(folder))
    exact = label_ok = 0
    for b in bins:
        h = hashlib.sha1(b.read_bytes()).hexdigest()
        label = int(b.stem.rsplit("_", 1)[1])
        if h in ours:
            exact += 1
            label_ok += label in ours[h]
    print(f"EEMBC kws01: {len(bins)} inputs, byte-identical to one of our test-clip features: {exact}, "
          f"with the same label: {label_ok}")
    return exact == len(bins) and label_ok == len(bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        sys.exit(0 if verify() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
