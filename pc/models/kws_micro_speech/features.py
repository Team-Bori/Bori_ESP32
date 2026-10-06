#!/usr/bin/env python3
"""Host feature extraction for micro_speech: 1 s of 16 kHz audio -> 49 x 40 int8.

Uses the original example's own audio preprocessor model
(models/audio_preprocessor_int8.tflite, Signal library ops) on the TFLM Python
interpreter, frame by frame exactly like the example (evaluate.py / micro_speech_test.cc):
30 ms window (480 samples), 20 ms stride (320 samples), preprocessor state reset per clip.

  python features.py --verify      # bit-exact check against the example's test vectors
  python features.py some.wav      # print the 49x40 feature

Needs `tflite-micro` (Linux x86_64; on Windows run inside WSL).
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kws_common as kc  # noqa: E402

# Expected first-frame features from micro_speech_test.cc (NoFeatureTest / YesFeatureTest)
# at the pinned commit. Non-HiFi builds compare them with EXPECT_EQ (bit exact).
EXPECTED_30MS = {
    "no_30ms": [126, 103, 124, 102, 124, 102, 123, 100, 118, 97, 118, 100, 118, 98,
                121, 100, 121, 98, 117, 91, 96, 74, 54, 87, 100, 87, 109, 92,
                91, 80, 64, 55, 83, 74, 74, 78, 114, 95, 101, 81],
    "yes_30ms": [124, 105, 126, 103, 125, 101, 123, 100, 116, 98, 115, 97, 113, 90,
                 91, 82, 104, 96, 117, 97, 121, 103, 126, 101, 125, 104, 126, 104,
                 125, 101, 116, 90, 81, 74, 80, 71, 83, 76, 82, 71],
}
# Expected labels of the 1 s clips (micro_speech_test.cc TestAudioSample calls).
EXPECTED_1000MS = {"yes_1000ms": "yes", "no_1000ms": "no",
                   "silence_1000ms": "silence", "noise_1000ms": "silence"}


def read_wav(path: Path, pad_to: int | None = kc.CLIP_SAMPLES) -> np.ndarray:
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (kc.SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: need 16 kHz mono 16-bit PCM")
        s = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if pad_to is not None:
        s = np.pad(s[:pad_to], (0, max(0, pad_to - len(s))))
    return s.astype(np.int16)


class FeatureExtractor:
    def __init__(self):
        from tflite_micro.python.tflite_micro import runtime
        self._pp = runtime.Interpreter.from_file(str(kc.fetch_original("models/audio_preprocessor_int8.tflite")))

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """All complete frames of `samples` -> [frames, 40] int8 (49 frames for 1 s)."""
        self._pp.reset()  # noise estimate and gain state start fresh for every clip
        frames = []
        for start in range(0, len(samples) - kc.WINDOW_SAMPLES + 1, kc.STRIDE_SAMPLES):
            self._pp.set_input(samples[start:start + kc.WINDOW_SAMPLES].reshape(1, -1), 0)
            self._pp.invoke()
            frames.append(np.asarray(self._pp.get_output(0)).reshape(-1))
        return np.asarray(frames, dtype=np.int8)


def verify() -> bool:
    from tflite_micro.python.tflite_micro import runtime

    fx = FeatureExtractor()
    ok = True
    for name, expected in EXPECTED_30MS.items():
        f = fx(read_wav(kc.fetch_original(f"testdata/{name}.wav"), pad_to=None))
        same = f.shape == (1, kc.FEATURE_SIZE) and f[0].tolist() == expected
        diff = int(np.abs(f[0].astype(int) - np.array(expected)).max()) if f.shape[0] else None
        print(f"{name}: frames {f.shape[0]}, bit-exact {same} (max diff {diff})")
        ok &= same
    model = runtime.Interpreter.from_file(str(kc.fetch_original("models/micro_speech_quantized.tflite")))
    for name, label in EXPECTED_1000MS.items():
        f = fx(read_wav(kc.fetch_original(f"testdata/{name}.wav"), pad_to=None))
        model.set_input(f.reshape(1, -1), 0)
        model.invoke()
        got = kc.LABELS[int(np.asarray(model.get_output(0)).argmax())]
        print(f"{name}: frames {f.shape[0]}, predicted {got}, expected {label}")
        ok &= f.shape == (kc.FEATURE_COUNT, kc.FEATURE_SIZE) and got == label
    print("VERIFY", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", type=Path)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        sys.exit(0 if verify() else 1)
    if args.wav:
        np.set_printoptions(linewidth=200, threshold=10000)
        print(FeatureExtractor()(read_wav(args.wav)))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
