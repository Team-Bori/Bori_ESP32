#!/usr/bin/env python3
"""Host int8 accuracy of micro_speech on the Speech Commands v0.02 test set, and the
package samples for models/kws_micro_speech_int8.

  python evaluate.py              # features for every test clip (cached), accuracy, samples, report

Category mapping of the test-set folders (model order silence, unknown, yes, no):
  _silence_ -> silence, yes -> yes, no -> no, _unknown_ and the other 8 words -> unknown

Runs the original micro_speech_quantized.tflite on the TFLM Python interpreter
(the same reference kernels as `bori_package.py check`). Needs tflite-micro (Linux/WSL).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kws_common as kc  # noqa: E402
from features import FeatureExtractor, read_wav  # noqa: E402

EVAL_PER_CLASS = 40   # 160 eval samples in the package (4 classes x 40)
FEATURE_CACHE = kc.CACHE / "speech_commands" / "test_v0.02_features.npz"


def test_files():
    files = sorted(kc.TEST_SET_DIR.glob("*/*.wav"))
    if not files:
        raise SystemExit(f"no test set in {kc.TEST_SET_DIR}; run prepare.py (set BORI_CACHE in WSL)")
    return files


def load_features(files):
    names = np.array([f"{p.parent.name}/{p.name}" for p in files])
    if FEATURE_CACHE.exists():
        z = np.load(FEATURE_CACHE)
        if z["names"].tolist() == names.tolist():
            return z["features"], names
    fx = FeatureExtractor()
    feats = np.empty((len(files), kc.FEATURE_COUNT, kc.FEATURE_SIZE), dtype=np.int8)
    for i, p in enumerate(files):
        feats[i] = fx(read_wav(p))
        if i % 500 == 0:
            print(f"features {i}/{len(files)}")
    np.savez_compressed(FEATURE_CACHE, features=feats, names=names)
    return feats, names


def main():
    from tflite_micro.python.tflite_micro import runtime

    files = test_files()
    feats, names = load_features(files)
    folders = np.array([n.split("/")[0] for n in names])
    labels = np.array([kc.folder_label(f) for f in folders])

    model = runtime.Interpreter.from_file(str(kc.fetch_original("models/micro_speech_quantized.tflite")))
    raw = np.empty((len(feats), 4), dtype=np.int8)
    for i, f in enumerate(feats):
        model.set_input(f.reshape(1, -1), 0)
        model.invoke()
        raw[i] = np.asarray(model.get_output(0)).reshape(-1)
    pred = raw.argmax(axis=1)  # first maximum wins, like the firmware

    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    per_class = {kc.LABELS[k]: round(float(cm[k, k] / cm[k].sum()), 4) for k in range(4)}
    per_folder = {f: round(float((pred[folders == f] == labels[folders == f]).mean()), 4)
                  for f in sorted(set(folders))}

    # Eval samples: EVAL_PER_CLASS per category, evenly spaced over that category's clips
    # (unknown therefore mixes _unknown_ and all 8 other command words).
    idx = np.concatenate([np.flatnonzero(labels == k)[np.linspace(0, (labels == k).sum() - 1, EVAL_PER_CLASS)
                                                      .round().astype(int)] for k in range(4)])
    samples = kc.OUT / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    np.save(samples / "eval_inputs.npy", feats[idx].reshape(len(idx), 1, -1))
    np.save(samples / "eval_expected.npy", labels[idx].astype(np.int64))
    # Demo: the example's own yes_1000ms.wav (expected "yes").
    demo = FeatureExtractor()(read_wav(kc.fetch_original("testdata/yes_1000ms.wav")))
    np.save(samples / "demo_input.npy", demo.reshape(1, -1))
    np.save(samples / "demo_expected.npy", np.array([2], dtype=np.int64))

    report = {
        "test_set": "speech_commands_test_set_v0.02 (4,890 clips)",
        "clips": int(len(labels)), "class_counts": {kc.LABELS[k]: int((labels == k).sum()) for k in range(4)},
        "int8_accuracy": round(float((pred == labels).mean()), 4),
        "balanced_accuracy": round(float(np.mean(list(per_class.values()))), 4),
        "per_class_recall": per_class, "per_folder_accuracy": per_folder,
        "confusion_int8": cm.tolist(), "labels": kc.LABELS,
        "eval_samples": int(len(idx)), "eval_names": names[idx].tolist(),
        "eval_int8_accuracy": round(float((pred[idx] == labels[idx]).mean()), 4),
        "demo": "testdata/yes_1000ms.wav", "tflm_commit": kc.TFLM_COMMIT,
    }
    (kc.OUT / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "eval_names"}, indent=2))


if __name__ == "__main__":
    main()
