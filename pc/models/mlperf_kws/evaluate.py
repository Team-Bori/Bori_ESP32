#!/usr/bin/env python3
"""Host int8 accuracy of the MLPerf Tiny KWS reference model and the package samples.

  python evaluate.py        (training venv: tensorflow; needs the Speech Commands v0.02 test set
                             and, for the official-set comparison, EEMBC's kws01 inputs)

- accuracy on all 4,890 Speech Commands v0.02 test clips with our features (features.py)
- accuracy on EEMBC's 1,000 official benchmark inputs (datasets/kws01 of eembc/benchmark-runner-ml)
  and on our features of the same clips, to show the feature differences do not matter
- package samples: EVAL_PER_CLASS clips per class (evenly spaced), demo = a "yes" clip
The EEMBC inputs are only read for this comparison; they are not redistributed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "mlperf_kws_dscnn_int8"
EVAL_PER_CLASS = 15  # 180 samples


def run(tflite: Path, x: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    it = tf.lite.Interpreter(model_path=str(tflite))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = np.empty((len(x), 12), dtype=np.int8)
    for k in range(len(x)):
        it.set_tensor(i["index"], x[k].reshape(1, 49, 10, 1))
        it.invoke()
        out[k] = it.get_tensor(o["index"])[0]
    return out.argmax(axis=1)


def main():
    tflite = OUT / "model.tflite"
    mfcc, folders, names = F.all_features()
    labels = np.array([F.FOLDER_TO_LABEL[f] for f in folders])
    q = np.stack([F.quantize(m) for m in mfcc])
    pred = run(tflite, q)
    per_class = {F.LABELS[c]: round(float((pred[labels == c] == c).mean()), 4) for c in range(12)}
    report = {"test_clips": int(len(q)), "int8_accuracy": round(float((pred == labels).mean()), 4),
              "per_class_recall": per_class}

    # Official MLPerf Tiny accuracy set (EEMBC kws01) vs our features of the same clips.
    bins = sorted(F.EEMBC_KWS01.glob("*.bin"))
    if bins:
        e = np.stack([np.frombuffer(b.read_bytes(), dtype=np.int8) for b in bins])
        e_lab = np.array([int(b.stem.rsplit("_", 1)[1]) for b in bins])
        e_pred = run(tflite, e)
        ours = {}
        for k, x in enumerate(q):
            ours.setdefault(hashlib.sha1(x.tobytes()).hexdigest(), k)
        # same clip = byte-identical, else the closest same-label clip (the +-1 rounding cases)
        qi = q.reshape(len(q), -1).astype(np.int16)
        match, far = [], 0
        for b, lab in zip(e, e_lab):
            k = ours.get(hashlib.sha1(b.tobytes()).hexdigest())
            if k is None:
                cand = np.flatnonzero(labels == lab)
                d = np.abs(qi[cand] - b.astype(np.int16)[None]).sum(axis=1)
                k = int(cand[d.argmin()]) if d.min() <= 100 else -1
                far += k < 0
            match.append(k)
        match = np.array(match)
        ok = match >= 0
        report["eembc_kws01"] = {
            "inputs": int(len(bins)), "int8_accuracy_eembc_inputs": round(float((e_pred == e_lab).mean()), 4),
            "matched_clips": int(ok.sum()), "not_in_our_test_set": int(far),
            "accuracy_on_matched_eembc_inputs": round(float((e_pred[ok] == e_lab[ok]).mean()), 4),
            "accuracy_on_our_features_same_clips": round(float((pred[match[ok]] == e_lab[ok]).mean()), 4),
            "predictions_differing": int((e_pred[ok] != pred[match[ok]]).sum()),
        }

    idx = np.concatenate([np.flatnonzero(labels == c)[np.linspace(0, (labels == c).sum() - 1, EVAL_PER_CLASS)
                                                      .round().astype(int)] for c in range(12)])
    yes = np.flatnonzero((labels == 9) & (pred == 9))
    demo = int([k for k in yes if k not in set(idx.tolist())][0])
    samples = OUT / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    np.save(samples / "eval_inputs.npy", q[idx])
    np.save(samples / "eval_expected.npy", labels[idx].astype(np.int64))
    np.save(samples / "demo_input.npy", q[demo])
    np.save(samples / "demo_expected.npy", np.array([9], dtype=np.int64))
    report.update({"eval_samples": int(len(idx)), "eval_int8_accuracy": round(float((pred[idx] == labels[idx]).mean()), 4),
                   "eval_names": names[idx].tolist(), "demo": str(names[demo])})
    (OUT / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "eval_names"}, indent=2))


if __name__ == "__main__":
    main()
