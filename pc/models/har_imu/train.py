#!/usr/bin/env python3
"""Human activity recognition 1D CNN on the UCI HAR raw inertial signals.

  python pc/models/har_imu/train.py --compare            # train 3/6/9-channel variants, write compare report
  python pc/models/har_imu/train.py --channels 9         # write models/har_imu_int8 artifacts for one variant
  python pc/bori_package.py build models/har_imu_int8/manifest.json

Data: UCI "Human Activity Recognition Using Smartphones" (CC BY 4.0, DOI 10.24432/C54S4K),
downloaded once to ~/.cache/bori/uci_har (not committed). Official train/test split
(subjects disjoint) is kept; the test split is only used for the final numbers.
Validation (early stopping) holds out whole training subjects.

Input: one 2.56 s window = 128 samples at 50 Hz x C channels, standardized per channel
with the training mean/std (stored in the manifest), then int8-quantized.
"""
import argparse
import hashlib
import json
import os
import platform
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

SEED = 42
URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
CACHE = Path.home() / ".cache" / "bori" / "uci_har"
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "har_imu_int8"

LABELS = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]
# Channel sets. total_acc is what an accelerometer measures (g, gravity included);
# body_gyro is the gyroscope (rad/s); body_acc is total_acc minus a low-pass gravity estimate.
CHANNEL_SETS = {
    3: ["total_acc_x", "total_acc_y", "total_acc_z"],
    6: ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"],
    9: ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"],
}
VAL_SUBJECTS = 3            # training subjects held out for early stopping
EVAL_PER_CLASS = 25         # 150 eval samples in the package
REPRESENTATIVE_SAMPLES = 500


def dataset_dir() -> Path:
    d = CACHE / "UCI HAR Dataset"
    if d.exists():
        return d
    CACHE.mkdir(parents=True, exist_ok=True)
    outer = CACHE / "har.zip"
    if not outer.exists():
        print(f"downloading {URL}")
        outer.write_bytes(urllib.request.urlopen(URL, timeout=120).read())
    digest = hashlib.sha256(outer.read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        raise SystemExit(f"sha256 mismatch for {outer}: {digest}")
    with zipfile.ZipFile(outer) as z:
        z.extractall(CACHE)
    with zipfile.ZipFile(CACHE / "UCI HAR Dataset.zip") as z:
        z.extractall(CACHE)
    return d


def load_split(split: str, channels: list[str]):
    d = dataset_dir() / split
    x = np.stack([np.loadtxt(d / "Inertial Signals" / f"{c}_{split}.txt", dtype=np.float32)
                  for c in channels], axis=-1)                       # [N, 128, C]
    y = np.loadtxt(d / f"y_{split}.txt", dtype=np.int64) - 1          # 0..5
    subjects = np.loadtxt(d / f"subject_{split}.txt", dtype=np.int64)
    return x, y, subjects


def build_model(c: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, c)),
        tf.keras.layers.Conv1D(16, 5, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 5, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation="softmax"),
    ])


def convert_int8(model, rep):
    def representative():
        for x in rep:
            yield [x[None].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_ops(tflite: bytes) -> list[str]:
    import sys
    sys.path.insert(0, str(ROOT / "pc"))
    import btf_format as bf
    return bf.tflite_info(tflite)["ops"]


def run_int8(tflite: bytes, x_float: np.ndarray):
    it = tf.lite.Interpreter(model_content=tflite)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    q = np.clip(np.round(x_float / scale) + zp, -128, 127).astype(np.int8)
    raw = np.empty((len(q), 6), dtype=np.int8)
    for i in range(len(q)):
        it.set_tensor(inp["index"], q[i][None])
        it.invoke()
        raw[i] = it.get_tensor(out["index"])[0]
    return q, raw, (float(scale), int(zp)), out["quantization"]


def train_variant(c: int, epochs: int):
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    names = CHANNEL_SETS[c]
    x_tr, y_tr, s_tr = load_split("train", names)
    x_te, y_te, _ = load_split("test", names)

    # Hold out whole subjects for validation (no window of a validation subject is trained on).
    val_subj = np.unique(s_tr)[-VAL_SUBJECTS:]
    vmask = np.isin(s_tr, val_subj)
    mean = x_tr[~vmask].reshape(-1, c).mean(axis=0)
    std = x_tr[~vmask].reshape(-1, c).std(axis=0)
    norm = lambda a: (a - mean) / std  # noqa: E731

    model = build_model(c)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                            restore_best_weights=True)
    hist = model.fit(norm(x_tr[~vmask]), y_tr[~vmask], validation_data=(norm(x_tr[vmask]), y_tr[vmask]),
                     epochs=epochs, batch_size=64, verbose=0, callbacks=[stop], shuffle=True)
    best_epoch = int(np.argmax(hist.history["val_accuracy"])) + 1

    float_acc = float(model.evaluate(norm(x_te), y_te, verbose=0)[1])
    rng = np.random.default_rng(SEED)
    rep = norm(x_tr[~vmask])[rng.choice(int((~vmask).sum()), REPRESENTATIVE_SAMPLES, replace=False)]
    tflite = convert_int8(model, rep)
    q_te, raw, in_q, out_q = run_int8(tflite, norm(x_te))
    pred = raw.argmax(axis=1)
    int8_acc = float((pred == y_te).mean())
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(y_te, pred):
        cm[t, p] += 1
    return {
        "channels": c, "channel_names": names, "params": int(model.count_params()),
        "tflite_bytes": len(tflite), "best_epoch": best_epoch,
        "val_subjects": [int(s) for s in val_subj],
        "float_test_accuracy": round(float_acc, 4), "int8_test_accuracy": round(int8_acc, 4),
        "confusion_int8": cm.tolist(), "ops": tflite_ops(tflite),
        "normalization": {"mean": [float(v) for v in mean], "std": [float(v) for v in std]},
        "input_quantization": {"scale": in_q[0], "zero_point": in_q[1]},
    }, tflite, q_te, y_te, raw


def write_artifacts(report, tflite, q_te, y_te, raw):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model.tflite").write_bytes(tflite)
    pred = raw.argmax(axis=1)
    # Evenly spaced over each class's test windows, so every test subject is represented
    # (the first windows of a class all come from the first test subject).
    idx = np.concatenate([
        np.flatnonzero(y_te == k)[np.linspace(0, (y_te == k).sum() - 1, EVAL_PER_CLASS).round().astype(int)]
        for k in range(6)])
    samples = OUT / "samples"
    samples.mkdir(exist_ok=True)
    np.save(samples / "eval_inputs.npy", q_te[idx])
    np.save(samples / "eval_expected.npy", y_te[idx])
    # Demo: first correctly classified, confident walking window outside the eval set.
    probs = (raw.astype(np.float32) + 128) / 256.0
    eval_set = set(idx.tolist())
    demo = next(i for i in range(len(y_te))
                if i not in eval_set and y_te[i] == 0 and pred[i] == 0 and probs[i].max() >= 0.9)
    np.save(samples / "demo_input.npy", q_te[demo])
    np.save(samples / "demo_expected.npy", np.array([0], dtype=np.int64))
    report = dict(report)
    report.update({"eval_samples": int(len(idx)),
                   "eval_int8_accuracy": round(float((pred[idx] == y_te[idx]).mean()), 4),
                   "demo_test_index": int(demo), "labels": LABELS,
                   "tensorflow": tf.__version__, "numpy": np.__version__,
                   "python": platform.python_version(), "seed": SEED})
    (OUT / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, choices=sorted(CHANNEL_SETS), default=9)
    ap.add_argument("--compare", action="store_true", help="train every channel set, write compare report")
    ap.add_argument("--epochs", type=int, default=100, help="maximum epochs (early stopping on val subjects)")
    args = ap.parse_args()

    if args.compare:
        rows = []
        for c in sorted(CHANNEL_SETS):
            r, *_ = train_variant(c, args.epochs)
            print(f"C={c}: float {r['float_test_accuracy']}, int8 {r['int8_test_accuracy']}, "
                  f"{r['tflite_bytes']} B, best epoch {r['best_epoch']}")
            rows.append({k: r[k] for k in ("channels", "channel_names", "params", "tflite_bytes", "best_epoch",
                                           "float_test_accuracy", "int8_test_accuracy", "confusion_int8")})
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "channel_compare.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return

    r, tflite, q_te, y_te, raw = train_variant(args.channels, args.epochs)
    print(json.dumps(write_artifacts(r, tflite, q_te, y_te, raw), indent=2))


if __name__ == "__main__":
    main()
