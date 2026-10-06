#!/usr/bin/env python3
"""Trains the small MNIST CNN, converts it to a full-int8 TFLite model and writes
the package inputs for models/cnn_mnist_int8.

  python pc/models/cnn_mnist/train.py            # train + convert + samples + report
  python pc/bori_package.py build models/cnn_mnist_int8/manifest.json

Reproducible: fixed seeds and TF op determinism (same TF version -> same weights).
Writes:
  models/cnn_mnist_int8/model.tflite
  models/cnn_mnist_int8/samples/{demo_input,demo_expected,eval_inputs,eval_expected}.npy
  models/cnn_mnist_int8/train_report.json   (float / int8 accuracy on the full test set)

This is also the reference template for user models (docs/USER_MODEL_GUIDE.md).
"""
import argparse
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

SEED = 42
OUT = Path(__file__).resolve().parents[3] / "models" / "cnn_mnist_int8"
EVAL_PER_CLASS = 20          # 200 eval samples in the package
REPRESENTATIVE_SAMPLES = 500  # calibration images for int8 quantization


def build_model(c1: int, c2: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(c1, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(c2, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


def convert_int8(model: tf.keras.Model, rep_images: np.ndarray) -> bytes:
    def representative():
        for img in rep_images:
            yield [img[None].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def quantize_images(images: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(images / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def int8_predict(tflite: bytes, q_images: np.ndarray) -> np.ndarray:
    """Raw int8 outputs of the TFLite model (TF Lite interpreter, reference for the host)."""
    it = tf.lite.Interpreter(model_content=tflite)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    res = np.empty((len(q_images), 10), dtype=np.int8)
    for i, x in enumerate(q_images):
        it.set_tensor(inp["index"], x[None])
        it.invoke()
        res[i] = it.get_tensor(out["index"])[0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--filters", type=int, nargs=2, default=(8, 16), metavar=("C1", "C2"))
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) / 255.0)[..., None]
    x_test = (x_test.astype(np.float32) / 255.0)[..., None]

    model = build_model(*args.filters)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=128,
              validation_split=0.1, verbose=2, shuffle=True)
    _, float_acc = model.evaluate(x_test, y_test, verbose=0)

    rng = np.random.default_rng(SEED)
    rep = x_train[rng.choice(len(x_train), REPRESENTATIVE_SAMPLES, replace=False)]
    tflite = convert_int8(model, rep)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model.tflite").write_bytes(tflite)

    it = tf.lite.Interpreter(model_content=tflite)
    in_scale, in_zp = it.get_input_details()[0]["quantization"]
    q_test = quantize_images(x_test, in_scale, in_zp)
    raw = int8_predict(tflite, q_test)
    pred = raw.argmax(axis=1)  # same rule as the firmware (first maximum wins)
    int8_acc = float((pred == y_test).mean())

    # Eval samples: the first EVAL_PER_CLASS test images of each class (test order).
    idx = np.concatenate([np.flatnonzero(y_test == c)[:EVAL_PER_CLASS] for c in range(10)])
    samples = OUT / "samples"
    samples.mkdir(exist_ok=True)
    np.save(samples / "eval_inputs.npy", q_test[idx])
    np.save(samples / "eval_expected.npy", y_test[idx].astype(np.int64))
    eval_acc = float((pred[idx] == y_test[idx]).mean())

    # Demo: first test image (outside the eval set) that the int8 model classifies with
    # the top score and a clear margin.
    out_scale, out_zp = it.get_output_details()[0]["quantization"]
    probs = (raw.astype(np.float32) - out_zp) * out_scale
    eval_set = set(idx.tolist())
    demo = next(i for i in range(len(y_test))
                if i not in eval_set and pred[i] == y_test[i] and probs[i].max() >= 0.99)
    np.save(samples / "demo_input.npy", q_test[demo])
    np.save(samples / "demo_expected.npy", np.array([y_test[demo]], dtype=np.int64))

    report = {
        "seed": SEED, "epochs": args.epochs, "filters": list(args.filters),
        "params": int(model.count_params()), "tflite_bytes": len(tflite),
        "test_samples": int(len(y_test)),
        "float_test_accuracy": round(float(float_acc), 4),
        "int8_test_accuracy": round(int8_acc, 4),
        "eval_samples": int(len(idx)), "eval_int8_accuracy": round(eval_acc, 4),
        "demo_test_index": int(demo), "demo_label": int(y_test[demo]),
        "input_quantization": {"scale": float(in_scale), "zero_point": int(in_zp)},
        "output_quantization": {"scale": float(out_scale), "zero_point": int(out_zp)},
        "tensorflow": tf.__version__, "numpy": np.__version__, "python": platform.python_version(),
    }
    (OUT / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
