import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from model_format import build_model_bin, read_model_bin

INPUT_SIZE = 64
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10


def qsym(x: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0:
        scale = 1.0
    q = np.round(x / scale)
    return np.clip(q, -127, 127).astype(np.int8)


def qinfer(x, q):
    qw1, qw2, qb1, qb2 = q["qw1"], q["qw2"], q["qb1"], q["qb2"]
    input_scale = q["input_scale"]
    w1_scale = q["w1_scale"]
    hidden_scale = q["hidden_scale"]
    w2_scale = q["w2_scale"]

    qx = np.round(x / input_scale).clip(-127, 127).astype(np.int8)
    W1 = qw1.reshape(INPUT_SIZE, HIDDEN_SIZE)
    W2 = qw2.reshape(HIDDEN_SIZE, OUTPUT_SIZE)

    acc1 = qx.astype(np.int32) @ W1.astype(np.int32) + qb1
    a1 = np.maximum(acc1 * (input_scale * w1_scale), 0.0)
    q1 = np.round(a1 / hidden_scale).clip(0, 127).astype(np.int8)

    acc2 = q1.astype(np.int32) @ W2.astype(np.int32) + qb2
    logits = acc2.astype(np.float32) * (hidden_scale * w2_scale)
    return int(np.argmax(logits)), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="../models/mlp/default_model.bin")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_digits()
    X = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_SIZE,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    fp_acc = accuracy_score(y_test, clf.predict(X_test))

    w1 = clf.coefs_[0].astype(np.float32)
    b1 = clf.intercepts_[0].astype(np.float32)
    w2 = clf.coefs_[1].astype(np.float32)
    b2 = clf.intercepts_[1].astype(np.float32)

    input_scale = 16.0 / 127.0
    w1_scale = max(float(np.max(np.abs(w1)) / 127.0), 1e-12)
    w2_scale = max(float(np.max(np.abs(w2)) / 127.0), 1e-12)

    x_all = ds.data.astype(np.float32)
    z1 = x_all @ w1 + b1
    a1 = np.maximum(z1, 0.0)
    hidden_max = float(np.max(a1))
    hidden_scale = max(hidden_max / 127.0, 1e-6)

    qw1 = qsym(w1, w1_scale)
    qw2 = qsym(w2, w2_scale)
    qb1 = np.round(b1 / (input_scale * w1_scale)).astype(np.int32)
    qb2 = np.round(b2 / (hidden_scale * w2_scale)).astype(np.int32)

    q = {
        "input_scale": input_scale,
        "w1_scale": w1_scale,
        "hidden_scale": hidden_scale,
        "w2_scale": w2_scale,
        "qw1": qw1,
        "qw2": qw2,
        "qb1": qb1,
        "qb2": qb2,
    }

    q_pred = np.array([qinfer(x, q)[0] for x in X_test])
    q_acc = float(np.mean(q_pred == y_test))

    output = Path(args.output)
    if not output.is_absolute():
        output = (Path(__file__).resolve().parent / output).resolve()

    total_size = build_model_bin(
        output,
        qw1,
        qb1,
        qw2,
        qb2,
        input_scale,
        w1_scale,
        hidden_scale,
        w2_scale,
    )
    info = read_model_bin(output)

    fp_bytes = w1.nbytes + b1.nbytes + w2.nbytes + b2.nbytes
    int8_bytes = qw1.nbytes + qb1.nbytes + qw2.nbytes + qb2.nbytes

    print(f"FP32 test accuracy : {fp_acc:.4f}")
    print(f"INT8 test accuracy : {q_acc:.4f}")
    print(f"INT8 model bytes   : {int8_bytes}")
    print(f"FP32 model bytes   : {fp_bytes}")
    print(f"model.bin bytes    : {total_size}")
    print(f"checksum           : 0x{info['checksum']:08x}")
    print(f"Output             : {output}")


if __name__ == "__main__":
    main()
