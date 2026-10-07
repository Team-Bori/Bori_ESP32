#!/usr/bin/env python3
"""Anomalous sound detection autoencoder (DCASE 2023 Task 2 baseline style) for tflm_runtime.

  python pc/models/anomaly_ae/train.py [--machine fan] [--k 32] [--epochs 100]
  python pc/bori_package.py build models/anomaly_ae_int8/manifest.json

Pipeline
  features   log-mel (128 bands, n_fft 1024 = 64 ms, hop 512, power 2, 10*log10) of a 10 s clip,
             5 consecutive frames concatenated -> 309 vectors x 640 (same as the DCASE baseline)
  model      dense AE 640-128-128-128-128-8-128-128-128-128-640 (BatchNorm + ReLU), trained on
             the normal training clips, Adam 1e-3, batch 256, 100 epochs (DCASE baseline settings).
             Features are scaled with ONE global mean/std (not per dimension), so the MSE ranks
             clips exactly like the baseline's MSE on raw dB values (only a constant factor apart)
  on board   input = K evenly spaced vectors of one clip (int8 [K, 640]); the graph computes
             the anomaly score itself: mean((x - AE(x))^2) -> int8 [1, 1]
             (the whole clip, 309 x 640 int8 = 198 KB, would not fit the 152 KB arena)
  decision   score >= threshold -> anomaly; threshold = 90th percentile of a gamma distribution
             fitted to the normal training clips' scores (DCASE baseline method)
  metrics    DCASE 2023: AUC per domain = normal clips of that domain vs all anomalous clips;
             pAUC (p = 0.1) over all normal vs all anomalous clips, sklearn roc_auc_score(max_fpr=0.1)
             (McClish-standardized, as the official evaluator)

Trained weights are cached (~/.cache/bori/dcase2023/<machine>_ae.weights.h5); --reuse skips training.

Data: DCASE 2023 Task 2 development dataset (Zenodo 10.5281/zenodo.7882613), CC BY-NC-SA 4.0
(non-commercial). Downloaded once to ~/.cache/bori/dcase2023 (not committed).
Writes models/anomaly_ae_int8/{model.tflite, samples/*.npy, train_report.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

SEED = 42
ZENODO = "https://zenodo.org/api/records/7882613/files/dev_{m}.zip/content"
ZIP_MD5 = {"fan": "9348591e96fb0ad499a1e33b082562fc"}
CACHE = Path(os.environ.get("BORI_CACHE", Path.home() / ".cache" / "bori")) / "dcase2023"
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "anomaly_ae_int8"

SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 512
FRAMES = 5
DIM = N_MELS * FRAMES  # 640
LABELS = ["normal", "anomaly"]
PER_GROUP = 12          # package eval samples per (domain, label): 48 in total
GAMMA_PERCENTILE = 0.9


def dataset(machine: str) -> Path:
    d = CACHE / machine
    if d.exists():
        return d
    CACHE.mkdir(parents=True, exist_ok=True)
    z = CACHE / f"dev_{machine}.zip"
    if not z.exists():
        print(f"downloading dev_{machine}.zip")
        z.write_bytes(urllib.request.urlopen(ZENODO.format(m=machine), timeout=1200).read())
    if machine in ZIP_MD5 and hashlib.md5(z.read_bytes()).hexdigest() != ZIP_MD5[machine]:
        raise SystemExit(f"md5 mismatch for {z}")
    with zipfile.ZipFile(z) as f:
        f.extractall(CACHE)
    return d


def clip_vectors(path: Path) -> np.ndarray:
    """10 s clip -> [n_vectors, 640] float32 (DCASE baseline feature)."""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    assert sr == SR, path
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    logmel = 10.0 * np.log10(mel + sys.float_info.epsilon)  # 20 / power * log10
    n = logmel.shape[1] - FRAMES + 1
    vec = np.zeros((n, DIM), dtype=np.float32)
    for t in range(FRAMES):
        vec[:, N_MELS * t:N_MELS * (t + 1)] = logmel[:, t:t + n].T
    return vec


def load_split(machine: str, split: str):
    """All clips of a split -> (vectors [clips, 309, 640] float16, names, labels, domains)."""
    cache = CACHE / f"{machine}_{split}_features.npz"
    files = sorted((dataset(machine) / split).glob("*.wav"))
    names = np.array([f.name for f in files])
    if cache.exists():
        z = np.load(cache)
        if z["names"].tolist() == names.tolist():
            feats = z["features"]
            return feats, names, *_meta(names)
    feats = np.stack([clip_vectors(f) for f in files]).astype(np.float16)
    np.savez(cache, features=feats, names=names)
    return feats, names, *_meta(names)


def _meta(names):
    labels = np.array([1 if "_anomaly_" in n else 0 for n in names])
    domains = np.array(["target" if "_target_" in n else "source" for n in names])
    return labels, domains


def subsample(n_vectors: int, k: int) -> np.ndarray:
    return np.linspace(0, n_vectors - 1, k).round().astype(int)


def build_ae():
    import tensorflow as tf

    model = tf.keras.Sequential(name="ae")
    model.add(tf.keras.layers.Input(shape=(DIM,)))
    for units in (128, 128, 128, 128, 8, 128, 128, 128, 128):
        model.add(tf.keras.layers.Dense(units))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(DIM))
    return model


def domain_metrics(scores, labels, domains):
    """DCASE 2023 Task 2 AUC (per domain) and pAUC."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for dom in ("source", "target"):
        m = ((domains == dom) & (labels == 0)) | (labels == 1)
        out[f"auc_{dom}"] = round(float(roc_auc_score(labels[m], scores[m])), 4)
    out["pauc"] = round(float(roc_auc_score(labels, scores, max_fpr=0.1)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="fan")
    ap.add_argument("--k", type=int, default=32, help="vectors per clip on the board")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--reuse", action="store_true", help="load the cached trained weights instead of training")
    args = ap.parse_args()

    import tensorflow as tf
    from scipy import stats

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    tr, tr_names, _, tr_dom = load_split(args.machine, "train")
    te, te_names, te_lab, te_dom = load_split(args.machine, "test")
    n_vec = tr.shape[1]
    x_tr = tr.reshape(-1, DIM).astype(np.float32)
    mean, std = float(x_tr.mean()), float(x_tr.std())  # one global affine scale
    x_tr = (x_tr - mean) / std
    print(f"train clips {len(tr)} ({(tr_dom == 'target').sum()} target), vectors {len(x_tr)}, test clips {len(te)}")

    ae = build_ae()
    weights = CACHE / f"{args.machine}_ae.weights.h5"
    if args.reuse and weights.exists():
        ae.load_weights(weights)
        losses = json.loads(weights.with_suffix(".json").read_text())
    else:
        ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        hist = ae.fit(x_tr, x_tr, epochs=args.epochs, batch_size=256, validation_split=0.1, shuffle=True,
                      verbose=2)
        ae.save_weights(weights)
        losses = {"loss": float(hist.history["loss"][-1]), "val_loss": float(hist.history["val_loss"][-1])}
        weights.with_suffix(".json").write_text(json.dumps(losses))

    norm = lambda a: ((a.astype(np.float32) - mean) / std)  # noqa: E731

    def float_scores(clips, idx=None):
        res = []
        for c in clips:
            v = norm(c if idx is None else c[idx])
            res.append(float(np.mean((v - ae.predict(v, verbose=0, batch_size=512)) ** 2)))
        return np.array(res)

    idx = subsample(n_vec, args.k)
    report = {"machine": args.machine, "k": args.k, "vectors_per_clip": int(n_vec),
              "epochs": args.epochs, "final_loss": round(losses["loss"], 5),
              "final_val_loss": round(losses["val_loss"], 5)}
    s_full = float_scores(te)
    report["float_all_vectors"] = domain_metrics(s_full, te_lab, te_dom)
    s_k = float_scores(te, idx)
    report[f"float_k{args.k}"] = domain_metrics(s_k, te_lab, te_dom)
    print("float all vectors:", report["float_all_vectors"], f"| float K={args.k}:", report[f"float_k{args.k}"])

    # Deployed graph: K vectors -> reconstruction -> mean squared error -> [1, 1]
    x_in = tf.keras.Input(shape=(DIM,), batch_size=args.k, name="vectors")
    recon = ae(x_in, training=False)
    score = tf.keras.layers.Lambda(
        lambda t: tf.reshape(tf.reduce_mean(tf.math.squared_difference(t[0], t[1])), [1, 1]),
        name="mse")([x_in, recon])
    scorer = tf.keras.Model(x_in, score)
    rng = np.random.default_rng(SEED)

    def representative():
        # Normal clips only would calibrate the score (MSE) range to normal values, and every
        # higher (anomalous) score would saturate at the int8 maximum. Half of the calibration
        # clips get Gaussian noise so the squared-error and score ranges also cover large errors.
        for n, i in enumerate(rng.choice(len(tr), 200, replace=False)):
            v = norm(tr[i][idx])
            if n % 2:
                v = v + rng.normal(0.0, rng.uniform(0.1, 0.7), v.shape).astype(np.float32)
            yield [v]

    conv = tf.lite.TFLiteConverter.from_keras_model(scorer)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model.tflite").write_bytes(tflite)

    it = tf.lite.Interpreter(model_content=tflite)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    (in_s, in_z), (out_s, out_z) = inp["quantization"], outp["quantization"]

    def q_clip(c):
        return np.clip(np.round(norm(c[idx]) / in_s) + in_z, -128, 127).astype(np.int8)

    def int8_score(q):
        it.set_tensor(inp["index"], q)
        it.invoke()
        return (float(it.get_tensor(outp["index"])[0, 0]) - out_z) * out_s

    q_te = np.stack([q_clip(c) for c in te])
    s_int8 = np.array([int8_score(q) for q in q_te])
    s_int8_train = np.array([int8_score(q_clip(c)) for c in tr])
    report[f"int8_k{args.k}"] = domain_metrics(s_int8, te_lab, te_dom)

    # Gamma fit on the float scores of the same K vectors (the int8 scores are too coarse,
    # a few dozen distinct values, for a stable fit); the board compares its dequantized score.
    s_float_train = float_scores(tr, idx)
    # floc=0: with a free location scipy pins loc at the smallest score and returns a degenerate
    # shape (~0.2), putting the 90% point far outside the data; floc=0 agrees with the empirical 90%.
    a, loc, scale = stats.gamma.fit(s_float_train, floc=0)
    threshold = float(stats.gamma.ppf(GAMMA_PERCENTILE, a, loc=loc, scale=scale))
    pred = (s_int8 >= threshold).astype(int)
    report["threshold"] = {"method": f"gamma fit (floc=0) on the float scores of {len(tr)} normal training clips (K vectors), {GAMMA_PERCENTILE:.0%} point",
                           "value": round(threshold, 6), "gamma": [float(a), float(loc), float(scale)],
                           "train_clips_above": round(float((s_int8_train >= threshold).mean()), 4)}
    report["int8_decision"] = {dom: {"accuracy": round(float((pred[te_dom == dom] == te_lab[te_dom == dom]).mean()), 4),
                                     "normal_ok": round(float((pred[(te_dom == dom) & (te_lab == 0)] == 0).mean()), 4),
                                     "anomaly_found": round(float((pred[(te_dom == dom) & (te_lab == 1)] == 1).mean()), 4)}
                               for dom in ("source", "target")}
    report["int8_decision"]["all"] = round(float((pred == te_lab).mean()), 4)
    pct = lambda a: [round(float(v), 4) for v in np.percentile(a, [0, 10, 50, 90, 100])]  # noqa: E731
    report["int8_score_percentiles_0_10_50_90_100"] = {
        "train_normal": pct(s_int8_train),
        "test_normal": pct(s_int8[te_lab == 0]), "test_anomaly": pct(s_int8[te_lab == 1]),
        "distinct_train_values": int(len(np.unique(s_int8_train)))}
    print(f"int8 K={args.k}:", report[f"int8_k{args.k}"], "threshold", round(threshold, 4), report["int8_decision"])

    # Package samples: PER_GROUP clips per (domain, label), evenly spaced in file order.
    picks = []
    for dom in ("source", "target"):
        for lab in (0, 1):
            g = np.flatnonzero((te_dom == dom) & (te_lab == lab))
            picks += g[np.linspace(0, len(g) - 1, PER_GROUP).round().astype(int)].tolist()
    picks = np.array(picks)
    samples = OUT / "samples"
    samples.mkdir(exist_ok=True)
    np.save(samples / "eval_inputs.npy", q_te[picks])
    np.save(samples / "eval_expected.npy", te_lab[picks].astype(np.int64))
    # Demo: the source-domain anomaly (outside the eval picks) with the highest score.
    cand = [i for i in np.flatnonzero((te_dom == "source") & (te_lab == 1)) if i not in set(picks.tolist())]
    demo = max(cand, key=lambda i: s_int8[i])
    np.save(samples / "demo_input.npy", q_te[demo])
    np.save(samples / "demo_expected.npy", np.array([1], dtype=np.int64))

    report.update({
        "eval_samples": int(len(picks)), "eval_names": te_names[picks].tolist(),
        "eval_int8_accuracy": round(float((pred[picks] == te_lab[picks]).mean()), 4),
        "demo": te_names[demo], "demo_score": round(float(s_int8[demo]), 5),
        "normalization": {"mean": round(mean, 6), "std": round(std, 6),
                          "formula": "x_norm = (logmel_dB - mean) / std (one global value for all 640 dims)"},
        "input_quantization": {"scale": float(in_s), "zero_point": int(in_z)},
        "output_quantization": {"scale": float(out_s), "zero_point": int(out_z)},
        "tflite_bytes": len(tflite), "params": int(ae.count_params()),
        "tensorflow": tf.__version__, "numpy": np.__version__, "python": platform.python_version(),
    })
    (OUT / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("normalization", "eval_names")}, indent=2))


if __name__ == "__main__":
    main()
