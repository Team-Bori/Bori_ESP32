#!/usr/bin/env python3
"""Host int8 accuracy of person_detect.tflite on Visual Wake Words val images, and the
package samples for models/person_detection_int8.

  python evaluate.py      (after prepare.py; needs tflite-micro + Pillow: Linux / WSL,
                           with BORI_CACHE pointing at the same cache as prepare.py)

- host accuracy: HOST_EVAL_IMAGES random VWW val (COCO minival) images, any license,
  with the conversion-time preprocessing (resize whole image) and, for comparison,
  the TF-slim eval preprocessing (central 87.5% crop)
- package: the license-filtered images chosen by prepare.py (class balanced) + one demo
  person image; writes samples/*.npy and image_licenses.csv (id, license, attribution link)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pd_common as pc  # noqa: E402
from preprocess import to_input  # noqa: E402


def run(model, x: np.ndarray) -> np.ndarray:
    raw = np.empty((len(x), 2), dtype=np.int8)
    for i in range(len(x)):
        model.set_input(x[i][None], 0)
        model.invoke()
        raw[i] = np.asarray(model.get_output(0)).reshape(-1)
    return raw


def accuracy(raw, y):
    pred = raw.argmax(axis=1)  # first maximum wins, like the firmware
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return {"accuracy": round(float((pred == y).mean()), 4),
            "recall_no_person": round(float(cm[0, 0] / cm[0].sum()), 4),
            "recall_person": round(float(cm[1, 1] / cm[1].sum()), 4),
            "confusion": cm.tolist()}


def main():
    from tflite_micro.python.tflite_micro import runtime
    import PIL

    index = json.loads(pc.INDEX.read_text())
    sel = json.loads((pc.COCO / "vww_selection.json").read_text())
    rec = {r["id"]: r for r in index["images"]}
    model = runtime.Interpreter.from_file(str(pc.ORIGINAL / "person_detect.tflite"))

    host = [rec[i] for i in sel["host_eval"]]
    y_host = np.array([r["label"] for r in host])
    report = {"host_eval_images": len(host), "host_person": int(y_host.sum()),
              "pillow": PIL.__version__, "tflm_commit": pc.TFLM_COMMIT}
    for variant, crop in (("resize", False), ("center_crop_0.875", True)):
        x = np.stack([to_input(pc.IMAGES / r["file_name"], center_crop=crop) for r in host])
        report[f"host_{variant}"] = accuracy(run(model, x), y_host)
        print(variant, report[f"host_{variant}"])

    # Package: per_class no_person + per_class person for eval, the extra person image is the demo.
    pkg = [rec[i] for i in sel["package"]]
    x_pkg = np.stack([to_input(pc.IMAGES / r["file_name"]) for r in pkg])
    y_pkg = np.array([r["label"] for r in pkg])
    raw_pkg = run(model, x_pkg)
    person_idx = [i for i, r in enumerate(pkg) if r["label"] == 1]
    # Demo: the person image the model is most confident about (shows a clear "person" result).
    demo = max(person_idx, key=lambda i: int(raw_pkg[i, 1]) - int(raw_pkg[i, 0]))
    eval_idx = [i for i in range(len(pkg)) if i != demo]
    order = sorted(eval_idx, key=lambda i: (y_pkg[i], pkg[i]["id"]))

    samples = pc.OUT / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    np.save(samples / "eval_inputs.npy", x_pkg[order])
    np.save(samples / "eval_expected.npy", y_pkg[order].astype(np.int64))
    np.save(samples / "demo_input.npy", x_pkg[demo])
    np.save(samples / "demo_expected.npy", np.array([1], dtype=np.int64))
    report["package"] = {"eval_samples": len(order), **accuracy(raw_pkg[order], y_pkg[order]),
                         "demo_image_id": pkg[demo]["id"],
                         "demo_scores": raw_pkg[demo].tolist()}
    print("package", report["package"])

    with open(pc.OUT / "image_licenses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["use", "sample_index", "coco_image_id", "label", "license", "attribution_flickr_page", "coco_url"])
        for k, i in enumerate(order):
            r = pkg[i]
            w.writerow(["eval", k, r["id"], pc.LABELS[r["label"]], pc.REDISTRIBUTABLE[r["license"]],
                        pc.flickr_page(r["flickr_url"]), r["coco_url"]])
        r = pkg[demo]
        w.writerow(["demo", 0, r["id"], pc.LABELS[r["label"]], pc.REDISTRIBUTABLE[r["license"]],
                    pc.flickr_page(r["flickr_url"]), r["coco_url"]])
    (pc.OUT / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
