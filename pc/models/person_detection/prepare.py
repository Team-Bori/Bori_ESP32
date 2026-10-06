#!/usr/bin/env python3
"""Prepares the person_detection model and Visual Wake Words evaluation images.

- downloads person_detect.tflite at the pinned tflite-micro commit (sha256 checked),
  copies it unchanged to models/person_detection_int8/model.tflite
- downloads COCO 2014 annotations + the VWW val (COCO minival) id list, labels every
  minival image (person box > 0.5% of the image area -> person) and records its license
- picks the host evaluation images (random, any license, never redistributed) and the
  package images (only REDISTRIBUTABLE licenses, class balanced) and downloads them

Everything except the model goes to ~/.cache/bori (not committed).
  python prepare.py [--package-per-class 30]
"""
import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pd_common as pc  # noqa: E402


def download(url: str, dst: Path, sha: str | None = None):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dst)
    if sha and pc.sha256(dst) != sha:
        raise SystemExit(f"sha256 mismatch for {dst}")
    return dst


def build_index() -> dict:
    if pc.INDEX.exists():
        return json.loads(pc.INDEX.read_text())
    zip_path = download(pc.ANNOTATIONS_URL, pc.COCO / "annotations_trainval2014.zip", pc.ANNOTATIONS_SHA256)
    ann_path = pc.COCO / "annotations" / "instances_val2014.json"
    if not ann_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extract("annotations/instances_val2014.json", pc.COCO)
    ids = [int(x) for x in download(pc.MINIVAL_URL, pc.COCO / "mscoco_minival_ids.txt").read_text().split()]
    d = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in d["images"]}
    person = next(c["id"] for c in d["categories"] if c["name"] == "person")
    has_person = set()
    for a in d["annotations"]:
        if a["category_id"] == person:
            im = images[a["image_id"]]
            if a["area"] > pc.VWW_AREA_THRESHOLD * im["width"] * im["height"]:
                has_person.add(a["image_id"])
    licenses = {l["id"]: l["name"] for l in d["licenses"]}
    index = {"licenses": licenses, "images": [
        {"id": i, "label": int(i in has_person), "license": images[i]["license"],
         "file_name": images[i]["file_name"], "coco_url": images[i]["coco_url"],
         "flickr_url": images[i]["flickr_url"], "width": images[i]["width"], "height": images[i]["height"]}
        for i in ids]}
    pc.INDEX.write_text(json.dumps(index))
    return index


def choose(index: dict, per_class: int):
    rng = random.Random(pc.SEED)
    imgs = index["images"]
    host = sorted(rng.sample(imgs, pc.HOST_EVAL_IMAGES), key=lambda r: r["id"])
    by_label = defaultdict(list)
    for r in imgs:
        if r["license"] in pc.REDISTRIBUTABLE:
            by_label[r["label"]].append(r)
    package = []
    for label in (0, 1):
        pool = sorted(by_label[label], key=lambda r: r["id"])
        package += rng.sample(pool, per_class + (1 if label == 1 else 0))  # +1 person image for the demo
    return host, package


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-per-class", type=int, default=30)
    args = ap.parse_args()

    model = download(pc.MODEL_URL, pc.ORIGINAL / "person_detect.tflite", pc.MODEL_SHA256)
    for name in ("person.bmp", "no_person.bmp"):
        download(f"{pc.EXAMPLE_TESTDATA}/{name}", pc.ORIGINAL / name)
    pc.OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, pc.OUT / "model.tflite")

    index = build_index()
    host, package = choose(index, args.package_per_class)
    todo = {r["file_name"]: r["coco_url"] for r in host + package}
    pc.IMAGES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda kv: download(kv[1], pc.IMAGES / kv[0]), todo.items()))

    selection = {"host_eval": [r["id"] for r in host], "package": [r["id"] for r in package],
                 "package_per_class": args.package_per_class}
    (pc.COCO / "vww_selection.json").write_text(json.dumps(selection))
    n_person = sum(r["label"] for r in host)
    print(f"index: {len(index['images'])} minival images")
    print(f"host eval: {len(host)} images ({n_person} person / {len(host) - n_person} no_person)")
    lic = defaultdict(int)
    for r in package:
        lic[pc.REDISTRIBUTABLE[r["license"]]] += 1
    print(f"package images: {len(package)} ({dict(lic)}); downloaded {len(todo)} files to {pc.IMAGES}")


if __name__ == "__main__":
    main()
