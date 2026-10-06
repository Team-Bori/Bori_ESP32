"""Shared settings for the person_detection scripts."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

TFLM_COMMIT = "22c2469a233981012ac16ad849f59c7105655aa9"
MODEL_URL = ("https://raw.githubusercontent.com/tensorflow/tflite-micro/"
             f"{TFLM_COMMIT}/tensorflow/lite/micro/models/person_detect.tflite")
MODEL_SHA256 = "808cfdfc0cf3a6fa6f6fa26bfa379ea97c16d5db7334637766e39c3408502e9d"
EXAMPLE_TESTDATA = ("https://raw.githubusercontent.com/tensorflow/tflite-micro/"
                    f"{TFLM_COMMIT}/tensorflow/lite/micro/examples/person_detection/testdata")

# COCO 2014 annotations and the Visual Wake Words "val" split (= COCO minival, 8,059 val2014 images).
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
ANNOTATIONS_SHA256 = "031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009"
MODELS_COMMIT = "930a6f98f7debcc32ca7afbca4a176dbf9211e03"  # tensorflow/models
MINIVAL_URL = ("https://raw.githubusercontent.com/tensorflow/models/"
               f"{MODELS_COMMIT}/research/object_detection/data/mscoco_minival_ids.txt")
VWW_AREA_THRESHOLD = 0.005   # person box area > 0.5% of the image -> "person" (VWW definition)

# COCO license ids that allow redistributing a modified (resized, grayscale) copy.
# 1-3 are NonCommercial, 6 is NoDerivs: never put into a package.
REDISTRIBUTABLE = {4: "CC BY 2.0", 5: "CC BY-SA 2.0", 7: "No known copyright restrictions",
                   8: "United States Government Work"}

LABELS = ["no_person", "person"]  # model output order (model_settings.h)

CACHE = Path(os.environ.get("BORI_CACHE", Path.home() / ".cache" / "bori"))
ORIGINAL = CACHE / "person_detection_original"
COCO = CACHE / "coco"
IMAGES = COCO / "val2014"
INDEX = COCO / "vww_minival_index.json"

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "person_detection_int8"

HOST_EVAL_IMAGES = 2000   # random minival images for the host accuracy (any license, not redistributed)
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flickr_page(flickr_url: str) -> str:
    """Flickr photo page for attribution, from a static image URL (.../<photo id>_<secret>_z.jpg)."""
    photo_id = flickr_url.rsplit("/", 1)[-1].split("_", 1)[0]
    return f"https://www.flickr.com/photo.gne?id={photo_id}"
