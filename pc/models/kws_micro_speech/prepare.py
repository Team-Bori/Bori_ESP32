#!/usr/bin/env python3
"""Prepares models/kws_micro_speech_int8 from the TFLite Micro micro_speech example.

- downloads the original example files at the pinned commit and checks their sha256
- copies micro_speech_quantized.tflite (unchanged) to models/kws_micro_speech_int8/model.tflite
- downloads the Speech Commands v0.02 test set (CC BY 4.0) to ~/.cache/bori (not committed)
- prints the model's input/output tensors and quantization

Then: features.py --verify, evaluate.py, bori_package.py build.
"""
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kws_common as kc  # noqa: E402

sys.path.insert(0, str(kc.ROOT / "pc"))
import btf_format as bf  # noqa: E402


def fetch_test_set():
    if kc.TEST_SET_DIR.exists() and any(kc.TEST_SET_DIR.glob("yes/*.wav")):
        return
    archive = kc.TEST_SET_DIR.parent / "speech_commands_test_set_v0.02.tar.gz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {kc.TEST_SET_URL} (112 MB)")
        with urllib.request.urlopen(kc.TEST_SET_URL, timeout=600) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
    if kc.sha256(archive) != kc.TEST_SET_SHA256:
        raise SystemExit(f"sha256 mismatch for {archive}")
    kc.TEST_SET_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(kc.TEST_SET_DIR)


def main():
    for name in kc.ORIGINAL_FILES:
        p = kc.fetch_original(name)
        print(f"{name}: {p.stat().st_size} B, sha256 {kc.sha256(p)[:16]}...")

    kc.OUT.mkdir(parents=True, exist_ok=True)
    model = kc.OUT / "model.tflite"
    shutil.copyfile(kc.fetch_original("models/micro_speech_quantized.tflite"), model)
    info = bf.tflite_info(model.read_bytes())
    (tin, _), (tout, _) = info["inputs"][0], info["outputs"][0]
    print(f"model: {model.stat().st_size} B, ops {info['ops']}")
    print(f"input {tin.to_json()}")
    print(f"output {tout.to_json()}  labels {kc.LABELS}")

    fetch_test_set()
    folders = sorted(p.name for p in kc.TEST_SET_DIR.iterdir() if p.is_dir())
    print("test set:", {f: len(list((kc.TEST_SET_DIR / f).glob('*.wav'))) for f in folders})


if __name__ == "__main__":
    main()
