#!/usr/bin/env python3
"""Downloads the MLPerf Tiny KWS reference model at the pinned commit (sha256 checked) to
models/mlperf_kws_dscnn_int8/model.tflite. The Speech Commands v0.02 test set is shared with
kws_micro_speech (pc/models/kws_micro_speech/prepare.py)."""
import hashlib
import urllib.request
from pathlib import Path

COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
URL = ("https://raw.githubusercontent.com/mlcommons/tiny/"
       f"{COMMIT}/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite")
SHA256 = "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
OUT = Path(__file__).resolve().parents[3] / "models" / "mlperf_kws_dscnn_int8" / "model.tflite"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch for {OUT}: {digest}")
    print(f"{OUT} ok ({OUT.stat().st_size} B)")


if __name__ == "__main__":
    main()
