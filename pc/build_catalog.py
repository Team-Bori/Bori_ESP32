#!/usr/bin/env python3
"""Collects models/*/manifest.json into models/catalog.json.

Each model directory owns its manifest, so agents/people adding models in parallel
never edit the same file; run this script after merging to refresh the catalog.

  python pc/build_catalog.py           # write models/catalog.json
  python pc/build_catalog.py --check   # exit 1 if catalog.json is out of date
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CATALOG = MODELS / "catalog.json"

# Fields copied from each manifest (build inputs and op lists stay in the manifest).
FIELDS = ("model_id", "name", "description", "task", "firmware", "package_format",
          "input", "output", "package", "source", "dataset", "host", "board")
REQUIRED = ("model_id", "name", "task", "firmware", "package_format", "files")


def load_entries():
    entries = []
    for manifest in sorted(MODELS.glob("*/manifest.json")):
        if manifest.parent.name == "generated":
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED if k not in m]
        if missing:
            raise SystemExit(f"{manifest}: missing {missing}")
        entry = {k: m[k] for k in FIELDS if k in m}
        pkg = m["files"].get("package", "default_model.bin")
        entry["package_path"] = (manifest.parent / pkg).relative_to(ROOT).as_posix()
        if not (manifest.parent / pkg).exists():
            raise SystemExit(f"{manifest}: package {pkg} not found (run bori_package.py build)")
        entries.append(entry)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    catalog = {"version": 1, "models": load_entries()}
    text = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = CATALOG.read_text(encoding="utf-8") if CATALOG.exists() else ""
        if current != text:
            print("models/catalog.json is out of date; run python pc/build_catalog.py")
            return 1
        print("catalog up to date")
        return 0
    CATALOG.write_text(text, encoding="utf-8")
    print(f"wrote {CATALOG.relative_to(ROOT)} ({len(catalog['models'])} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
