#!/usr/bin/env python3
"""Build, inspect, validate and host-check BTF1 model packages for tflm_runtime.

  python pc/bori_package.py build    models/<id>/manifest.json
  python pc/bori_package.py inspect  models/<id>/default_model.bin
  python pc/bori_package.py validate models/<id>/default_model.bin
  python pc/bori_package.py check    models/<id>/manifest.json [--update-manifest]
  python pc/bori_package.py make-test  models/<id>/default_model.bin out.npz [--no-labels]
  python pc/bori_package.py check-test test.npz models/<id>/default_model.bin

See docs/ADDING_A_MODEL.md and docs/PACKAGE_FORMAT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import btf_format as bf  # noqa: E402

PC_DIR = Path(__file__).resolve().parent
REPO_ROOT = PC_DIR.parent

# tflite-micro (host check) only has Linux wheels; on Windows the check runs in WSL.
WSL_DISTRO = os.environ.get("BORI_WSL_DISTRO", "Ubuntu-22.04")
WSL_PYTHON = os.environ.get("BORI_WSL_PYTHON", "~/bori-tflm/bin/python")


def fail(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        fail(f"cannot read manifest {path}: {e}")


def save_manifest(path: Path, manifest: dict):
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def package_path(manifest_path: Path, manifest: dict) -> Path:
    return manifest_path.parent / manifest.get("files", {}).get("package", "default_model.bin")


# ---------------------------------------------------------------------------
# build

def _load_array(base: Path, ref):
    import numpy as np

    if isinstance(ref, str):
        return np.load(base / ref, allow_pickle=False)
    return np.asarray(ref)


def _inputs_as_bytes(arr, t: bf.TensorDesc, quantize: bool, what: str) -> list[bytes]:
    import numpy as np

    want = np.dtype(t.np_dtype)
    if arr.dtype != want:
        if quantize and arr.dtype.kind == "f" and t.dtype in ("int8", "uint8", "int16"):
            info = np.iinfo(want)
            arr = np.clip(np.round(arr / t.scale) + t.zero_point, info.min, info.max).astype(want)
        else:
            fail(f"{what}: dtype {arr.dtype} but the input tensor is {t.dtype} "
                 f"(store quantized values, or set build.quantize_inputs for float data)")
    shape = tuple(t.shape)
    per = t.elements
    if arr.size % per:
        fail(f"{what}: {arr.size} values is not a multiple of the input size {per} {list(shape)}")
    arr = np.ascontiguousarray(arr.reshape((-1,) + shape))
    return [arr[i].tobytes() for i in range(arr.shape[0])]


def _expected_rows(arr, label_dim: int, n: int, what: str) -> list[list[float]]:
    import numpy as np

    a = np.asarray(arr, dtype=np.float64).reshape(n, -1) if n else np.zeros((0, label_dim))
    if a.shape[1] != label_dim:
        fail(f"{what}: needs {label_dim} value(s) per sample, got {a.shape[1]}")
    return [list(map(float, r)) for r in a]


def cmd_build(args):
    mpath = args.manifest.resolve()
    m = load_manifest(mpath)
    base = mpath.parent
    b = m.get("build", {})
    files = m.setdefault("files", {})
    tflite_path = base / files.get("tflite", "model.tflite")
    tfl = tflite_path.read_bytes()

    info = bf.tflite_info(tfl)
    if len(info["inputs"]) != 1 or len(info["outputs"]) != 1:
        fail(f"tflm_runtime needs exactly 1 input and 1 output "
             f"(model has {len(info['inputs'])}/{len(info['outputs'])})")
    tin, tout = info["inputs"][0][0], info["outputs"][0][0]

    task = m.get("task")
    labels = b.get("labels")
    arena = b.get("arena_bytes", 0)
    if arena in ("auto", None):
        arena = 0
    pkg = bf.Package(model_id=m.get("model_id", ""), task=task, tflite=tfl, input=tin, output=tout,
                     arena_bytes=int(arena), task_param=float(b.get("task_param", 0.5 if task == "binary_score" else 0.0)),
                     labels=labels)
    quant = bool(b.get("quantize_inputs", False))
    if "demo_input" in b:
        demo = _inputs_as_bytes(_load_array(base, b["demo_input"]), tin, quant, "demo_input")
        if len(demo) != 1:
            fail("demo_input must hold exactly one sample")
        pkg.demo_input = demo[0]
        if "demo_expected" in b:
            pkg.demo_expected = _expected_rows(_load_array(base, b["demo_expected"]), pkg.label_dim, 1,
                                               "demo_expected")[0]
    if "eval_inputs" in b:
        pkg.eval_inputs = _inputs_as_bytes(_load_array(base, b["eval_inputs"]), tin, quant, "eval_inputs")
        pkg.eval_expected = _expected_rows(_load_array(base, b["eval_expected"]), pkg.label_dim,
                                           len(pkg.eval_inputs), "eval_expected")
    try:
        data = bf.build(pkg)
    except bf.PackageError as e:
        fail(str(e))

    out = args.out or package_path(mpath, m)
    files.setdefault("package", os.path.relpath(out, base).replace("\\", "/"))
    problems, warnings = validate_bytes(data)
    for w in warnings:
        print(f"warning: {w}")
    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        fail("package not written")
    out.write_bytes(data)

    m["firmware"] = "tflm_runtime"
    m["package_format"] = "BTF1"
    m["input"] = tin.to_json()
    m["output"] = tout.to_json()
    m["ops"] = info["ops"]
    m["package"] = {"size_bytes": len(data), "checksum": f"0x{pkg.checksum:08x}",
                    "sha256": hashlib.sha256(data).hexdigest(), "tflite_bytes": len(tfl),
                    "arena_bytes": pkg.arena_bytes, "eval_samples": len(pkg.eval_inputs),
                    "demo": pkg.demo_input is not None}
    save_manifest(mpath, m)
    print(f"wrote {out} ({len(data)} bytes, checksum 0x{pkg.checksum:08x}, "
          f"{len(pkg.eval_inputs)} eval samples)")
    print(f"checksum=0x{pkg.checksum:08x}")


# ---------------------------------------------------------------------------
# inspect / validate

def describe(pkg: bf.Package) -> dict:
    return {
        "model_id": pkg.model_id, "task": pkg.task, "size_bytes": pkg.total_size,
        "checksum": f"0x{pkg.checksum:08x}", "arena_bytes": pkg.arena_bytes or "auto",
        "task_param": pkg.task_param, "tflite_bytes": len(pkg.tflite),
        "input": pkg.input.to_json(), "output": pkg.output.to_json(),
        "labels": pkg.labels, "label_dim": pkg.label_dim,
        "demo": pkg.demo_input is not None, "demo_expected": pkg.demo_expected,
        "eval_samples": len(pkg.eval_inputs), "sections": pkg.sections,
    }


def validate_bytes(data: bytes) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errors, warnings = [], []
    try:
        pkg = bf.parse(data)
    except bf.PackageError as e:
        return [f"{e.code}: {e}"], warnings
    try:
        info = bf.tflite_info(pkg.tflite)
    except ImportError:
        warnings.append("`tflite` package missing: tensor and op checks skipped (pip install -r pc/requirements.txt)")
        return errors, warnings
    except bf.PackageError as e:
        return [str(e)], warnings
    if info["subgraphs"] != 1:
        warnings.append(f"model has {info['subgraphs']} subgraphs; only control-flow-free models are tested")
    if len(info["inputs"]) != 1 or len(info["outputs"]) != 1:
        errors.append("model must have exactly 1 input and 1 output")
    else:
        for name, desc, (actual, tname) in (("input", pkg.input, info["inputs"][0]),
                                            ("output", pkg.output, info["outputs"][0])):
            if actual.to_json() != desc.to_json():
                errors.append(f"{name} tensor '{tname}' {actual.to_json()} != package {desc.to_json()}")
            if desc.dtype != "int8":
                warnings.append(f"{name} tensor is {desc.dtype}; Bori models are int8 end to end")
    supported = set(bf.supported_ops())
    missing = sorted({op for op in info["ops"] if op not in supported})
    if missing:
        errors.append(f"unsupported_op: {', '.join(missing)} not registered in tflm_runtime "
                      "(esp32_code/tflm_runtime/main/ops.def)")
    if pkg.task == "classification" and not pkg.labels:
        warnings.append("classification package without labels")
    if not pkg.eval_inputs:
        warnings.append("no eval samples: the 'a' command will report no_eval_samples")
    if pkg.demo_input is None:
        warnings.append("no demo input: 'i', 'b' and metrics run on a zero input")
    return errors, warnings


def cmd_inspect(args):
    data = args.package.read_bytes()
    try:
        pkg = bf.parse(data)
    except bf.PackageError as e:
        fail(f"{e.code}: {e}")
    d = describe(pkg)
    try:
        d["ops"] = bf.tflite_info(pkg.tflite)["ops"]
    except ImportError:
        pass
    print(json.dumps(d, indent=2, ensure_ascii=False))


def cmd_validate(args):
    data = args.package.read_bytes()
    if args.max_bytes and len(data) > args.max_bytes:
        fail(f"package is {len(data)} bytes, limit {args.max_bytes}")
    errors, warnings = validate_bytes(data)
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    pkg = bf.parse(data)
    print("VALID")
    print(f"format=BTF1")
    print(f"firmware=tflm_runtime")
    print(f"model_id={pkg.model_id}")
    print(f"size={pkg.total_size}")
    print(f"checksum=0x{pkg.checksum:08x}")


# ---------------------------------------------------------------------------
# check (host TFLM interpreter)

def to_wsl_path(p: Path) -> str:
    p = p.resolve()
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}" + p.as_posix()[len(p.drive):]


def run_host_check(pkg_path: Path, max_eval: int) -> dict:
    script = PC_DIR / "tflm_check.py"
    extra = ["--max-eval", str(max_eval)] if max_eval else []
    try:
        import tflite_micro  # noqa: F401
        cmd = [sys.executable, str(script), str(pkg_path)] + extra
    except ImportError:
        if os.name != "nt" or shutil.which("wsl") is None:
            fail("host check needs `pip install tflite-micro` (Linux x86_64 only)")
        inner = " ".join([WSL_PYTHON, f"'{to_wsl_path(script)}'", f"'{to_wsl_path(pkg_path)}'"] + extra)
        cmd = ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", inner]
    proc = subprocess.run(cmd, capture_output=True)
    out = proc.stdout.decode("utf-8", "replace").replace("\0", "").strip().splitlines()
    for line in reversed(out):
        if line.startswith("{"):
            return json.loads(line)
    err = proc.stderr.decode("utf-8", "replace").replace("\0", "")
    fail(f"host check failed (exit {proc.returncode}):\n{err[-2000:]}\n"
         f"WSL setup: see docs/ADDING_A_MODEL.md (host check)")


def cmd_check(args):
    target = args.target.resolve()
    manifest = None
    if target.suffix == ".json":
        manifest = load_manifest(target)
        pkg_path = package_path(target, manifest)
    else:
        pkg_path = target
    data = pkg_path.read_bytes()
    errors, warnings = validate_bytes(data)
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    res = run_host_check(pkg_path, args.max_eval)
    if not res.get("ok"):
        fail(res.get("error", "host check failed"))
    arena = res.get("arena_bytes_host")
    print(f"host arena (64-bit TFLM reference kernels): {arena} bytes")
    print(f"board budget: see docs/ADDING_A_MODEL.md (max_arena_bytes); ESP-NN kernels may add scratch")
    if "demo" in res:
        print(f"demo: {json.dumps(res['demo'])}")
    ev = res.get("eval")
    if ev:
        line = f"eval: {ev['correct']}/{ev['samples']} correct (accuracy {ev['accuracy']})"
        if "mean_abs_err" in ev:
            line += f", mean abs err {ev['mean_abs_err']:.5f}, max {ev['max_abs_err']:.5f}"
        print(line + f", host {ev['host_avg_us']} us/inference")

    if manifest is not None and args.update_manifest:
        host = manifest.setdefault("host", {})
        host["arena_bytes_estimate"] = arena
        host["check_tool"] = "tflite-micro (pip, TFLM reference kernels, x86_64)"
        if ev:
            host["eval_samples"] = ev["samples"]
            host["eval_accuracy_int8"] = ev["accuracy"]
            if "mean_abs_err" in ev:
                host["eval_mean_abs_err"] = round(ev["mean_abs_err"], 6)
                host["eval_max_abs_err"] = round(ev["max_abs_err"], 6)
        save_manifest(target, manifest)
        print(f"updated {target}")


# ---------------------------------------------------------------------------
# server test files

def cmd_make_test(args):
    import numpy as np

    pkg = bf.parse(args.package.read_bytes())
    if not pkg.eval_inputs:
        fail("package has no eval samples")
    n = len(pkg.eval_inputs) if not args.count else min(args.count, len(pkg.eval_inputs))
    x = np.stack([np.frombuffer(s, dtype=pkg.input.np_dtype).reshape(pkg.input.shape)
                  for s in pkg.eval_inputs[:n]])
    arrays = {"x": x}
    if not args.no_labels:
        y = np.asarray(pkg.eval_expected[:n], dtype=np.float32)
        arrays["y"] = y[:, 0].astype(np.int64) if pkg.task != "regression" else y
    np.savez(args.out, **arrays)
    print(f"wrote {args.out}: x {list(x.shape)} {x.dtype}" + ("" if args.no_labels else f", y {list(arrays['y'].shape)}"))


def cmd_check_test(args):
    import test_data as td

    pkg = bf.parse(args.package.read_bytes())
    try:
        x, y = td.load(args.test)
        data = td.prepare(x, y, input_shape=pkg.input.shape, input_dtype=pkg.input.dtype,
                          task=pkg.task, output_elements=pkg.output.elements, labels=pkg.labels)
    except td.TestDataError as e:
        fail(str(e))
    print(f"OK: {len(data.samples)} samples of {pkg.input.bytes} bytes, "
          f"{'with' if data.expected else 'without'} expected values")


def main():
    ap = argparse.ArgumentParser(description="BTF1 model package tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="manifest.json -> package .bin")
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("inspect", help="print the package contents as JSON")
    p.add_argument("package", type=Path)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("validate", help="structure, checksum, tensors, ops (prints checksum=0x...)")
    p.add_argument("package", type=Path)
    p.add_argument("--max-bytes", type=int, default=bf.MODEL_PARTITION_BYTES)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("check", help="validate + run on the host TFLM interpreter (arena, accuracy)")
    p.add_argument("target", type=Path, help="package .bin or manifest.json")
    p.add_argument("--max-eval", type=int, default=0)
    p.add_argument("--update-manifest", action="store_true", help="store host results in manifest.json")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("make-test", help="write a server test file (.npz) from the package eval samples")
    p.add_argument("package", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--no-labels", action="store_true")
    p.set_defaults(func=cmd_make_test)

    p = sub.add_parser("check-test", help="check a server test file against a package")
    p.add_argument("test", type=Path)
    p.add_argument("package", type=Path)
    p.set_defaults(func=cmd_check_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
