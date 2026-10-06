#!/usr/bin/env python3
"""Runs a BTF1 package on the host with the TFLite Micro Python interpreter.

Needs the `tflite-micro` pip package, which only ships Linux x86_64 wheels.
On Windows, `bori_package.py check` runs this script inside WSL for you.

Prints one JSON object (last line of stdout):
  arena_bytes_host  arena used on the host (64-bit; the ESP32 value differs, see
                    docs/ADDING_A_MODEL.md "arena")
  demo / eval       predictions and accuracy with the same decision rule as the firmware
"""
import argparse
import ctypes
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import btf_format as bf  # noqa: E402

HOST_ARENA = 4 * 1024 * 1024


def capture_c_output(func):
    """Runs func() and returns (result, text the C++ code printed to fd 1 and fd 2).

    TFLM's MicroPrintf (used by print_allocations) writes to stderr.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        for fd in (1, 2):
            os.dup2(tmp.fileno(), fd)
        try:
            result = func()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            ctypes.CDLL(None).fflush(None)  # C stdio buffers of the TFLM extension
            for fd, keep in saved.items():
                os.dup2(keep, fd)
                os.close(keep)
        tmp.seek(0)
        return result, tmp.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=Path)
    ap.add_argument("--max-eval", type=int, default=0, help="evaluate at most N samples (0 = all)")
    args = ap.parse_args()

    from tflite_micro.python.tflite_micro import runtime

    pkg = bf.parse(args.package.read_bytes())

    def make():
        it = runtime.Interpreter.from_bytes(pkg.tflite, arena_size=HOST_ARENA)
        it.print_allocations()
        return it

    try:
        interp, text = capture_c_output(make)
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": f"host interpreter failed: {e}"}))
        return 1
    m = re.search(r"Arena allocation total (\d+) bytes", text)
    arena_host = int(m.group(1)) if m else None

    in_shape = tuple(pkg.input.shape)

    def run(raw: bytes):
        x = np.frombuffer(raw, dtype=pkg.input.np_dtype).reshape(in_shape)
        interp.set_input(x, 0)
        t0 = time.perf_counter()
        interp.invoke()
        dt = time.perf_counter() - t0
        out = np.ascontiguousarray(interp.get_output(0)).astype(pkg.output.np_dtype).tobytes()
        return bf.dequantize(pkg.output, out), dt

    result = {"ok": True, "model_id": pkg.model_id, "task": pkg.task,
              "arena_bytes_host": arena_host, "package_arena_bytes": pkg.arena_bytes}

    if pkg.demo_input is not None:
        outs, _ = run(pkg.demo_input)
        r = bf.evaluate_output(pkg.task, outs, pkg.demo_expected, pkg.task_param)
        r["output"] = outs[:16]
        result["demo"] = r

    n = len(pkg.eval_inputs)
    if args.max_eval:
        n = min(n, args.max_eval)
    if n:
        correct, abs_err, max_err, times, preds = 0, 0.0, 0.0, [], []
        for i in range(n):
            outs, dt = run(pkg.eval_inputs[i])
            r = bf.evaluate_output(pkg.task, outs, pkg.eval_expected[i], pkg.task_param)
            correct += int(r.get("correct", False))
            abs_err += r.get("abs_err", 0.0)
            max_err = max(max_err, r.get("abs_err", 0.0))
            times.append(dt)
            preds.append(r.get("pred"))
        ev = {"samples": n, "correct": correct, "accuracy": round(correct / n, 4),
              "host_avg_us": round(1e6 * sum(times) / n, 1)}
        if pkg.task == "regression":
            ev["mean_abs_err"] = abs_err / n
            ev["max_abs_err"] = max_err
        else:
            ev["predictions"] = preds
        result["eval"] = ev

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
