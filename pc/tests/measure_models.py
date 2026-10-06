#!/usr/bin/env python3
"""Re-measures every tflm_runtime model on a connected board and updates its manifest.

For each models/*/manifest.json with firmware tflm_runtime (or the ones given):
  write the package to the model partition -> verify -> info -> bench -> eval
and stores the results in manifest["board"] (previous values are kept under
"board_history"). Run after a firmware change; the board must already run tflm_runtime.

  python pc/tests/measure_models.py --port COM6 [model_id ...] [--dry-run]

Writes the model partition (0x200000) only. Needs esptool (ESP-IDF python env).
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rpi"))
import esp_monitor as em  # noqa: E402

DEFAULT_ESPTOOL = str(Path.home() / ".espressif" / "python_env" / "idf5.3_py3.11_env" / "Scripts" / "python.exe")


def board_call(port, func):
    board = em.EspBoard(port)
    try:
        return func(board)
    finally:
        board.close()


def measure(port, esptool_python, manifest_path: Path, dry_run: bool):
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    pkg = manifest_path.parent / m["files"].get("package", "default_model.bin")
    checksum = m["package"]["checksum"]
    print(f"== {m['model_id']} ({pkg.stat().st_size} B, {checksum})")
    subprocess.run([esptool_python, "-m", "esptool", "--chip", "esp32", "-p", port, "-b", "921600",
                    "write_flash", "0x200000", str(pkg)], check=True, capture_output=True)

    def run(board):
        board.reset()
        _, seen = board.wait_for("ready", 15, allow_boot=True, allow_error=True)
        boot = next(x for x in seen if x.get("type") == "boot")
        info = board.command("m", "info", 5)
        if info.get("model", {}).get("checksum") != checksum:
            raise SystemExit(f"{m['model_id']}: board checksum {info.get('model', {}).get('checksum')} != {checksum}")
        bench = board.command("b", "bench", 60)
        ev = board.command("a", "eval", 900)
        return boot, info, bench, ev

    boot, info, bench, ev = board_call(port, run)
    board = {
        "measured_at": datetime.date.today().isoformat(),
        "firmware": f"tflm_runtime {boot.get('fw_version')}",
        "cpu_freq_mhz": info["board"]["cpu_freq_mhz"],
        "latency_us": {"bench_avg": round(bench["avg_us"], 1), "bench_min": bench["min_us"],
                       "bench_max": bench["max_us"], "bench_iterations": bench["iterations"],
                       "eval_avg": round(ev["avg_us"], 1)},
        "arena_used_bytes": info["model"]["arena_used_bytes"],
        "max_arena_bytes": info["model"]["max_arena_bytes"],
        "eval_samples": ev["samples"], "eval_accuracy": ev["accuracy"], "eval_wall_ms": ev["wall_ms"],
    }
    if "mean_abs_err" in ev:
        board["eval_mean_abs_err"] = ev["mean_abs_err"]
    old = m.get("board", {})
    print(f"   bench {old.get('latency_us', {}).get('bench_avg')} -> {board['latency_us']['bench_avg']} us, "
          f"eval {old.get('eval_accuracy')} -> {board['eval_accuracy']}, arena {board['arena_used_bytes']}")
    if host := m.get("host", {}).get("eval_accuracy_int8"):
        if abs(host - board["eval_accuracy"]) > 1e-9:
            print(f"   WARNING: board eval {board['eval_accuracy']} != host {host}")
    if not dry_run:
        if old:
            m.setdefault("board_history", []).append(old)
        m["board"] = {**old, **board}  # keep fields this script does not measure (stream tests ...)
        manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--esptool-python", default=DEFAULT_ESPTOOL)
    ap.add_argument("--dry-run", action="store_true", help="measure but do not update manifests")
    ap.add_argument("models", nargs="*")
    args = ap.parse_args()
    manifests = []
    for mp in sorted((ROOT / "models").glob("*/manifest.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("firmware") == "tflm_runtime" and (not args.models or m["model_id"] in args.models):
            manifests.append(mp)
    for mp in manifests:
        measure(args.port, args.esptool_python, mp, args.dry_run)


if __name__ == "__main__":
    main()
