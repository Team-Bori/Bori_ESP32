#!/usr/bin/env python3
"""ESP32 serial monitor for the Raspberry Pi relay.

Talks to the ESP32 firmwares using the JSON-line protocol
(v1: esp32_code/mlp_esp32/PROTOCOL.md, v2: docs/PROTOCOL_v2.md) and prints every
board message as one JSON line (NDJSON) on stdout, tagged with the port and a
timestamp. Optionally each message is also POSTed to a server URL.

Works with both firmwares:
  mlp           protocol v1 (m, i, b, p)
  tflm_runtime  protocol v2 (v1 commands + a, binary frames for streamed tests)

Examples:
  python3 esp_monitor.py /dev/ttyUSB0 info
  python3 esp_monitor.py /dev/ttyUSB0 infer
  python3 esp_monitor.py /dev/ttyUSB0 bench
  python3 esp_monitor.py /dev/ttyUSB0 eval                      # v2: package eval samples
  python3 esp_monitor.py /dev/ttyUSB0 run-test test.npz --post http://server/api/metrics
  python3 esp_monitor.py /dev/ttyUSB0 firmware-id
  python3 esp_monitor.py /dev/ttyUSB0 wait-ready --reset
  python3 esp_monitor.py /dev/ttyUSB0 verify --expect-checksum 0x12550a83 --expect-firmware mlp
  python3 esp_monitor.py /dev/ttyUSB0 listen --duration 60
  python3 esp_monitor.py /dev/ttyUSB0 monitor --interval 5 --count 12

Exit codes:
  0 success, 1 usage/serial error, 2 timeout, 3 board reported an error,
  4 board rebooted unexpectedly, 5 bad test file
"""

import argparse
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import serial

SUPPORTED_PROTOCOLS = (1, 2)
BAUD = 115200

EXIT_OK = 0
EXIT_SERIAL = 1
EXIT_TIMEOUT = 2
EXIT_BOARD_ERROR = 3
EXIT_REBOOT = 4
EXIT_BAD_INPUT = 5

# Binary frames (docs/PROTOCOL_v2.md)
FRAME_PING = 0x01
FRAME_SET_BAUD = 0x02
FRAME_TEST_BEGIN = 0x10
FRAME_TEST_SAMPLE = 0x11
FRAME_TEST_END = 0x12
FRAME_RETRY_CODES = ("crc_error", "frame_timeout")
FRAME_RETRIES = 3


class BoardError(Exception):
    def __init__(self, message, exit_code, payload=None):
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = payload or {}


def firmware_of(msg):
    """firmware_id from a boot/info message; v1 mlp builds did not send one."""
    return msg.get("firmware_id", "mlp")


def frame(ftype, seq, payload=b""):
    head = struct.pack("<BBI", ftype, seq & 0xFF, len(payload))
    crc = zlib.crc32(head + payload) & 0xFFFFFFFF
    return b"\xa5\x5a" + head + payload + struct.pack("<I", crc)


class EspBoard:
    def __init__(self, port, show_logs=False):
        self.port = port
        self.show_logs = show_logs
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = BAUD
        self.ser.timeout = 0.1
        # Keep DTR/RTS released so opening the port does not reset the board.
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self._buf = b""

    def close(self):
        self.ser.close()

    def reset(self):
        """Hardware reset through the auto-reset circuit (EN on RTS)."""
        self.ser.dtr = False
        self.ser.rts = True
        time.sleep(0.1)
        self.ser.rts = False
        self._buf = b""

    def set_host_baud(self, baud):
        self.ser.baudrate = baud
        self.ser.reset_input_buffer()
        self._buf = b""

    def _read_line(self, deadline):
        while time.monotonic() < deadline:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line, self._buf = self._buf[:nl], self._buf[nl + 1:]
                return line.decode("utf-8", "replace").strip()
            # Block for one byte (up to ser.timeout), then take whatever else is buffered.
            # Asking for a fixed count would wait out the timeout on every short reply.
            chunk = self.ser.read(1)
            if chunk and self.ser.in_waiting:
                chunk += self.ser.read(self.ser.in_waiting)
            self._buf += chunk
        return None

    def read_message(self, deadline):
        """Return the next JSON message, skipping log lines. None on timeout."""
        while True:
            line = self._read_line(deadline)
            if line is None:
                return None
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    log(f"ignored malformed JSON line: {line}")
                    continue
            if line and self.show_logs:
                log(line)

    def wait_for(self, expected_type, timeout, allow_boot=False, allow_error=False):
        """Wait for a message of expected_type, collecting everything seen on the way."""
        deadline = time.monotonic() + timeout
        seen = []
        while True:
            msg = self.read_message(deadline)
            if msg is None:
                raise BoardError(f"timed out waiting for '{expected_type}'", EXIT_TIMEOUT,
                                 {"seen": seen})
            seen.append(msg)
            mtype = msg.get("type")
            if mtype == expected_type:
                return msg, seen
            if mtype == "boot" and not allow_boot:
                raise BoardError("board rebooted unexpectedly", EXIT_REBOOT, msg)
            if mtype == "error" and expected_type != "error" and not allow_error:
                raise BoardError(msg.get("message", "board error"), EXIT_BOARD_ERROR, msg)

    def command(self, cmd, expected_type, timeout):
        self.ser.reset_input_buffer()
        self._buf = b""
        self.ser.write(cmd.encode())
        self.ser.flush()
        msg, _ = self.wait_for(expected_type, timeout)
        return msg

    def send_frame(self, ftype, seq, payload=b""):
        self.ser.write(frame(ftype, seq, payload))
        self.ser.flush()

    def frame_request(self, ftype, seq, payload, expected_type, timeout):
        """Sends a frame and waits for the reply with the same seq. Retries transport errors."""
        for attempt in range(FRAME_RETRIES + 1):
            self.send_frame(ftype, seq, payload)
            deadline = time.monotonic() + timeout
            while True:
                msg = self.read_message(deadline)
                if msg is None:
                    if attempt < FRAME_RETRIES:
                        break  # resend (the board answers a repeated sample without re-running it)
                    raise BoardError(f"timed out waiting for '{expected_type}' (seq {seq})", EXIT_TIMEOUT)
                mtype = msg.get("type")
                if mtype == "boot":
                    raise BoardError("board rebooted unexpectedly", EXIT_REBOOT, msg)
                # tflm_runtime builds up to 66a9038 sent test_summary without seq.
                if mtype == expected_type and msg.get("seq", seq & 0xFF) == seq & 0xFF:
                    return msg, attempt
                if mtype == "error" and msg.get("seq") in (seq & 0xFF, -1):
                    if msg.get("code") in FRAME_RETRY_CODES and attempt < FRAME_RETRIES:
                        time.sleep(0.15)  # the board drops input until the line is idle
                        self.ser.reset_input_buffer()
                        self._buf = b""
                        break
                    raise BoardError(msg.get("message", "board error"), EXIT_BOARD_ERROR, msg)
                # other messages (metrics, logs, stale replies) are ignored
        raise BoardError("frame retries exhausted", EXIT_BOARD_ERROR)


def log(text):
    print(text, file=sys.stderr, flush=True)


class Emitter:
    """Prints messages as NDJSON and optionally POSTs them to a server."""

    def __init__(self, port, post_url=None):
        self.port = port
        self.post_url = post_url

    def emit(self, msg, post=True):
        record = dict(msg)
        record["port"] = self.port
        record["ts"] = time.time()
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        if self.post_url and post:
            self._post(line)

    def _post(self, body):
        req = urllib.request.Request(
            self.post_url, data=body.encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception as e:  # never stop monitoring because the server is down
            log(f"POST to {self.post_url} failed: {e}")


def cmd_wait_ready(board, out, args):
    if args.reset:
        board.reset()
    _, seen = board.wait_for("ready", args.timeout, allow_boot=True, allow_error=True)
    for msg in seen:
        out.emit(msg)
    boot = next((m for m in seen if m.get("type") == "boot"), None)
    if boot and boot.get("proto") not in SUPPORTED_PROTOCOLS:
        log(f"warning: firmware protocol v{boot.get('proto')}, monitor supports {SUPPORTED_PROTOCOLS}")
    ready = seen[-1]
    if not ready.get("model_loaded"):
        return EXIT_BOARD_ERROR
    return EXIT_OK


def cmd_single(cmd, expected_type, timeout):
    def run(board, out, args):
        out.emit(board.command(cmd, expected_type, timeout))
        return EXIT_OK
    return run


def cmd_firmware_id(board, out, args):
    """Prints the firmware id (mlp | tflm_runtime). Used by the deploy scripts."""
    try:
        info = board.command("m", "info", 3)
    except BoardError:
        # Busy or just booting: reset and read the boot message instead.
        board.reset()
        boot, _ = board.wait_for("boot", args.timeout, allow_boot=True, allow_error=True)
        print(firmware_of(boot))
        return EXIT_OK
    print(firmware_of(info))
    return EXIT_OK


def cmd_monitor(board, out, args):
    """Periodically run a benchmark and report running statistics."""
    out.emit(board.command("m", "info", 5))

    runs = failures = reboots = 0
    fps_values = []
    try:
        while args.count == 0 or runs < args.count:
            runs += 1
            try:
                msg = board.command("b", "bench", 30)
                fps_values.append(msg.get("fps", 0.0))
                out.emit(msg)
            except BoardError as e:
                failures += 1
                if e.exit_code == EXIT_REBOOT:
                    reboots += 1
                out.emit({"type": "monitor_event", "event": "failure",
                          "reason": str(e), "detail": e.payload})
                if e.exit_code == EXIT_REBOOT:
                    # Let the board finish booting before the next round.
                    try:
                        board.wait_for("ready", 10, allow_boot=True, allow_error=True)
                    except BoardError:
                        pass

            out.emit({
                "type": "monitor_summary",
                "runs": runs,
                "failures": failures,
                "reboots": reboots,
                "error_rate": failures / runs,
                "avg_fps": sum(fps_values) / len(fps_values) if fps_values else None,
            })
            if args.count == 0 or runs < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return EXIT_OK if failures == 0 else EXIT_BOARD_ERROR


def cmd_verify(board, out, args):
    """Post-deploy check: reset, boot cleanly, stay up, and run the expected model."""
    failures = []
    exit_code = EXIT_OK

    board.reset()
    try:
        _, seen = board.wait_for("ready", args.timeout, allow_boot=True, allow_error=True)
    except BoardError as e:
        out.emit({"type": "verify", "ok": False, "failures": [f"no ready message: {e}"],
                  "detail": e.payload})
        return EXIT_TIMEOUT
    for msg in seen:
        out.emit(msg)
    ready = seen[-1]
    boot = next((m for m in reversed(seen) if m.get("type") == "boot"), {})
    load_errors = [m for m in seen if m.get("type") == "error"]

    if boot and boot.get("proto") not in SUPPORTED_PROTOCOLS:
        failures.append(f"unsupported protocol v{boot.get('proto')}")
        exit_code = EXIT_BOARD_ERROR
    if args.expect_firmware and boot and firmware_of(boot) != args.expect_firmware:
        failures.append(f"firmware {firmware_of(boot)} != expected {args.expect_firmware}")
        exit_code = EXIT_BOARD_ERROR

    # A boot loop (like a crash in app_main) shows up as another boot message.
    settle_deadline = time.monotonic() + args.settle
    while True:
        msg = board.read_message(settle_deadline)
        if msg is None:
            break
        out.emit(msg)
        if msg.get("type") == "boot":
            failures.append("board rebooted after becoming ready (boot loop)")
            exit_code = EXIT_REBOOT
            break

    if not ready.get("model_loaded") and not args.allow_no_model:
        reason = ", ".join(f"{e.get('code')}: {e.get('message')}" for e in load_errors)
        failures.append("model not loaded" + (f" ({reason})" if reason else ""))
        exit_code = exit_code or EXIT_BOARD_ERROR

    if args.expect_checksum and exit_code != EXIT_REBOOT:
        try:
            info = board.command("m", "info", 5)
            out.emit(info)
            actual = info.get("model", {}).get("checksum", "")
            if actual.lower() != args.expect_checksum.lower():
                failures.append(f"model checksum {actual or 'none'} != expected {args.expect_checksum}")
                exit_code = exit_code or EXIT_BOARD_ERROR
        except BoardError as e:
            failures.append(f"info request failed: {e}")
            exit_code = exit_code or e.exit_code

    out.emit({"type": "verify", "ok": not failures, "failures": failures,
              "firmware_id": firmware_of(boot) if boot else None, "proto": boot.get("proto")})
    return exit_code


def cmd_listen(board, out, args):
    """Forward the board's own periodic 'metrics' reports without sending commands."""
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    last_seq = None
    reboots = 0
    try:
        while deadline is None or time.monotonic() < deadline:
            wait_until = time.monotonic() + 1.0
            if deadline is not None:
                wait_until = min(wait_until, deadline)
            msg = board.read_message(wait_until)
            if msg is None:
                continue
            mtype = msg.get("type")
            if mtype == "boot":
                if last_seq is not None:
                    reboots += 1
                    out.emit({"type": "monitor_event", "event": "reboot",
                              "reboots": reboots, "detail": msg})
                last_seq = None
            elif mtype == "metrics":
                seq = msg.get("seq")
                if last_seq is not None and seq is not None and seq != last_seq + 1:
                    out.emit({"type": "monitor_event", "event": "metrics_gap",
                              "expected_seq": last_seq + 1, "got_seq": seq})
                last_seq = seq
            out.emit(msg)
    except KeyboardInterrupt:
        pass
    return EXIT_OK if reboots == 0 else EXIT_REBOOT


# ---------------------------------------------------------------------------
# run-test: stream a server test file to the board (protocol v2)

def switch_baud(board, baud, out):
    """Asks the board to change baud and follows it. Returns the baud in use afterwards."""
    current = board.ser.baudrate
    if baud == current:
        return current
    msg, _ = board.frame_request(FRAME_SET_BAUD, 0, struct.pack("<I", baud), "baud", 3)
    board.set_host_baud(baud)
    time.sleep(0.05)
    try:
        board.frame_request(FRAME_PING, 1, b"", "pong", 1)
        return baud
    except BoardError:
        log(f"no answer at {baud} baud; falling back to {current}")
        board.set_host_baud(current)
        try:
            board.wait_for("baud", msg.get("confirm_ms", 3000) / 1000 + 2, allow_error=True)
        except BoardError:
            pass
        out.emit({"type": "monitor_event", "event": "baud_fallback", "requested": baud, "using": current},
                 post=False)
        return current


def cmd_run_test(board, out, args):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pc"))
    import test_data as td  # needs numpy on the Raspberry Pi

    info = board.command("m", "info", 5)
    if info.get("proto", 1) < 2 or firmware_of(info) != "tflm_runtime":
        raise BoardError("run-test needs the tflm_runtime firmware (protocol v2)", EXIT_BOARD_ERROR, info)
    model = info.get("model", {})
    if not model.get("loaded"):
        raise BoardError("no model loaded on the board", EXIT_BOARD_ERROR, model)

    try:
        x, y = td.load(args.test_file)
        if args.no_labels:
            y = None
        data = td.prepare(x, y, input_shape=model["input"]["shape"], input_dtype=model["input"]["dtype"],
                          task=model["task"], output_elements=model["output_count"]
                          if "output_count" in model else _elements(model["output"]["shape"]),
                          labels=model.get("labels"))
    except (td.TestDataError, OSError, KeyError, ValueError) as e:
        out.emit({"type": "test_report", "ok": False, "error": f"bad test file: {e}",
                  "file": str(args.test_file)})
        return EXIT_BAD_INPUT
    samples = data.samples[:args.limit] if args.limit else data.samples
    expected = data.expected[:len(samples)] if data.expected else None
    has_labels = expected is not None

    baud = switch_baud(board, args.baud, out) if args.baud else board.ser.baudrate
    retries = 0
    results = []
    started = time.monotonic()
    try:
        begin_payload = struct.pack("<IBBH", len(samples), int(has_labels), int(args.report_outputs), 0)
        begin, _ = board.frame_request(FRAME_TEST_BEGIN, 0, begin_payload, "test_begin", 5)
        out.emit(begin, post=False)
        sample_bytes = len(samples[0]) + (4 * len(expected[0]) if has_labels else 0)
        # transfer time (10 bits per byte) + inference; generous so slow models do not time out
        timeout = sample_bytes * 10 / baud + args.sample_timeout
        for i, raw in enumerate(samples):
            payload = raw
            if has_labels:
                payload += struct.pack(f"<{len(expected[i])}f", *expected[i])
            msg, attempts = board.frame_request(FRAME_TEST_SAMPLE, (i + 1) & 0xFF, payload, "test_result", timeout)
            retries += attempts
            results.append(msg)
            if args.emit_samples:
                out.emit(msg, post=False)
        summary, _ = board.frame_request(FRAME_TEST_END, 0, b"", "test_summary", 10)
    finally:
        if baud != BAUD:
            try:
                switch_baud(board, BAUD, out)
            except BoardError as e:
                log(f"could not restore {BAUD} baud: {e}")
                board.set_host_baud(BAUD)
    wall = time.monotonic() - started

    out.emit(summary, post=False)
    report = {
        "type": "test_report", "ok": True, "file": Path(args.test_file).name,
        "model": summary.get("model"), "task": summary.get("task"),
        "samples": summary.get("samples"), "labeled": summary.get("labeled"),
        "correct": summary.get("correct"), "accuracy": summary.get("accuracy"),
        "mean_abs_err": summary.get("mean_abs_err"),
        "avg_us": summary.get("avg_us"), "min_us": summary.get("min_us"), "max_us": summary.get("max_us"),
        "invoke_errors": summary.get("invoke_errors"), "error_rate": summary.get("error_rate"),
        "memory": summary.get("memory"), "arena_used_bytes": model.get("arena_used_bytes"),
        "baud": baud, "transport_retries": retries, "wall_s": round(wall, 3),
        "firmware_id": firmware_of(info), "cpu_freq_mhz": info.get("board", {}).get("cpu_freq_mhz"),
    }
    out.emit({k: v for k, v in report.items() if v is not None})
    return EXIT_OK if not summary.get("invoke_errors") else EXIT_BOARD_ERROR


def _elements(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def main():
    ap = argparse.ArgumentParser(description="ESP32 serial monitor (mlp v1 / tflm_runtime v2)")
    ap.add_argument("port", help="serial port, e.g. /dev/ttyUSB0")
    ap.add_argument("--post", metavar="URL", help="also POST each message to this URL")
    ap.add_argument("--show-logs", action="store_true",
                    help="print the board's human-readable log lines to stderr")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("wait-ready", help="wait until the board has booted and is ready")
    p.add_argument("--reset", action="store_true", help="reset the board first")
    p.add_argument("--timeout", type=float, default=10.0)
    p.set_defaults(func=cmd_wait_ready)

    sub.add_parser("info", help="board specs, model and memory").set_defaults(
        func=cmd_single("m", "info", 5))
    sub.add_parser("infer", help="run one inference on the demo input").set_defaults(
        func=cmd_single("i", "inference", 15))
    sub.add_parser("bench", help="run the benchmark").set_defaults(
        func=cmd_single("b", "bench", 60))
    sub.add_parser("eval", help="v2: run all evaluation samples in the package").set_defaults(
        func=cmd_single("a", "eval", 900))

    p = sub.add_parser("firmware-id", help="print the firmware running on the board")
    p.add_argument("--timeout", type=float, default=10.0)
    p.set_defaults(func=cmd_firmware_id)

    p = sub.add_parser("verify", help="post-deploy check: clean boot, no boot loop, model loaded")
    p.add_argument("--expect-checksum", metavar="0xXXXXXXXX",
                   help="fail unless the loaded model has this checksum")
    p.add_argument("--expect-firmware", metavar="ID", help="fail unless this firmware (mlp, tflm_runtime) booted")
    p.add_argument("--allow-no-model", action="store_true",
                   help="do not fail when no model is loaded (firmware-only update)")
    p.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for ready")
    p.add_argument("--settle", type=float, default=3.0,
                   help="seconds to watch for a reboot after ready")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("listen", help="forward the board's periodic metrics reports")
    p.add_argument("--duration", type=float, default=0,
                   help="seconds to listen (0 = until Ctrl+C)")
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("monitor", help="run benchmarks periodically")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between runs")
    p.add_argument("--count", type=int, default=0, help="number of runs (0 = until Ctrl+C)")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("run-test", help="v2: stream a server test file (.npz/.npy) and report metrics")
    p.add_argument("test_file", type=Path)
    p.add_argument("--baud", type=int, default=921600,
                   help="serial speed during the transfer (115200/230400/460800/921600; 0 = keep)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N samples")
    p.add_argument("--no-labels", action="store_true", help="ignore y (latency/memory only)")
    p.add_argument("--report-outputs", action="store_true", help="ask for output values per sample")
    p.add_argument("--emit-samples", action="store_true", help="print every test_result line")
    p.add_argument("--sample-timeout", type=float, default=10.0,
                   help="seconds allowed per sample on top of the transfer time")
    p.set_defaults(func=cmd_run_test)

    args = ap.parse_args()

    try:
        board = EspBoard(args.port, show_logs=args.show_logs)
    except serial.SerialException as e:
        log(f"cannot open {args.port}: {e}")
        return EXIT_SERIAL

    out = Emitter(args.port, args.post)
    try:
        return args.func(board, out, args)
    except BoardError as e:
        out.emit({"type": "monitor_event", "event": "failure",
                  "reason": str(e), "detail": e.payload})
        return e.exit_code
    finally:
        board.close()


if __name__ == "__main__":
    sys.exit(main())
