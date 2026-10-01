#!/usr/bin/env python3
"""ESP32 MLP serial monitor for the Raspberry Pi relay.

Talks to the ESP32 firmware using the JSON-line protocol described in
esp32_code/mlp_esp32/PROTOCOL.md, and prints every board message as one JSON
line (NDJSON) on stdout, tagged with the port and a timestamp. Optionally each
message is also POSTed to a server URL.

Examples:
  python3 esp_monitor.py /dev/ttyUSB0 info
  python3 esp_monitor.py /dev/ttyUSB0 infer
  python3 esp_monitor.py /dev/ttyUSB0 bench
  python3 esp_monitor.py /dev/ttyUSB0 wait-ready --reset
  python3 esp_monitor.py /dev/ttyUSB0 verify --expect-checksum 0x12550a83
  python3 esp_monitor.py /dev/ttyUSB0 listen --duration 60
  python3 esp_monitor.py /dev/ttyUSB0 monitor --interval 5 --count 12
  python3 esp_monitor.py /dev/ttyUSB0 monitor --post http://server/api/metrics

Exit codes:
  0 success, 1 usage/serial error, 2 timeout, 3 board reported an error,
  4 board rebooted unexpectedly
"""

import argparse
import json
import sys
import time
import urllib.request

import serial

PROTOCOL_VERSION = 1
BAUD = 115200

EXIT_OK = 0
EXIT_SERIAL = 1
EXIT_TIMEOUT = 2
EXIT_BOARD_ERROR = 3
EXIT_REBOOT = 4


class BoardError(Exception):
    def __init__(self, message, exit_code, payload=None):
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = payload or {}


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

    def _read_line(self, deadline):
        while time.monotonic() < deadline:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line, self._buf = self._buf[:nl], self._buf[nl + 1:]
                return line.decode("utf-8", "replace").strip()
            self._buf += self.ser.read(256)
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


def log(text):
    print(text, file=sys.stderr, flush=True)


class Emitter:
    """Prints messages as NDJSON and optionally POSTs them to a server."""

    def __init__(self, port, post_url=None):
        self.port = port
        self.post_url = post_url

    def emit(self, msg):
        record = dict(msg)
        record["port"] = self.port
        record["ts"] = time.time()
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        if self.post_url:
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
    _, seen = board.wait_for("ready", args.timeout, allow_boot=True)
    for msg in seen:
        out.emit(msg)
    boot = next((m for m in seen if m.get("type") == "boot"), None)
    if boot and boot.get("proto") != PROTOCOL_VERSION:
        log(f"warning: firmware protocol v{boot.get('proto')}, monitor expects v{PROTOCOL_VERSION}")
    ready = seen[-1]
    if not ready.get("model_loaded"):
        return EXIT_BOARD_ERROR
    return EXIT_OK


def cmd_single(cmd, expected_type, timeout):
    def run(board, out, args):
        out.emit(board.command(cmd, expected_type, timeout))
        return EXIT_OK
    return run


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
                        board.wait_for("ready", 10, allow_boot=True)
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
        failures.append("model not loaded")
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

    out.emit({"type": "verify", "ok": not failures, "failures": failures})
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


def main():
    ap = argparse.ArgumentParser(description="ESP32 MLP serial monitor")
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
    sub.add_parser("infer", help="run one inference").set_defaults(
        func=cmd_single("i", "inference", 5))
    sub.add_parser("bench", help="run the benchmark").set_defaults(
        func=cmd_single("b", "bench", 30))

    p = sub.add_parser("verify", help="post-deploy check: clean boot, no boot loop, model loaded")
    p.add_argument("--expect-checksum", metavar="0xXXXXXXXX",
                   help="fail unless the loaded model has this checksum")
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
