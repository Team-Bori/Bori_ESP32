"""BTF1 model package format (docs/PACKAGE_FORMAT.md).

Pure Python (no numpy, no tflite needed) for parsing/validating, so the Raspberry Pi
can use it as is. Building a package from a .tflite needs `numpy` and `tflite`
(see pc/requirements.txt).
"""
from __future__ import annotations

import math
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"BTF1"
FORMAT_VERSION = 1
HEADER_SIZE = 64
SECTION_ENTRY_SIZE = 16
TENSOR_DESC_SIZE = 40
MAX_DIMS = 6
MODEL_ID_LEN = 32
TFLITE_ALIGN = 16
MAX_LABEL_DIM = 64

# Model partition (esp32_code/*/partitions.csv): 0x200000, 2 MB.
MODEL_PARTITION_OFFSET = 0x200000
MODEL_PARTITION_BYTES = 0x200000

TASKS = {"classification": 0, "regression": 1, "binary_score": 2}
TASK_NAMES = {v: k for k, v in TASKS.items()}

SEC_TFLITE = 1
SEC_TENSORS = 2
SEC_LABELS = 3
SEC_DEMO_INPUT = 4
SEC_DEMO_EXPECTED = 5
SEC_EVAL_INPUTS = 6
SEC_EVAL_EXPECTED = 7
SECTION_NAMES = {
    SEC_TFLITE: "tflite", SEC_TENSORS: "tensors", SEC_LABELS: "labels",
    SEC_DEMO_INPUT: "demo_input", SEC_DEMO_EXPECTED: "demo_expected",
    SEC_EVAL_INPUTS: "eval_inputs", SEC_EVAL_EXPECTED: "eval_expected",
}

# BTF dtype code -> (name, element size, numpy dtype string)
DTYPES = {
    1: ("int8", 1, "<i1"),
    2: ("uint8", 1, "<u1"),
    3: ("int16", 2, "<i2"),
    4: ("int32", 4, "<i4"),
    5: ("float32", 4, "<f4"),
}
DTYPE_CODES = {name: code for code, (name, _, _) in DTYPES.items()}

MODEL_ID_RE = re.compile(r"^[a-z0-9_]{1,31}$")

HEADER_FMT = "<4sHHII32sBBHIfI"
assert struct.calcsize(HEADER_FMT) == HEADER_SIZE
TENSOR_FMT = "<BBH6ifiI"
assert struct.calcsize(TENSOR_FMT) == TENSOR_DESC_SIZE

OPS_DEF = Path(__file__).resolve().parent.parent / "esp32_code" / "tflm_runtime" / "main" / "ops.def"


class PackageError(ValueError):
    """Invalid package. `code` is the matching firmware error code."""

    def __init__(self, message: str, code: str = "package_invalid"):
        super().__init__(message)
        self.code = code


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass
class TensorDesc:
    dtype: str
    shape: list[int]
    scale: float
    zero_point: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def bytes(self) -> int:
        return self.elements * DTYPES[DTYPE_CODES[self.dtype]][1]

    @property
    def np_dtype(self) -> str:
        return DTYPES[DTYPE_CODES[self.dtype]][2]

    def pack(self) -> bytes:
        if self.dtype not in DTYPE_CODES:
            raise PackageError(f"unsupported tensor dtype {self.dtype}")
        if not 1 <= len(self.shape) <= MAX_DIMS or any(d <= 0 for d in self.shape):
            raise PackageError(f"unsupported tensor shape {self.shape}")
        dims = list(self.shape) + [0] * (MAX_DIMS - len(self.shape))
        return struct.pack(TENSOR_FMT, DTYPE_CODES[self.dtype], len(self.shape), 0, *dims,
                           float(self.scale), int(self.zero_point), self.bytes)

    @classmethod
    def unpack(cls, data: bytes) -> "TensorDesc":
        code, ndim, _r, *rest = struct.unpack(TENSOR_FMT, data)
        dims, scale, zp, nbytes = rest[:MAX_DIMS], rest[MAX_DIMS], rest[MAX_DIMS + 1], rest[MAX_DIMS + 2]
        if code not in DTYPES or not 1 <= ndim <= MAX_DIMS:
            raise PackageError("bad tensor descriptor")
        t = cls(DTYPES[code][0], list(dims[:ndim]), scale, zp)
        if any(d <= 0 for d in t.shape) or t.bytes != nbytes:
            raise PackageError("bad tensor descriptor (shape/bytes)")
        return t

    def to_json(self) -> dict:
        return {"shape": list(self.shape), "dtype": self.dtype,
                "scale": float(self.scale), "zero_point": int(self.zero_point), "bytes": self.bytes}


@dataclass
class Package:
    model_id: str
    task: str
    tflite: bytes
    input: TensorDesc
    output: TensorDesc
    arena_bytes: int = 0          # 0 = auto (largest free block on the board)
    task_param: float = 0.0       # binary_score: threshold, regression: tolerance
    labels: list[str] | None = None
    demo_input: bytes | None = None
    demo_expected: list[float] | None = None
    eval_inputs: list[bytes] = field(default_factory=list)
    eval_expected: list[list[float]] = field(default_factory=list)
    # filled by parse()
    total_size: int = 0
    checksum: int = 0
    sections: list[dict] = field(default_factory=list)

    @property
    def label_dim(self) -> int:
        return self.output.elements if self.task == "regression" else 1


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def build(pkg: Package) -> bytes:
    """Serializes `pkg` and returns the package bytes (also sets total_size/checksum)."""
    check_semantics(pkg)
    label_dim = pkg.label_dim

    sections: list[tuple[int, bytes, int, int]] = []  # (type, data, count, align)
    sections.append((SEC_TFLITE, pkg.tflite, 0, TFLITE_ALIGN))
    sections.append((SEC_TENSORS, pkg.input.pack() + pkg.output.pack(), 2, 4))
    if pkg.labels:
        sections.append((SEC_LABELS, b"".join(l.encode("utf-8") + b"\0" for l in pkg.labels),
                         len(pkg.labels), 4))
    if pkg.demo_input is not None:
        sections.append((SEC_DEMO_INPUT, pkg.demo_input, 1, 4))
        if pkg.demo_expected is not None:
            sections.append((SEC_DEMO_EXPECTED, struct.pack(f"<{label_dim}f", *pkg.demo_expected), 1, 4))
    if pkg.eval_inputs:
        sections.append((SEC_EVAL_INPUTS, b"".join(pkg.eval_inputs), len(pkg.eval_inputs), 4))
        flat = [v for e in pkg.eval_expected for v in e]
        sections.append((SEC_EVAL_EXPECTED, struct.pack(f"<{len(flat)}f", *flat), len(pkg.eval_expected), 4))

    table_end = HEADER_SIZE + SECTION_ENTRY_SIZE * len(sections)
    body = bytearray()
    entries = []
    pos = table_end
    for stype, data, count, align in sections:
        aligned = _align(pos, align)
        body += b"\0" * (aligned - pos)
        entries.append(struct.pack("<IIII", stype, aligned, len(data), count))
        body += data
        pos = aligned + len(data)
    total = pos

    model_id = pkg.model_id.encode("ascii").ljust(MODEL_ID_LEN, b"\0")
    header_wo_crc = struct.pack(HEADER_FMT, MAGIC, FORMAT_VERSION, HEADER_SIZE, total, 0, model_id,
                                TASKS[pkg.task], len(sections), label_dim, int(pkg.arena_bytes),
                                float(pkg.task_param), 0)
    blob = bytearray(header_wo_crc + b"".join(entries) + bytes(body))
    assert len(blob) == total
    crc = crc32(bytes(blob[16:]))
    struct.pack_into("<I", blob, 12, crc)
    pkg.total_size = total
    pkg.checksum = crc
    return bytes(blob)


def check_semantics(pkg: Package) -> None:
    """Checks that do not depend on the byte layout (shared by build and parse)."""
    if not MODEL_ID_RE.match(pkg.model_id):
        raise PackageError("model_id must be 1-31 chars of a-z, 0-9, _")
    if pkg.task not in TASKS:
        raise PackageError(f"task must be one of {sorted(TASKS)}")
    if pkg.arena_bytes < 0:
        raise PackageError("arena_bytes must be >= 0 (0 = auto)")
    if len(pkg.tflite) < 8 or pkg.tflite[4:8] != b"TFL3":
        raise PackageError("tflite data is not a TFLite flatbuffer (TFL3)")
    out_n = pkg.output.elements
    label_dim = pkg.label_dim
    if label_dim > MAX_LABEL_DIM:
        raise PackageError(f"regression output has {out_n} values; at most {MAX_LABEL_DIM} supported")
    if pkg.task == "binary_score" and out_n != 1:
        raise PackageError("binary_score needs a single output value")
    if pkg.task == "classification" and pkg.labels and len(pkg.labels) != out_n:
        raise PackageError(f"{len(pkg.labels)} labels but the output has {out_n} classes")
    if pkg.labels and any("\0" in l for l in pkg.labels):
        raise PackageError("labels must not contain NUL")
    in_bytes = pkg.input.bytes
    if pkg.demo_input is not None and len(pkg.demo_input) != in_bytes:
        raise PackageError(f"demo input is {len(pkg.demo_input)} bytes, input tensor is {in_bytes}")
    if pkg.demo_expected is not None:
        if pkg.demo_input is None:
            raise PackageError("demo_expected without demo_input")
        if len(pkg.demo_expected) != label_dim:
            raise PackageError(f"demo_expected needs {label_dim} values")
    if len(pkg.eval_inputs) != len(pkg.eval_expected):
        raise PackageError("eval inputs and expected values differ in count")
    for i, (x, y) in enumerate(zip(pkg.eval_inputs, pkg.eval_expected)):
        if len(x) != in_bytes:
            raise PackageError(f"eval sample {i} is {len(x)} bytes, input tensor is {in_bytes}")
        if len(y) != label_dim:
            raise PackageError(f"eval expected {i} needs {label_dim} values")
    if pkg.task in ("classification", "binary_score"):
        limit = out_n if pkg.task == "classification" else 2
        for y in ([pkg.demo_expected] if pkg.demo_expected else []) + pkg.eval_expected:
            if y[0] != int(y[0]) or not 0 <= y[0] < limit:
                raise PackageError(f"class label {y[0]} out of range 0..{limit - 1}")


def parse(data: bytes, partition_bytes: int = MODEL_PARTITION_BYTES) -> Package:
    """Parses and validates package bytes the same way the firmware does."""
    if len(data) < HEADER_SIZE or data[:4] != MAGIC:
        raise PackageError("bad magic (expected BTF1)")
    (magic, version, header_size, total, crc, model_id_raw, task, section_count, label_dim,
     arena, task_param, _flags) = struct.unpack_from(HEADER_FMT, data)
    if version != FORMAT_VERSION:
        raise PackageError(f"unsupported format version {version}")
    table_end = header_size + section_count * SECTION_ENTRY_SIZE
    if header_size < HEADER_SIZE or total < table_end or total > len(data):
        raise PackageError(f"bad sizes: header {header_size}, total {total}, file {len(data)}")
    if total > partition_bytes:
        raise PackageError(f"package is {total} bytes; the model partition holds {partition_bytes}")
    if len(data) != total:
        raise PackageError(f"file is {len(data)} bytes but the header says {total}")
    actual = crc32(data[16:total])
    if actual != crc:
        raise PackageError(f"crc32 expected 0x{crc:08x}, actual 0x{actual:08x}", "checksum_mismatch")
    if b"\0" not in model_id_raw:
        raise PackageError("model_id is not NUL terminated")
    model_id = model_id_raw.split(b"\0", 1)[0].decode("ascii", "replace")
    if task not in TASK_NAMES:
        raise PackageError("unknown task")
    if not 1 <= label_dim <= MAX_LABEL_DIM:
        raise PackageError("label_dim out of range")

    secs = {}
    sections = []
    for i in range(section_count):
        stype, off, size, count = struct.unpack_from("<IIII", data, header_size + i * SECTION_ENTRY_SIZE)
        if off < table_end or off > total or size > total - off:
            raise PackageError(f"section {i} (type {stype}) out of bounds")
        if stype in secs:
            raise PackageError("duplicate section")
        secs[stype] = (off, size, count)
        sections.append({"type": SECTION_NAMES.get(stype, f"unknown_{stype}"), "offset": off,
                         "size": size, "count": count})

    def sec(t):
        if t not in secs:
            return None, 0
        off, size, count = secs[t]
        return data[off:off + size], count

    tfl, _ = sec(SEC_TFLITE)
    if tfl is None or SEC_TENSORS not in secs:
        raise PackageError("missing tflite or tensors section")
    if secs[SEC_TFLITE][0] % TFLITE_ALIGN or tfl[4:8] != b"TFL3":
        raise PackageError("tflite section must be 16-byte aligned and start with a TFL3 model")
    tens, tcount = sec(SEC_TENSORS)
    if tcount != 2 or len(tens) != 2 * TENSOR_DESC_SIZE:
        raise PackageError("bad tensor descriptors (need 1 input + 1 output)")
    tin = TensorDesc.unpack(tens[:TENSOR_DESC_SIZE])
    tout = TensorDesc.unpack(tens[TENSOR_DESC_SIZE:])

    labels = None
    lab, lcount = sec(SEC_LABELS)
    if lab is not None:
        parts = lab.split(b"\0")
        if len(parts) < lcount + 1:
            raise PackageError("labels section is malformed")
        labels = [p.decode("utf-8") for p in parts[:lcount]]

    in_bytes = tin.bytes
    demo_in, _ = sec(SEC_DEMO_INPUT)
    demo_exp_raw, _ = sec(SEC_DEMO_EXPECTED)
    demo_exp = None
    if demo_exp_raw is not None:
        if demo_in is None or len(demo_exp_raw) != 4 * label_dim:
            raise PackageError("bad demo expected section")
        demo_exp = list(struct.unpack(f"<{label_dim}f", demo_exp_raw))

    ev_in, ev_n = sec(SEC_EVAL_INPUTS)
    ev_exp, ev_n2 = sec(SEC_EVAL_EXPECTED)
    eval_inputs, eval_expected = [], []
    if ev_in is not None or ev_exp is not None:
        if ev_in is None or ev_exp is None or ev_n != ev_n2 or len(ev_in) != ev_n * in_bytes \
                or len(ev_exp) != ev_n * label_dim * 4:
            raise PackageError("eval inputs/expected sections do not match")
        eval_inputs = [ev_in[i * in_bytes:(i + 1) * in_bytes] for i in range(ev_n)]
        flat = struct.unpack(f"<{ev_n * label_dim}f", ev_exp)
        eval_expected = [list(flat[i * label_dim:(i + 1) * label_dim]) for i in range(ev_n)]

    pkg = Package(model_id=model_id, task=TASK_NAMES[task], tflite=tfl, input=tin, output=tout,
                  arena_bytes=arena, task_param=task_param, labels=labels, demo_input=demo_in,
                  demo_expected=demo_exp, eval_inputs=eval_inputs, eval_expected=eval_expected,
                  total_size=total, checksum=crc, sections=sections)
    if pkg.label_dim != label_dim:
        raise PackageError("label_dim does not match the task/output")
    check_semantics(pkg)
    return pkg


def evaluate_output(task: str, outputs: list[float], expected: list[float] | None,
                    task_param: float) -> dict:
    """Same decision rule as the firmware (main.cc evaluate_output)."""
    r: dict = {}
    if task == "classification":
        best = max(range(len(outputs)), key=lambda i: (outputs[i], -i))
        r["pred"], r["score"] = best, outputs[best]
        if expected is not None:
            r["correct"] = int(round(expected[0])) == best
    elif task == "binary_score":
        r["score"] = outputs[0]
        r["pred"] = 1 if outputs[0] >= task_param else 0
        if expected is not None:
            r["correct"] = int(round(expected[0])) == r["pred"]
    else:
        if expected is not None:
            err = max(abs(o - e) for o, e in zip(outputs, expected))
            r["abs_err"] = err
            r["correct"] = err <= task_param
    return r


def dequantize(t: TensorDesc, raw: bytes) -> list[float]:
    code = DTYPE_CODES[t.dtype]
    fmt = {1: "b", 2: "B", 3: "h", 4: "i", 5: "f"}[code]
    vals = struct.unpack(f"<{t.elements}{fmt}", raw)
    if t.dtype == "float32":
        return [float(v) for v in vals]
    # float32 arithmetic like the firmware
    return [float(struct.unpack("<f", struct.pack("<f", (v - t.zero_point) * t.scale))[0]) for v in vals]


def supported_ops(path: Path = OPS_DEF) -> list[str]:
    """BuiltinOperator names registered by tflm_runtime (from ops.def)."""
    text = path.read_text(encoding="utf-8")
    return re.findall(r"^BORI_OP\(\s*\w+\s*,\s*(\w+)\s*\)", text, re.M)


# ---------------------------------------------------------------------------
# tflite inspection (needs the `tflite` package)

_TFLITE_TYPES = {0: "float32", 2: "int32", 3: "uint8", 7: "int16", 9: "int8"}


def tflite_info(tflite_bytes: bytes) -> dict:
    """Ops and input/output tensors of a .tflite (first subgraph)."""
    import tflite  # pip install tflite

    model = tflite.Model.GetRootAsModel(tflite_bytes, 0)
    op_names = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_")}
    ops = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        name = op_names.get(code, f"builtin_{code}")
        if name == "CUSTOM":
            name = f"CUSTOM({(oc.CustomCode() or b'?').decode()})"
        ops.append(name)
    sg = model.Subgraphs(0)

    def tensor(idx):
        t = sg.Tensors(idx)
        q = t.Quantization()
        scales = q.ScaleAsNumpy() if q is not None and q.ScaleLength() else []
        zps = q.ZeroPointAsNumpy() if q is not None and q.ZeroPointLength() else []
        if len(scales) > 1:
            raise PackageError("per-channel quantized input/output tensors are not supported")
        dtype = _TFLITE_TYPES.get(t.Type())
        if dtype is None:
            raise PackageError(f"unsupported tensor type {t.Type()} for {t.Name().decode()}")
        return TensorDesc(dtype, [int(d) for d in t.ShapeAsNumpy()],
                          float(scales[0]) if len(scales) else 0.0,
                          int(zps[0]) if len(zps) else 0), t.Name().decode()

    inputs = [tensor(sg.Inputs(i)) for i in range(sg.InputsLength())]
    outputs = [tensor(sg.Outputs(i)) for i in range(sg.OutputsLength())]
    return {"ops": ops, "inputs": inputs, "outputs": outputs,
            "subgraphs": model.SubgraphsLength(), "version": model.Version()}
