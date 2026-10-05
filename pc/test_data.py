"""Server test files (docs/TEST_DATA_FORMAT.md).

A test file holds inputs that are already in the model's input tensor format
(no preprocessing anywhere), plus optional expected values:

  .npz  x: [N, *input_shape]  (dtype = input tensor dtype, e.g. int8)
        y: optional. classification / binary_score: [N] class indices (int) or label strings
                     regression: [N, output_values] (or [N] for one output value), float
  .npy  x only

Used by pc/bori_package.py (make-test / check-test) and rpi/esp_monitor.py (run-test).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NP_DTYPES = {"int8": np.int8, "uint8": np.uint8, "int16": np.int16, "int32": np.int32,
             "float32": np.float32}


class TestDataError(ValueError):
    pass


@dataclass
class TestData:
    samples: list[bytes]
    expected: list[list[float]] | None


def load(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if "x" not in z.files:
                raise TestDataError(f"{path.name}: .npz needs an 'x' array (has {z.files})")
            x = z["x"]
            y = z["y"] if "y" in z.files else None
    elif path.suffix == ".npy":
        x, y = np.load(path, allow_pickle=False), None
    else:
        raise TestDataError("test file must be .npz (x, optional y) or .npy (x)")
    return x, y


def prepare(x: np.ndarray, y: np.ndarray | None, *, input_shape: list[int], input_dtype: str,
            task: str, output_elements: int, labels: list[str] | None = None) -> TestData:
    """Validates x/y against the model and returns the raw sample bytes and expected values."""
    want = NP_DTYPES.get(input_dtype)
    if want is None:
        raise TestDataError(f"unsupported input dtype {input_dtype}")
    if x.dtype != want:
        raise TestDataError(f"x dtype is {x.dtype}, the model input is {input_dtype} "
                            "(test files must already be quantized; no preprocessing is done)")
    shape = tuple(input_shape)
    no_batch = shape[1:] if len(shape) > 1 and shape[0] == 1 else None
    if x.shape == shape or (no_batch is not None and x.shape == no_batch):
        x = x.reshape((1,) + shape)
    elif x.shape[1:] == shape:
        pass
    elif no_batch is not None and x.shape[1:] == no_batch:
        x = x.reshape((x.shape[0],) + shape)
    else:
        raise TestDataError(f"x shape {list(x.shape)} does not match [N, {', '.join(map(str, shape))}]")
    n = x.shape[0]
    if n == 0:
        raise TestDataError("x has no samples")
    x = np.ascontiguousarray(x)
    samples = [x[i].tobytes() for i in range(n)]

    if y is None:
        return TestData(samples, None)
    if y.shape[0] != n:
        raise TestDataError(f"y has {y.shape[0]} entries, x has {n} samples")
    if task == "regression":
        yf = np.asarray(y, dtype=np.float64).reshape(n, -1)
        if yf.shape[1] != output_elements:
            raise TestDataError(f"regression y needs {output_elements} values per sample")
        return TestData(samples, [list(map(float, r)) for r in yf])

    if y.dtype.kind in "US":
        if not labels:
            raise TestDataError("y holds label strings but the model has no labels")
        index = {l: i for i, l in enumerate(labels)}
        try:
            yi = [index[str(v)] for v in y.reshape(n)]
        except KeyError as e:
            raise TestDataError(f"unknown label {e} (labels: {labels})") from None
    else:
        if y.dtype.kind not in "iub":
            raise TestDataError(f"classification y must be integers or label strings, got {y.dtype}")
        yi = [int(v) for v in y.reshape(n)]
    limit = output_elements if task == "classification" else 2
    bad = [v for v in yi if not 0 <= v < limit]
    if bad:
        raise TestDataError(f"class index {bad[0]} out of range 0..{limit - 1}")
    return TestData(samples, [[float(v)] for v in yi])
