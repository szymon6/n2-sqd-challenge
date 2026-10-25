"""JSON encoding of NumPy values in the score CLI output."""

import json

import numpy as np

from eval.score import _default


def test_json_default_handles_numpy_and_falls_back_to_str():
    out = json.dumps({"a": np.int64(3), "b": np.arange(2), "c": object()}, default=_default)
    data = json.loads(out)
    assert data["a"] == 3 and data["b"] == [0, 1] and data["c"].startswith("<object")
