"""Result artifacts: what a method produces and what :mod:`eval` scores.

One ``.npz`` file holds the five grader inputs and a JSON ``meta`` string (id, track, description,
reproduce command, seeds, run time). ``results/best/<track>/result.npz`` is the current best per track.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass
class Result:
    """The grader inputs plus metadata."""

    strings_a: np.ndarray
    strings_b: np.ndarray
    hcore: np.ndarray
    eri: np.ndarray
    core_energy: float
    claimed_energy: float
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            strings_a=np.asarray(self.strings_a, dtype=np.int64),
            strings_b=np.asarray(self.strings_b, dtype=np.int64),
            hcore=self.hcore,
            eri=self.eri,
            core_energy=float(self.core_energy),
            claimed_energy=float(self.claimed_energy),
            meta=json.dumps(self.meta, default=_json_default),
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Result":
        d = np.load(path, allow_pickle=False)
        return cls(
            strings_a=d["strings_a"],
            strings_b=d["strings_b"],
            hcore=d["hcore"],
            eri=d["eri"],
            core_energy=float(d["core_energy"]),
            claimed_energy=float(d["claimed_energy"]),
            meta=json.loads(str(d["meta"])),
        )

    @property
    def grader_args(self) -> tuple:
        """Positional arguments of ``grade_lab4b_exbonus``."""
        return self.claimed_energy, (self.strings_a, self.strings_b), self.hcore, self.eri, self.core_energy


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
