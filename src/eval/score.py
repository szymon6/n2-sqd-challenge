"""Score a result the grader's way.

``score_result`` validates, recomputes the energy with ``qiskit_addon_sqd.fermion.solve_sci`` exactly as
``lab4b._exbonus_answer`` does, optionally runs that grader function offline (it never contacts the
server; only ``grade_answer`` does), and reports the challenge metrics.

Command line::

    python -m eval.score results/best/track-b/result.npz [--no-grader] [--json out.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from eval.reference import CHEMICAL_ACCURACY_HA, best_estimate, load_reference
from eval.validate import claim_matches, validate_result
from n2sqd.hamiltonian import N2Hamiltonian, load_or_build
from n2sqd.result import Result
from n2sqd.solver import grader_energy


@dataclasses.dataclass
class Score:
    """Metrics of one result."""

    energy: float  # grader energy, rounded to 6 decimals (the ranking number)
    energy_raw: float
    claimed_energy: float
    matches: bool  # the grader's claim check
    dims: tuple[int, int]
    seconds_solve: float
    gap_to_reference_mha: float | None
    reference_label: str
    below_hf_mha: float
    below_ccsd_mha: float
    within_chemical_accuracy: bool | None
    grader_energy: float | None = None  # from qfest_grader's own code, offline
    grader_agrees: bool | None = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def score_result(
    result: Result,
    reference: N2Hamiltonian | None = None,
    run_grader_code: bool = True,
    reference_table: dict | None = None,
) -> Score:
    """Validate and score. Raises :class:`eval.validate.ValidationError` on an invalid result."""
    reference = reference or load_or_build()
    clean = validate_result(result, reference)
    ham = N2Hamiltonian(hcore=clean.hcore, eri=clean.eri, core_energy=clean.core_energy, norb=reference.norb, nelec=reference.nelec)
    e_raw, seconds = grader_energy(clean.strings_a, clean.strings_b, ham)
    energy = round(e_raw, 6)
    table = reference_table or load_reference()
    e_ref, label = best_estimate(table)
    gap = None if e_ref is None else (energy - e_ref) * 1e3
    score = Score(
        energy=energy,
        energy_raw=e_raw,
        claimed_energy=clean.claimed_energy,
        matches=claim_matches(clean.claimed_energy, e_raw),
        dims=(len(clean.strings_a), len(clean.strings_b)),
        seconds_solve=seconds,
        gap_to_reference_mha=gap,
        reference_label=label,
        below_hf_mha=(table["e_hf"] - energy) * 1e3,
        below_ccsd_mha=(table["e_ccsd"] - energy) * 1e3,
        within_chemical_accuracy=None if gap is None else gap <= CHEMICAL_ACCURACY_HA * 1e3,
    )
    if run_grader_code:
        facts = grader_facts(clean)
        if facts is not None:
            score.grader_energy = facts["energy"]
            score.grader_agrees = facts["energy"] == energy and facts["matches"] == score.matches
    return score


def grader_facts(result: Result) -> dict | None:
    """The exact facts the official grader would send, computed offline. ``None`` if qfest_grader is missing."""
    try:
        from qfest_grader.challenges.warsaw_2026 import lab4b
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return lab4b._exbonus_answer(result.grader_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a result artifact the grader's way.")
    parser.add_argument("result", type=Path, help="path to result.npz")
    parser.add_argument("--no-grader", action="store_true", help="skip the offline qfest_grader cross-check")
    parser.add_argument("--json", type=Path, default=None, help="also write the metrics to this file")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    result = Result.load(args.result)
    score = score_result(result, run_grader_code=not args.no_grader)
    out = score.as_dict()
    out["seconds_total"] = time.perf_counter() - t0
    out["result"] = str(args.result)
    out["meta"] = result.meta
    text = json.dumps(out, indent=2, default=_default)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    return 0


def _default(obj):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
