"""Evaluation harness: the referee for every result.

- :mod:`eval.validate` rejects anything the grader would reject (strings, integrals, claimed energy).
- :mod:`eval.score` recomputes the energy the grader's way, compares with the offline grader code, and
  reports the challenge metrics.
- :mod:`eval.reference` holds the reference energies (HF, CCSD, DMRG estimate) the gap is measured against.

Command line: ``python -m eval.score results/best/<track>/result.npz``.
"""

from eval.score import Score, score_result
from eval.validate import ValidationError, validate_result

__all__ = ["Score", "ValidationError", "score_result", "validate_result"]
