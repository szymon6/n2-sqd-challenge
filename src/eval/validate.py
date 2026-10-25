"""Validity checks that mirror ``qfest_grader.challenges.warsaw_2026.lab4b._exbonus_answer``.

Every check here raises :class:`ValidationError` with a message that names the rule, so an invalid result
is rejected instead of scored. Rule source: the grader, ``qfest-grader`` 0.3.0.
"""

from __future__ import annotations

import numpy as np

from n2sqd.hamiltonian import N2Hamiltonian, load_or_build
from n2sqd.result import Result
from n2sqd.strings import popcount

MAX_STRINGS = 300
INTEGRAL_TOL = 1e-6
ENERGY_TOL = 1e-4


class ValidationError(ValueError):
    """The result breaks a grader rule."""


def validate_strings(strings, nelec: int, norb: int, what: str) -> np.ndarray:
    """Return the strings as ``int64`` or raise: 1-D integer array, 1..300 entries, ascending, unique, in range, right popcount."""
    arr = np.asarray(strings)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValidationError(f"{what} must be a 1-D integer array of CI strings")
    arr = arr.astype(np.int64)
    if arr.size == 0:
        raise ValidationError(f"{what} is empty")
    if arr.size > MAX_STRINGS:
        raise ValidationError(f"{what} has {arr.size} strings, more than {MAX_STRINGS}")
    if np.any(np.diff(arr) <= 0):
        raise ValidationError(f"{what} must be strictly ascending without duplicates")
    if np.any(arr < 0) or np.any(arr >= (1 << norb)):
        raise ValidationError(f"{what} has a string outside [0, 2**{norb})")
    bad = popcount(arr) != nelec
    if np.any(bad):
        raise ValidationError(f"{what} has {int(bad.sum())} strings without exactly {nelec} occupied orbitals")
    return arr


def spectra(hcore: np.ndarray, eri: np.ndarray) -> list[np.ndarray]:
    """The three spectra the grader compares (symmetrised)."""
    n = hcore.shape[0]
    eye = np.eye(n)
    pairs = eri.reshape(n * n, n * n)
    mats = (hcore, pairs, pairs + np.kron(hcore, eye) + np.kron(eye, hcore))
    return [np.linalg.eigvalsh((m + m.T) / 2) for m in mats]


def validate_integrals(hcore, eri, core_energy, reference: N2Hamiltonian) -> tuple[np.ndarray, np.ndarray, float]:
    """Shapes, finiteness, core energy and spectra within the grader tolerance."""
    n = reference.norb
    try:
        h = np.asarray(hcore, dtype=float)
        g = np.asarray(eri, dtype=float)
        core = float(core_energy)
    except (TypeError, ValueError) as err:
        raise ValidationError(f"integrals are not numeric ({err})") from None
    if h.shape != (n, n) or g.shape != (n,) * 4:
        raise ValidationError(f"hcore must be ({n}, {n}) and eri ({n}, {n}, {n}, {n}); got {h.shape} and {g.shape}")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(g)) and np.isfinite(core)):
        raise ValidationError("integrals contain non-finite values")
    if abs(core - reference.core_energy) > INTEGRAL_TOL:
        raise ValidationError(
            f"core energy {core:.9f} differs from the reference {reference.core_energy:.9f} by more than {INTEGRAL_TOL}"
        )
    for name, mine, ref in zip(("hcore", "eri", "eri+hcore"), spectra(h, g), spectra(reference.hcore, reference.eri)):
        if not np.allclose(mine, ref, rtol=0, atol=INTEGRAL_TOL):
            raise ValidationError(f"spectrum of {name} differs from the N2 reference integrals (not an orbital rotation)")
    return h, g, core


def validate_result(result: Result, reference: N2Hamiltonian | None = None) -> Result:
    """Run every grader check except the energy comparison. Returns a canonical copy."""
    reference = reference or load_or_build()
    a = validate_strings(result.strings_a, reference.nelec[0], reference.norb, "strings_a")
    b = validate_strings(result.strings_b, reference.nelec[1], reference.norb, "strings_b")
    h, g, core = validate_integrals(result.hcore, result.eri, result.core_energy, reference)
    try:
        claimed = float(result.claimed_energy)
    except (TypeError, ValueError):
        raise ValidationError("claimed_energy must be a number") from None
    if not np.isfinite(claimed):
        raise ValidationError("claimed_energy must be finite")
    return Result(a, b, h, g, core, claimed, dict(result.meta))


def claim_matches(claimed: float, recomputed: float) -> bool:
    """The grader's rule: a claim may sit above the recomputed energy, never more than 1e-4 below it."""
    return claimed >= recomputed - ENERGY_TOL
