"""The harness must reject everything the grader rejects."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eval.validate import ValidationError, claim_matches, validate_integrals, validate_result, validate_strings
from n2sqd.rotation import rotate_integrals, rotation_from_params


def test_valid_result_passes(hf_result, ham):
    clean = validate_result(hf_result, ham)
    assert clean.strings_a.dtype == np.int64 and clean.strings_a.tolist() == [31]
    assert clean.meta == {"id": "hf"}


@pytest.mark.parametrize(
    "strings, message",
    [
        ([47, 31], "ascending"),
        ([31, 31, 47], "ascending"),
        ([7], "exactly 5"),
        ([-31], "outside"),
        ([31 << 22], "outside"),
        ([], "empty"),
        ([[31]], "1-D"),
        ([1.5], "1-D integer"),
    ],
)
def test_invalid_strings_rejected(strings, message):
    arr = np.array(strings, dtype=np.int64) if strings == [] else np.array(strings)
    with pytest.raises(ValidationError, match=message):
        validate_strings(arr, 5, 26, "strings_a")


def test_too_many_strings_rejected():
    from n2sqd.strings import all_strings

    with pytest.raises(ValidationError, match="more than 300"):
        validate_strings(all_strings()[:301], 5, 26, "strings_a")


def test_exactly_300_strings_allowed():
    from n2sqd.strings import all_strings

    out = validate_strings(all_strings()[:300], 5, 26, "strings_a")
    assert out.size == 300


def test_rotated_integrals_pass(ham):
    rng = np.random.default_rng(0)
    u = rotation_from_params(rng.normal(scale=0.3, size=26 * 25 // 2), 26)
    h, g = rotate_integrals(ham.hcore, ham.eri, u)
    validate_integrals(h, g, ham.core_energy, ham)


def test_scaled_or_mixed_integrals_rejected(ham):
    u = np.eye(26)
    u[5:7, 5:7] = [[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]]
    h_rot, g_rot = rotate_integrals(ham.hcore, ham.eri, u)
    with pytest.raises(ValidationError, match="spectrum of hcore"):
        validate_integrals(1.01 * ham.hcore, ham.eri, ham.core_energy, ham)
    with pytest.raises(ValidationError, match="spectrum of eri\\+hcore"):
        validate_integrals(ham.hcore, g_rot, ham.core_energy, ham)
    with pytest.raises(ValidationError, match="spectrum of eri\\+hcore"):
        validate_integrals(h_rot, ham.eri, ham.core_energy, ham)


def test_wrong_core_energy_rejected(ham):
    with pytest.raises(ValidationError, match="core energy"):
        validate_integrals(ham.hcore, ham.eri, ham.core_energy + 1e-3, ham)


def test_bad_shapes_and_values_rejected(ham):
    with pytest.raises(ValidationError, match="hcore must be"):
        validate_integrals(ham.hcore, ham.eri[0], ham.core_energy, ham)
    bad = ham.hcore.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValidationError, match="non-finite"):
        validate_integrals(bad, ham.eri, ham.core_energy, ham)
    with pytest.raises(ValidationError, match="not numeric"):
        validate_integrals("x", ham.eri, ham.core_energy, ham)


def test_bad_claimed_energy_rejected(hf_result, ham):
    with pytest.raises(ValidationError, match="finite"):
        validate_result(dataclasses.replace(hf_result, claimed_energy=float("nan")), ham)
    with pytest.raises(ValidationError, match="number"):
        validate_result(dataclasses.replace(hf_result, claimed_energy="abc"), ham)


def test_claim_rule():
    assert claim_matches(-108.5, -108.5)
    assert claim_matches(-108.4, -108.5)  # claiming higher is fine
    assert claim_matches(-108.50005, -108.5)  # within 1e-4 below
    assert not claim_matches(-108.5002, -108.5)
