"""Shared fixtures: the reference Hamiltonian (built once per session) and small valid results."""

from __future__ import annotations

import numpy as np
import pytest

from n2sqd.hamiltonian import build_n2_hamiltonian
from n2sqd.result import Result
from n2sqd.strings import hf_string, lab_reference_strings


@pytest.fixture(scope="session")
def ham():
    return build_n2_hamiltonian()


@pytest.fixture
def hf_result(ham):
    """The 1 × 1 Hartree–Fock subspace; its energy is the HF energy."""
    hf = np.array([hf_string()], dtype=np.int64)
    return Result(hf, hf, ham.hcore, ham.eri, ham.core_energy, ham.e_hf, {"id": "hf"})


@pytest.fixture
def small_result(ham):
    """HF plus a few singles per spin: cheap to diagonalise, still a real subspace."""
    strings = lab_reference_strings(12)
    return Result(strings, strings, ham.hcore, ham.eri, ham.core_energy, 0.0, {"id": "small"})
