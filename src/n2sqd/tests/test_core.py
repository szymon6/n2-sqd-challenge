"""Hamiltonian, strings, rotation, solver and result round trips."""

from __future__ import annotations

import numpy as np
import pytest

from n2sqd.hamiltonian import N2Hamiltonian, load_or_build
from n2sqd.result import Result
from n2sqd.rotation import check_orthogonal, natural_orbital_rotation, rotate_integrals, rotation_from_params
from n2sqd.solver import grader_energy, marginal_weights, solve_subspace
from n2sqd.strings import (
    all_strings,
    canonical,
    excitations,
    from_bits,
    hf_string,
    lab_reference_strings,
    popcount,
    to_bits,
)


def test_hamiltonian_matches_lab(ham):
    assert ham.norb == 26 and ham.nelec == (5, 5)
    assert ham.e_hf == pytest.approx(-108.330582753660, abs=1e-8)
    assert ham.core_energy == pytest.approx(-82.851493365509, abs=1e-8)
    assert ham.hcore.shape == (26, 26) and ham.eri.shape == (26,) * 4
    np.testing.assert_allclose(ham.hcore, ham.hcore.T, atol=1e-12)


def test_hamiltonian_cache_round_trip(ham, tmp_path):
    path = tmp_path / "ints.npz"
    ham.save(path)
    loaded = N2Hamiltonian.load(path)
    np.testing.assert_allclose(loaded.eri, ham.eri)
    assert loaded.nelec == ham.nelec and loaded.mo_coeff.shape == ham.mo_coeff.shape
    again = load_or_build(path)
    assert again.e_hf == ham.e_hf
    bare = N2Hamiltonian(ham.hcore, ham.eri, ham.core_energy)
    bare.save(tmp_path / "bare.npz")
    assert N2Hamiltonian.load(tmp_path / "bare.npz").mo_coeff is None


def test_strings_helpers():
    assert hf_string() == 31
    assert popcount(np.array([31, 0, 1 << 25])).tolist() == [5, 0, 1]
    bits = to_bits(np.array([31, 33]))
    assert bits.shape == (2, 26) and bits[0, :5].all() and bits[1, 5]
    assert from_bits(bits).tolist() == [31, 33]
    full = all_strings()
    assert full.size == 65780 and np.all(np.diff(full) > 0) and np.all(popcount(full) == 5)
    groups = excitations(31, 26, 2)
    assert [len(g) for g in groups] == [1, 105, 2100]
    ref = lab_reference_strings(300)
    assert ref.size == 300 and np.all(np.diff(ref) > 0) and np.all(popcount(ref) == 5) and 31 in ref
    assert canonical([5, 3, 3]).tolist() == [3, 5]


def test_rotation_round_trip(ham):
    rng = np.random.default_rng(2)
    u = rotation_from_params(rng.normal(scale=0.1, size=325), 26)
    check_orthogonal(u)
    h, g = rotate_integrals(ham.hcore, ham.eri, u)
    h2, g2 = rotate_integrals(h, g, u.T)
    np.testing.assert_allclose(h2, ham.hcore, atol=1e-10)
    np.testing.assert_allclose(g2, ham.eri, atol=1e-9)
    with pytest.raises(ValueError):
        check_orthogonal(np.ones((26, 26)))
    with pytest.raises(ValueError):
        check_orthogonal(np.ones((2, 3)))
    with pytest.raises(ValueError):
        rotation_from_params(np.zeros(3), 26)
    dm = np.diag(np.linspace(2, 0, 26))
    vecs, occ = natural_orbital_rotation(dm)
    assert occ[0] == pytest.approx(2.0) and vecs.shape == (26, 26)


def test_solver_matches_grader_path(ham):
    strings = lab_reference_strings(20)
    sol = solve_subspace(strings, strings, ham)
    e_grader, _ = grader_energy(strings, strings, ham)
    assert sol.energy == pytest.approx(e_grader, abs=1e-7)
    assert sol.dims == (20, 20) and sol.amplitudes.shape == (20, 20)
    assert sol.occupancies_a.sum() == pytest.approx(5.0, abs=1e-8)
    wa, wb = marginal_weights(sol.amplitudes)
    assert wa.sum() == pytest.approx(1.0) and wb.sum() == pytest.approx(1.0)
    warm = solve_subspace(strings, strings, ham, ci0=sol.amplitudes)
    assert warm.energy == pytest.approx(sol.energy, abs=1e-8)


def test_result_round_trip(ham, tmp_path):
    hf = np.array([31])
    res = Result(hf, hf, ham.hcore, ham.eri, ham.core_energy, ham.e_hf, {"id": "hf", "seed": np.int64(3), "v": np.arange(2)})
    path = res.save(tmp_path / "r.npz")
    back = Result.load(path)
    assert back.meta == {"id": "hf", "seed": 3, "v": [0, 1]}
    assert back.strings_a.dtype == np.int64 and back.claimed_energy == ham.e_hf
    assert len(back.grader_args) == 5
    with pytest.raises(TypeError):
        Result(hf, hf, ham.hcore, ham.eri, ham.core_energy, 0.0, {"bad": object()}).save(tmp_path / "bad.npz")


def test_solver_small_subspace_is_exact_from_any_warm_start(ham):
    # CI found a Davidson miss on a tiny, nearly degenerate subspace (stage 7 close-out); small spaces are solved exactly
    from pyscf import ao2mo
    from pyscf.fci import direct_spin1, selected_ci

    strings = lab_reference_strings(6)
    dim = len(strings) ** 2
    h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(ham.hcore, ham.eri, ham.norb, ham.nelec, 0.5), ham.norb)
    dense = np.array([selected_ci.contract_2e(h2e, selected_ci._as_SCIvector(np.eye(dim)[i].reshape(6, 6), (strings, strings)),
                                              ham.norb, ham.nelec).ravel() for i in range(dim)])
    evals, evecs = np.linalg.eigh((dense + dense.T) / 2)
    for ci0 in (None, evecs[:, -1], evecs[:, 1]):
        sol = solve_subspace(strings, strings, ham, ci0=ci0)
        assert sol.energy == pytest.approx(evals[0] + ham.core_energy, abs=1e-9)
        assert sol.amplitudes.shape == (6, 6) and np.linalg.norm(sol.amplitudes) == pytest.approx(1.0)
