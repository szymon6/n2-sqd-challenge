"""Symmetry classes of a basis, dead product-space fraction, kicks and the curvature probe (stage 6)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from n2sqd import symmetry
from n2sqd.hamiltonian import N2Hamiltonian
from n2sqd.result import Result
from n2sqd.rotation import rotate_integrals
from n2sqd.solver import solve_subspace
from n2sqd.strings import lab_reference_strings


def test_rhf_basis_has_irrep_classes_and_selection_rules(ham):
    classes = symmetry.orbital_classes(ham)
    assert classes.shape == (26,) and classes.min() == 0
    n = classes.max() + 1
    assert n >= 4  # the canonical RHF orbitals carry at least the four C2v irreps
    m = symmetry.coupling_matrix(ham)
    i, j = np.nonzero(m > 1e-6)
    assert np.all(classes[i] == classes[j])
    bits = symmetry.irrep_bits(ham, classes)
    assert bits is not None and bits.shape == (n,) and len(set(bits.tolist())) == n
    # the four-index selection rule holds for every large integral
    idx = np.nonzero(np.abs(ham.eri) > 1e-6)
    ob = bits[classes]
    assert np.all((ob[idx[0]] ^ ob[idx[1]]) == (ob[idx[2]] ^ ob[idx[3]]))
    assert symmetry.irrep_bits(ham, np.zeros(26, dtype=int)).tolist() == [0]
    assert symmetry.irrep_bits(ham, np.array([0, 1, 2] + [0] * 23)) is None  # three classes are not a Z2^k group


def test_report_finds_exact_zeros_of_a_symmetric_subspace(ham):
    strings = lab_reference_strings(20)
    sol = solve_subspace(strings, strings, ham)
    result = Result(strings, strings, ham.hcore, ham.eri, ham.core_energy, sol.energy)
    rep = symmetry.report(result, sol.amplitudes)
    assert rep["n_classes"] >= 4 and sum(rep["class_sizes"]) == 26 and rep["cross_class_pairs"] > 0
    assert sum(rep["string_irreps_a"]) == 20 and rep["string_irreps_a"] == rep["string_irreps_b"]
    assert 0 < rep["allowed_fraction"] < 1
    assert rep["weight_in_allowed_block"] == pytest.approx(1.0, abs=1e-9)
    assert rep["dead_fraction"] >= 1 - rep["allowed_fraction"] - 1e-9
    assert "dead_fraction" not in symmetry.report(result)


def test_kick_breaks_the_classes(ham):
    classes = symmetry.orbital_classes(ham)
    u = symmetry.kick_rotation(classes, scale=0.05, seed=3, mode="cross")
    assert np.allclose(u.T @ u, np.eye(26), atol=1e-10) and not np.allclose(u, np.eye(26))
    # cross mode leaves same-class pairs untouched in the generator: K_ij = 0 for equal classes
    k = np.real_if_close(__import__("scipy.linalg", fromlist=["logm"]).logm(u))
    same = classes[:, None] == classes[None, :]
    assert np.abs(k[same]).max() < 1e-8
    h, g = rotate_integrals(ham.hcore, ham.eri, u)
    kicked = N2Hamiltonian(h, g, ham.core_energy)
    assert symmetry.orbital_classes(kicked).max() == 0  # one class: no symmetry left
    u_all = symmetry.kick_rotation(classes, scale=0.05, seed=3, mode="all")
    assert np.allclose(u_all.T @ u_all, np.eye(26), atol=1e-10)
    mo = np.eye(28)
    kicked_mo = symmetry.kicked_orbitals(mo, u)
    assert np.allclose(kicked_mo[:2, :2], np.eye(2)) and np.allclose(kicked_mo[2:, 2:], u)
    with pytest.raises(ValueError):
        symmetry.kick_rotation(classes, 0.1, 0, mode="nope")
    targeted = symmetry.kick_rotation(classes, 0.1, 0, pairs=[(3, 0), (5, 7)])
    k_t = np.real_if_close(__import__("scipy.linalg", fromlist=["logm"]).logm(targeted))
    assert abs(abs(k_t[0, 3]) - 0.1) < 1e-8 and abs(abs(k_t[5, 7]) - 0.1) < 1e-8
    assert np.count_nonzero(np.abs(k_t) > 1e-8) == 4


def test_curvature_probe_is_even_in_theta_for_cross_pairs(ham):
    strings = lab_reference_strings(8)
    sol = solve_subspace(strings, strings, ham)
    result = Result(strings, strings, ham.hcore, ham.eri, ham.core_energy, sol.energy)
    classes = symmetry.orbital_classes(ham)
    cross = symmetry.class_pairs(classes, "cross")
    same = symmetry.class_pairs(classes, "same")
    assert len(cross) + len(same) == len(symmetry.class_pairs(classes, "all")) == 325
    rows = symmetry.curvature_probe(result, cross[:2], theta=0.05)
    assert len(rows) == 2
    for row in rows:
        assert row["class_i"] != row["class_j"]
        assert row["e_plus"] == pytest.approx(row["e_minus"], abs=1e-7)  # symmetric point: E(θ) = E(−θ)
        assert np.isfinite(row["curvature"])
    one = symmetry.curvature_probe(result, cross[:1], theta=0.05, one_sided=True)
    assert one[0]["e_minus"] is None and one[0]["curvature"] == pytest.approx(rows[0]["curvature"], abs=1e-4)


def test_cli_report_kick_probe(ham, tmp_path):
    strings = lab_reference_strings(6)
    sol = solve_subspace(strings, strings, ham)
    folder = tmp_path / "art"
    folder.mkdir()
    Result(strings, strings, ham.hcore, ham.eri, ham.core_energy, sol.energy).save(folder / "result.npz")
    np.save(folder / "orbitals.npy", np.asarray(ham.mo_coeff))
    assert symmetry.main(["report", str(folder), "--no-solve"]) == 0
    assert symmetry.main(["report", str(folder)]) == 0
    out = tmp_path / "kicked.npy"
    assert symmetry.main(["kick", "--folder", str(folder), "--out", str(out), "--scale", "0.02", "--seed", "1"]) == 0
    kicked = np.load(out)
    assert kicked.shape == (28, 28) and not np.allclose(kicked, ham.mo_coeff)
    assert symmetry.main(["kick", "--folder", str(folder), "--out", str(out), "--scale", "0.05", "--pairs", "0-3,4-5"]) == 0
    probe = tmp_path / "probe"
    assert symmetry.main(["probe", "--folder", str(folder), "--out", str(probe), "--theta", "0.1", "--slice", "0:1"]) == 0
    data = json.loads((probe / "curvature-0-1.json").read_text())
    assert len(data["rows"]) == 1 and (probe / "command.txt").exists() and (probe / "log.txt").exists()
    assert symmetry.main(["probe", "--folder", str(folder), "--out", str(probe), "--theta", "0.1", "--pairs", "same",
                          "--slice", ":1"]) == 0
