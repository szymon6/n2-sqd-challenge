"""Starting orbital families: valid rotations of the RHF active space, core untouched (stage 6/7)."""

from __future__ import annotations

import numpy as np
import pytest

from eval.validate import validate_integrals
from n2sqd import orbitals
from n2sqd.hamiltonian import N_FROZEN
from n2sqd.orbopt import integrals_for, scf_reference


@pytest.fixture(scope="module")
def mf():
    return scf_reference()


def _check(mf, mo, new_ham, ham):
    s = mf.mol.intor("int1e_ovlp")
    assert mo.shape == (28, 28)
    assert np.allclose(mo.T @ s @ mo, np.eye(28), atol=1e-8)
    assert np.allclose(mo[:, :N_FROZEN], np.asarray(mf.mo_coeff)[:, :N_FROZEN])
    validate_integrals(new_ham.hcore, new_ham.eri, new_ham.core_energy, ham)  # the grader spectra check passes


def test_pipek_mezey_localises_on_atoms(mf, ham):
    mo, note = orbitals.pipek_mezey(mf)
    _check(mf, mo, integrals_for(mo, mf), ham)
    assert "Pipek" in note["orbitals"]
    # localised occupied orbitals: at least one of the five sits mostly on one atom (lone pair)
    s = mf.mol.intor("int1e_ovlp")
    aoslice = mf.mol.aoslice_by_atom()
    act = mo[:, N_FROZEN:N_FROZEN + 5]
    pop0 = (act[aoslice[0, 2]:aoslice[0, 3]] * (s @ act)[aoslice[0, 2]:aoslice[0, 3]]).sum(axis=0)
    assert np.abs(pop0 - 0.5).max() > 0.3
    assert not np.allclose(mo, mf.mo_coeff)


def test_uhf_natural_orbitals_break_spin_symmetry(mf, ham):
    mo, note = orbitals.uhf_natural(mf)
    _check(mf, mo, integrals_for(mo, mf), ham)
    assert note["e_uhf"] < ham.e_hf - 0.05  # the broken-symmetry UHF lies far below RHF at 2.0 A
    assert note["uhf_s2"] > 1.0
    occ = np.array(note["occupations"])
    assert occ.shape == (26,) and occ.sum() == pytest.approx(10.0, abs=1e-3)  # UHF core differs slightly from the RHF core
    assert np.all(np.diff(occ) <= 1e-12)


def test_split_localised_from_a_diagonal_rdm(mf, ham, tmp_path):
    occ = np.zeros(26)
    occ[:5] = 1.8
    occ[5:8] = 0.2
    occ[8:] = 0.2 / 18
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(26, 26)))
    dm = q @ np.diag(occ) @ q.T
    mo, note = orbitals.split_localised(mf, dm)
    _check(mf, mo, integrals_for(mo, mf), ham)
    assert note["n_strong"] == 5 and sum(note["occupations"]) == pytest.approx(occ.sum(), abs=1e-8)


def test_trackb_accepts_the_new_families(tmp_path):
    from n2sqd import trackb

    for name in ("pm-local", "uhf-no"):
        mo, note = trackb.starting_orbitals(trackb.TrackBConfig(out=tmp_path, orbitals=name), scf_reference())
        assert mo.shape == (28, 28) and note["orbitals"]
