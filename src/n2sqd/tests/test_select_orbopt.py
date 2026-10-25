"""Grow–prune selection and fixed-space orbital optimisation on small subspaces."""

from __future__ import annotations

import numpy as np
import pytest

from eval.validate import validate_integrals
from n2sqd import select
from n2sqd.orbopt import FixedSpaceSCI, active_rotation, integrals_for, optimize_orbitals, scf_reference
from n2sqd.rotation import rotate_integrals
from n2sqd.solver import grader_energy, solve_subspace
from n2sqd.strings import lab_reference_strings


def test_embed_places_amplitudes_and_normalises():
    old = (np.array([3, 5]), np.array([3, 5]))
    new = (np.array([3, 5, 6]), np.array([3, 6, 5]))  # new must be sorted for searchsorted; use sorted beta
    new = (np.array([3, 5, 6]), np.array([3, 5, 6]))
    amps = np.array([[0.8, 0.0], [0.0, 0.6]])
    out = select.embed(amps, old, new)
    assert out.shape == (3, 3) and out[0, 0] == pytest.approx(0.8) and out[1, 1] == pytest.approx(0.6)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    # strings missing from the new space are dropped
    out2 = select.embed(amps, old, (np.array([5, 6]), np.array([5, 6])))
    assert out2[0, 0] == pytest.approx(1.0)
    assert select.embed(amps, old, (np.array([6]), np.array([6]))).sum() == 0


def test_prune_symmetric_and_asymmetric(ham):
    strings = lab_reference_strings(12)
    sol = solve_subspace(strings, strings, ham)
    a, b = select.prune(sol, 5)
    assert a.size == 5 and np.array_equal(a, b) and 31 in a and np.all(np.diff(a) > 0)
    sol_asym = solve_subspace(strings, strings[:8], ham)
    a2, b2 = select.prune(sol_asym, 4)
    assert a2.size == 4 and b2.size == 4 and set(b2) <= set(strings[:8])
    # removal-loss criterion: positive losses, HF string kept, errors without the Hamiltonian / with a bad name
    la, lb = select.removal_losses(sol, ham)
    assert la.shape == (12,) and np.all(la >= 0) and np.allclose(la, lb, atol=1e-6)
    a3, b3 = select.prune(sol, 5, criterion="loss", ham=ham)
    assert a3.size == 5 and 31 in a3 and np.array_equal(a3, b3)
    a4, b4 = select.prune(sol, 5, symmetric=False, criterion="loss", ham=ham)
    assert a4.size == 5 and b4.size == 5
    with pytest.raises(ValueError):
        select.prune(sol, 5, criterion="loss")
    with pytest.raises(ValueError):
        select.prune(sol, 5, criterion="nope")
    # the loss-pruned space is never much worse than the weight-pruned one on this tiny example
    e_w = solve_subspace(a, b, ham).energy
    e_l = solve_subspace(a3, b3, ham).energy
    assert e_l < e_w + 5e-3


def test_grow_and_refine_lower_the_energy(ham):
    strings = lab_reference_strings(6)
    sol = solve_subspace(strings, strings, ham)
    pool = select.grow(sol, ham, select_cutoff=1e-2, ci_coeff_cutoff=1e-2, max_new=40)
    assert 6 < len(pool[0]) <= 46 and set(strings) <= set(pool[0])
    pool_unranked = select.grow(sol, ham, select_cutoff=1e-2, ci_coeff_cutoff=1e-2, max_new=40, rank=False)
    assert len(pool_unranked[0]) == len(pool[0])
    full = select.grow(sol, ham, select_cutoff=1e-2, ci_coeff_cutoff=1e-2, max_new=None)
    assert len(full[0]) >= len(pool[0])
    capped = select._cap_new(np.arange(100), np.arange(10), 5)
    assert capped.size == 15
    # importance: positive gains, and the top-ranked strings lower the energy more than the bottom-ranked ones
    new = np.setdiff1d(full[0], strings)
    gain = select.importance(sol, new, ham, spin=0)
    assert gain.shape == new.shape and np.all(gain >= 0)
    assert select.importance(sol, np.zeros(0, dtype=np.int64), ham).size == 0
    order = np.argsort(-gain)
    top = np.union1d(strings, new[order[:5]])
    bottom = np.union1d(strings, new[order[-5:]])
    assert solve_subspace(top, top, ham).energy < solve_subspace(bottom, bottom, ham).energy
    gain_b = select.importance(sol, new[:3], ham, spin=1)
    np.testing.assert_allclose(gain_b, gain[:3], rtol=1e-6)
    final, steps = select.refine((strings, strings), ham, budget=10, iterations=2, select_cutoff=1e-2,
                                 ci_coeff_cutoff=1e-2, max_new=40, log=None)
    assert len(steps) >= 1 and final.energy < sol.energy
    assert final.dims == (10, 10)


def test_gradual_prune_schedule_and_selection(ham):
    assert select.prune_schedule(1000, 300, 1) == [300]
    assert select.prune_schedule(200, 300, 4) == [300]
    sched = select.prune_schedule(1000, 300, 3)
    assert sched[-1] == 300 and sched == sorted(sched, reverse=True) and all(300 < s < 1000 for s in sched[:-1])
    assert len(select.prune_schedule(1000, 300, 6)) == 6
    with pytest.raises(ValueError):
        select.prune_schedule(1000, 300, 0)
    strings = lab_reference_strings(6)
    sol = solve_subspace(strings, strings, ham)
    pool = select.grow(sol, ham, select_cutoff=1e-2, ci_coeff_cutoff=1e-2, max_new=40)
    one_shot = select.select_from_pool(sol, pool, 10, ham)
    gradual = select.select_from_pool(sol, pool, 10, ham, prune_steps=3)
    assert one_shot.dims == gradual.dims == (10, 10)
    assert gradual.energy < sol.energy and one_shot.energy < sol.energy
    assert np.array_equal(gradual.strings_a, gradual.strings_b)


def test_fixed_space_solver_and_orbital_optimisation(ham):
    strings = lab_reference_strings(8)
    solver = FixedSpaceSCI((strings, strings))
    e, ci = solver.kernel(ham.hcore, ham.eri, 26, (5, 5), ecore=ham.core_energy)
    assert e == pytest.approx(solve_subspace(strings, strings, ham).energy, abs=1e-7)
    e2, _ = solver.kernel(ham.hcore, ham.eri, 26, (5, 5), ci0=np.asarray(ci), ecore=ham.core_energy)
    assert e2 == pytest.approx(e, abs=1e-8)
    e3, _ = solver.kernel(ham.hcore, ham.eri, 26, (5, 5), ci0=True, ecore=ham.core_energy)
    assert e3 == pytest.approx(e, abs=1e-8)

    # The reference SCF is pinned to the cached orbitals; a fresh RHF run (the `ham` fixture) may rotate the
    # degenerate pi orbitals differently, so compare in the cached basis and by spectra against the fixture.
    from n2sqd.hamiltonian import load_or_build

    mf = scf_reference()
    cached = load_or_build()
    ham0 = integrals_for(mf.mo_coeff, mf)
    np.testing.assert_allclose(ham0.hcore, cached.hcore, atol=1e-8)
    np.testing.assert_allclose(np.linalg.eigvalsh(ham0.hcore), np.linalg.eigvalsh(ham.hcore), atol=1e-8)
    assert ham0.core_energy == pytest.approx(ham.core_energy, abs=1e-9)
    e_c = solve_subspace(strings, strings, cached).energy
    opt = optimize_orbitals((strings, strings), mf.mo_coeff, mf, max_cycle_macro=2)
    assert opt.energy < e_c - 1e-5
    h, g = rotate_integrals(cached.hcore, cached.eri, opt.rotation)
    validate_integrals(h, g, cached.core_energy, ham)
    e_rot, _ = grader_energy(strings, strings, type(ham)(h, g, cached.core_energy))
    assert e_rot == pytest.approx(opt.energy, abs=1e-6)
    ham1 = integrals_for(opt.mo_coeff, mf)
    np.testing.assert_allclose(ham1.hcore, h, atol=1e-7)
    with pytest.raises(ValueError):
        active_rotation(mf.mo_coeff, np.roll(mf.mo_coeff, 1, axis=1), mf.mol.intor("int1e_ovlp"))
