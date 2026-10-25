"""Orbital optimisation for a fixed product subspace (``strings_a × strings_b``).

The grader accepts the integrals in any orthogonal rotation of the 26 active orbitals.
For a fixed set of CI strings the energy depends on that rotation, so we
optimise it: PySCF's CASSCF machinery (``mcscf.CASSCF`` with ``internal_rotation=True`` and the two core
orbitals frozen) with :class:`FixedSpaceSCI`, a selected-CI solver that never changes its strings. This is
the orbital-optimisation step of SQD (arXiv:2405.05068, Sec. II A 4) done with a second-order optimiser
instead of gradient descent.

Everything is expressed through ``mo_coeff`` (AO → MO, 28 × 28, core columns untouched); the active
integrals for a given ``mo_coeff`` come from the same PySCF calls the grader uses.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from pyscf import fci

from n2sqd.hamiltonian import N_FROZEN, N2Hamiltonian, build_molecule


class FixedSpaceSCI(fci.selected_ci.SelectedCI):
    """PySCF selected-CI solver whose string sets are fixed (``kernel`` → ``kernel_fixed_space``)."""

    def __init__(self, ci_strs: tuple[np.ndarray, np.ndarray]):
        super().__init__()
        self.fixed_strs = (np.asarray(ci_strs[0], dtype=np.int64), np.asarray(ci_strs[1], dtype=np.int64))
        self.verbose = 0
        self.conv_tol = 1e-9
        self.max_cycle = 200

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        na, nb = len(self.fixed_strs[0]), len(self.fixed_strs[1])
        if isinstance(ci0, np.ndarray) and ci0.size == na * nb:
            ci0 = fci.selected_ci._as_SCIvector(np.asarray(ci0, dtype=float).reshape(na, nb), self.fixed_strs)
        else:
            ci0 = None  # CASSCF passes booleans or vectors from another space; start fresh
        e, ci = fci.selected_ci.kernel_fixed_space(self, h1e, eri, norb, nelec, ci_strs=self.fixed_strs, ci0=ci0)
        return e + ecore, ci


@dataclasses.dataclass
class OrbitalOptimisation:
    energy: float
    mo_coeff: np.ndarray
    rotation: np.ndarray  # active-space rotation U from the starting active orbitals
    converged: bool
    seconds: float
    macro_iterations: int


def scf_reference():
    """The lab's RHF object with the orbitals pinned to the cached ones (``data/n2_ints.npz``).

    Separate RHF runs may rotate the degenerate π orbitals differently (the grader's docstring warns about it,
    and Linux vs macOS BLAS do differ). Pinning ``mo_coeff`` to the cache keeps every run on one machine in
    the same orbital basis, so string sets and natural-orbital rotations stay meaningful across runs.
    """
    import pyscf

    from n2sqd.hamiltonian import load_or_build

    mol = build_molecule()
    mf = pyscf.scf.RHF(mol).run()
    cached = load_or_build()
    if cached.mo_coeff is not None and cached.mo_coeff.shape == mf.mo_coeff.shape:
        mf.mo_coeff = np.array(cached.mo_coeff)
        if cached.mo_energy is not None:
            mf.mo_energy = np.array(cached.mo_energy)
    return mf


def integrals_for(mo_coeff: np.ndarray, mf=None) -> N2Hamiltonian:
    """Active-space integrals for ``mo_coeff`` with the grader's PySCF calls (core = first two columns)."""
    import pyscf
    import pyscf.mcscf

    mf = mf or scf_reference()
    norb = mo_coeff.shape[1] - N_FROZEN
    cas = pyscf.mcscf.CASCI(mf, norb, (5, 5))
    hcore, core_energy = cas.get_h1cas(mo_coeff)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo_coeff), norb)
    return N2Hamiltonian(np.asarray(hcore), np.asarray(eri), float(core_energy), norb, (5, 5), e_hf=float(mf.e_tot),
                         mo_coeff=np.asarray(mo_coeff), mo_energy=None)


def active_rotation(mo_from: np.ndarray, mo_to: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Orthogonal matrix U with ``mo_to_active = mo_from_active @ U`` (columns 2: are active)."""
    u = mo_from[:, N_FROZEN:].T @ overlap @ mo_to[:, N_FROZEN:]
    err = np.abs(u.T @ u - np.eye(u.shape[0])).max()
    if err > 1e-6:
        raise ValueError(f"active orbitals did not stay in the active space (deviation {err:.1e})")
    return u


def optimize_orbitals(
    strings: tuple[np.ndarray, np.ndarray],
    mo_coeff: np.ndarray,
    mf=None,
    max_cycle_macro: int = 10,
    conv_tol: float = 1e-7,
    verbose: int = 0,
) -> OrbitalOptimisation:
    """Minimise the fixed-subspace energy over active-active orbital rotations starting from ``mo_coeff``."""
    from pyscf import mcscf

    mf = mf or scf_reference()
    norb = mo_coeff.shape[1] - N_FROZEN
    mc = mcscf.CASSCF(mf, norb, (5, 5))
    mc.frozen = list(range(N_FROZEN))
    mc.internal_rotation = True
    mc.fcisolver = FixedSpaceSCI(strings)
    mc.max_cycle_macro = max_cycle_macro
    mc.conv_tol = conv_tol
    mc.verbose = verbose
    t0 = time.perf_counter()
    mc.kernel(mo_coeff)
    overlap = mf.mol.intor("int1e_ovlp")
    u = active_rotation(mo_coeff, mc.mo_coeff, overlap)
    # the core columns must be untouched
    core_err = np.abs(mo_coeff[:, :N_FROZEN].T @ overlap @ mc.mo_coeff[:, :N_FROZEN] - np.eye(N_FROZEN)).max()
    if core_err > 1e-6:
        raise ValueError(f"core orbitals changed (deviation {core_err:.1e})")
    macro = getattr(mc, "_macro_iters", None)
    return OrbitalOptimisation(
        energy=float(mc.e_tot), mo_coeff=np.asarray(mc.mo_coeff), rotation=u, converged=bool(mc.converged),
        seconds=time.perf_counter() - t0, macro_iterations=-1 if macro is None else int(macro),
    )
