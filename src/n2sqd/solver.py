"""Diagonalisation of H in a product subspace ``strings_a × strings_b``.

Two entry points:

- :func:`grader_energy` is the official recomputation: ``qiskit_addon_sqd.fermion.solve_sci`` with
  default options, plus the core energy, as ``lab4b._exbonus_answer`` does. Use it for every reported
  number.
- :func:`solve_subspace` is the same PySCF solver (``fci.selected_ci.kernel_fixed_space``) with an
  optional warm start, for search loops (about 3 s instead of 18 s on a 300 × 300 subspace).
  It returns the same energy within solver tolerance.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from pyscf import fci

from n2sqd.hamiltonian import N2Hamiltonian


@dataclasses.dataclass
class SubspaceSolution:
    """Lowest eigenpair of H in the subspace, with per-orbital occupancies."""

    energy: float  # total energy including the core energy
    strings_a: np.ndarray
    strings_b: np.ndarray
    amplitudes: np.ndarray  # shape (len(strings_a), len(strings_b))
    occupancies_a: np.ndarray
    occupancies_b: np.ndarray
    seconds: float

    @property
    def dims(self) -> tuple[int, int]:
        return len(self.strings_a), len(self.strings_b)


def grader_energy(strings_a, strings_b, ham: N2Hamiltonian) -> tuple[float, float]:
    """Energy exactly as the grader computes it (before rounding), and the wall time in seconds."""
    from qiskit_addon_sqd.fermion import solve_sci

    a = np.asarray(strings_a, dtype=np.int64)
    b = np.asarray(strings_b, dtype=np.int64)
    t0 = time.perf_counter()
    result = solve_sci((a, b), ham.hcore, ham.eri, ham.norb, ham.nelec)
    return float(result.energy + ham.core_energy), time.perf_counter() - t0


SMALL_DIM = 100  # subspaces up to this many determinants start from their exact ground state


def _dense_ground(a: np.ndarray, b: np.ndarray, ham: N2Hamiltonian) -> np.ndarray:
    """Exact lowest eigenvector of H in ``a × b`` from the dense matrix (one ``contract_2e`` per determinant)."""
    from pyscf import ao2mo

    h2e = ao2mo.restore(1, fci.direct_spin1.absorb_h1e(ham.hcore, ham.eri, ham.norb, ham.nelec, 0.5), ham.norb)
    dim = len(a) * len(b)
    unit = np.eye(dim)
    dense = np.array([fci.selected_ci.contract_2e(h2e, fci.selected_ci._as_SCIvector(unit[i].reshape(len(a), len(b)), (a, b)),
                                                  ham.norb, ham.nelec).ravel() for i in range(dim)])
    _, vecs = np.linalg.eigh((dense + dense.T) / 2)
    return vecs[:, 0]


def solve_subspace(
    strings_a,
    strings_b,
    ham: N2Hamiltonian,
    ci0: np.ndarray | None = None,
    conv_tol: float = 1e-10,
    max_cycle: int = 200,
) -> SubspaceSolution:
    """Lowest eigenpair of H projected on ``strings_a × strings_b`` (no spin constraint, like the grader)."""
    a = np.asarray(strings_a, dtype=np.int64)
    b = np.asarray(strings_b, dtype=np.int64)
    myci = fci.selected_ci.SelectedCI()
    myci.conv_tol = conv_tol
    myci.max_cycle = max_cycle
    myci.verbose = 0
    if 1 < len(a) * len(b) <= SMALL_DIM:
        # tiny subspaces are diagonalised exactly: a Davidson start missed the lowest state of a nearly degenerate
        # 4 x 4 test subspace on the CI machine (stage 7 close-out); production subspaces (300 x 300) never get here
        ci0 = _dense_ground(a, b, ham)
    if ci0 is not None:
        # kernel_fixed_space uses ci0 only when it is an SCIvector; a plain array is silently replaced by the
        # default guess (found in stage 6: every "warm" solve of stages 1-5 was in fact cold). The default guess
        # (lowest diagonal determinant) is kept as a second start vector, so Davidson cannot miss the lowest state
        # when the warm vector happens to lie in another symmetry sector of the subspace.
        warm = np.asarray(ci0, dtype=float).reshape(len(a), len(b))
        hdiag = myci.make_hdiag(ham.hcore, ham.eri, (a, b), ham.norb, ham.nelec, compress=True)
        guess = [np.asarray(x, dtype=float).reshape(len(a), len(b)) for x in myci.get_init_guess((a, b), ham.norb, ham.nelec, 1, hdiag)]
        ci0 = fci.selected_ci._as_SCIvector(np.stack([warm / np.linalg.norm(warm)] + guess), (a, b))
    t0 = time.perf_counter()
    e, vec = fci.selected_ci.kernel_fixed_space(
        myci, ham.hcore, ham.eri, ham.norb, ham.nelec, ci_strs=(a, b), ci0=ci0
    )
    dm1a, dm1b = myci.make_rdm1s(vec, ham.norb, ham.nelec)
    amplitudes = np.array(vec)
    return SubspaceSolution(
        energy=float(e + ham.core_energy),
        strings_a=np.array(vec._strs[0]),
        strings_b=np.array(vec._strs[1]),
        amplitudes=amplitudes,
        occupancies_a=np.diagonal(dm1a).copy(),
        occupancies_b=np.diagonal(dm1b).copy(),
        seconds=time.perf_counter() - t0,
    )


def marginal_weights(amplitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-string weights Σ_b |C_ab|² (alpha) and Σ_a |C_ab|² (beta) of a normalised CI matrix."""
    c2 = np.abs(amplitudes) ** 2
    return c2.sum(axis=1), c2.sum(axis=0)
