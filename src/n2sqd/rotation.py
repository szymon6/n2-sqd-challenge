"""Orbital rotations of the active-space integrals.

The grader accepts the N2 integrals in any orbital basis related to the RHF basis by a real orthogonal
rotation of the 26 active orbitals. ``hcore' = Uᵀ h U`` and
``eri'[i,j,k,l] = Σ eri[p,q,r,s] U[p,i] U[q,j] U[r,k] U[s,l]`` keep the spectra the grader compares.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def rotate_integrals(hcore: np.ndarray, eri: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``hcore`` and ``eri`` (chemists' notation, shape n⁴) with the orthogonal matrix ``u``.

    Column ``i`` of ``u`` holds the new orbital ``i`` in the old basis.
    """
    u = np.asarray(u, dtype=float)
    check_orthogonal(u)
    h = u.T @ hcore @ u
    g = np.einsum("pqrs,pi->iqrs", eri, u, optimize=True)
    g = np.einsum("iqrs,qj->ijrs", g, u, optimize=True)
    g = np.einsum("ijrs,rk->ijks", g, u, optimize=True)
    g = np.einsum("ijks,sl->ijkl", g, u, optimize=True)
    return h, g


def check_orthogonal(u: np.ndarray, tol: float = 1e-8) -> None:
    """Raise ``ValueError`` unless ``u`` is square, real and orthogonal."""
    u = np.asarray(u)
    if u.ndim != 2 or u.shape[0] != u.shape[1] or np.iscomplexobj(u):
        raise ValueError("u must be a real square matrix")
    err = np.abs(u.T @ u - np.eye(u.shape[0])).max()
    if err > tol:
        raise ValueError(f"u is not orthogonal (max deviation {err:.2e})")


def rotation_from_params(k_flat: np.ndarray, norb: int) -> np.ndarray:
    """``U = exp(K)`` for the antisymmetric ``K`` whose strict upper triangle is ``k_flat`` (row-major).

    Same parametrisation as ``qiskit_addon_sqd.fermion.optimize_orbitals``.
    """
    k_flat = np.asarray(k_flat, dtype=float)
    if k_flat.size != norb * (norb - 1) // 2:
        raise ValueError("k_flat must have norb*(norb-1)/2 entries")
    k = np.zeros((norb, norb))
    k[np.triu_indices(norb, k=1)] = k_flat
    k = k - k.T
    return expm(k)


def natural_orbital_rotation(dm1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal matrix whose columns are natural orbitals (occupations descending) of the spin-summed 1-RDM."""
    dm1 = np.asarray(dm1, dtype=float)
    occ, vecs = np.linalg.eigh((dm1 + dm1.T) / 2)
    order = np.argsort(occ)[::-1]
    return vecs[:, order], occ[order]
