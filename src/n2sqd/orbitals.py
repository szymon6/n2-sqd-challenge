"""Starting orbital families for the Track B engine (other basins than the RHF start).

Stages 1–5 started every chain from the canonical RHF orbitals (or natural orbitals of a state reached from them),
and stage 6 found the converged basis to be a symmetric stationary point. These families start elsewhere:

- ``pm-local``: Pipek–Mezey localisation of the five occupied and the 21 virtual RHF active orbitals, separately
  (atom-centred bonds, lone pairs and correlating orbitals; no point-group symmetry at all).
- ``uhf-no``: natural orbitals of the spin-summed density of the symmetry-broken UHF solution (stretched N₂ breaks
  spin symmetry: two high-spin N atoms), projected onto the RHF active space.
- ``split-local``: natural orbitals of a 1-RDM (DMRG by default), then Pipek–Mezey localisation within the
  strongly and within the weakly occupied group ("split-localised" orbitals, Bytautas–Ivanic–Ruedenberg 2003).

Every function returns a 28 × 28 AO→MO matrix whose first two columns are the RHF core orbitals (the grader's
core energy must not change) and whose 26 active columns are an orthogonal rotation of the RHF active orbitals,
so ``n2sqd.orbopt.integrals_for`` passes the grader's integral check (``test_orbitals.py``).
"""

from __future__ import annotations

import numpy as np

from n2sqd.hamiltonian import N_FROZEN
from n2sqd.rotation import natural_orbital_rotation


def _localise(mol, mo_subset: np.ndarray) -> np.ndarray:
    """Pipek–Mezey localisation of the columns of ``mo_subset`` (same span, orthonormal)."""
    from pyscf import lo

    if mo_subset.shape[1] < 2:
        return np.array(mo_subset, dtype=float)
    loc = lo.PM(mol, mo_subset)
    loc.verbose = 0
    return np.asarray(loc.kernel(), dtype=float)


def _active_rotation_from(mf, new_active: np.ndarray) -> np.ndarray:
    """Re-express orbitals that span the RHF active space as ``mo_active @ U`` (exact orthogonal ``U``)."""
    s = mf.mol.intor("int1e_ovlp")
    act = np.asarray(mf.mo_coeff)[:, N_FROZEN:]
    u = act.T @ s @ new_active
    # polish: nearest orthogonal matrix (removes 1e-10 noise from the localiser)
    w, _, vt = np.linalg.svd(u)
    return w @ vt


def _assemble(mf, u: np.ndarray) -> np.ndarray:
    mo = np.array(mf.mo_coeff, dtype=float)
    mo[:, N_FROZEN:] = mo[:, N_FROZEN:] @ u
    return mo


def pipek_mezey(mf, n_occ: int = 5) -> tuple[np.ndarray, dict]:
    """Localised RHF orbitals: occupied active and virtual orbitals localised separately."""
    mo = np.asarray(mf.mo_coeff, dtype=float)
    occ = _localise(mf.mol, mo[:, N_FROZEN:N_FROZEN + n_occ])
    vir = _localise(mf.mol, mo[:, N_FROZEN + n_occ:])
    u = _active_rotation_from(mf, np.hstack([occ, vir]))
    return _assemble(mf, u), {"orbitals": "Pipek-Mezey localised RHF orbitals (occupied and virtual separately)"}


def uhf_solution(mol, mf_rhf, max_stability_cycles: int = 5, polarisation: float = 0.3):
    """Spin-polarised UHF of stretched N₂ (two high-spin atoms, ⟨S²⟩ ≈ 2.8, E ≈ −108.769 Ha).

    The guess biases the alpha density to the first atom and the beta density to the second; internal
    instabilities are then followed until the solution is stable. (Starting from the RHF density instead lands
    on a spatially broken, spin-restricted solution at −108.47 Ha: the "symmetry-broken RHF".)
    """
    from pyscf import scf

    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.max_cycle = 300
    dm = mf_rhf.make_rdm1()
    aoslice = mol.aoslice_by_atom()
    mask = np.zeros(mol.nao)
    mask[aoslice[0, 2]:aoslice[0, 3]] = 1.0
    p0 = np.diag(mask)
    p1 = np.eye(mol.nao) - p0
    bias = polarisation * (p0 @ dm @ p0 - p1 @ dm @ p1)
    mf.kernel(dm0=np.array([dm / 2 + bias, dm / 2 - bias]))
    for _ in range(max_stability_cycles):
        mo_new, _, stable, _ = mf.stability(return_status=True)
        if stable:
            break
        mf.kernel(dm0=mf.make_rdm1(mo_new, mf.mo_occ))
    return mf


def uhf_natural(mf) -> tuple[np.ndarray, dict]:
    """Natural orbitals of the UHF spin-summed density within the RHF active space."""
    mf_u = uhf_solution(mf.mol, mf)
    s = mf.mol.intor("int1e_ovlp")
    dm = mf_u.make_rdm1()
    dm_tot = dm[0] + dm[1]
    act = np.asarray(mf.mo_coeff)[:, N_FROZEN:]
    dm_act = act.T @ s @ dm_tot @ s @ act  # density in the RHF active MO basis
    u, occ = natural_orbital_rotation(dm_act)
    ss, mult = mf_u.spin_square()
    return _assemble(mf, u), {"orbitals": "UHF natural orbitals (projected on the RHF active space)",
                              "e_uhf": float(mf_u.e_tot), "uhf_s2": float(ss), "occupations": occ.tolist()}


def split_localised(mf, dm1_active: np.ndarray, threshold: float = 1.0) -> tuple[np.ndarray, dict]:
    """Natural orbitals of ``dm1_active`` (26 × 26, RHF active basis), localised within the strongly (> threshold)
    and within the weakly occupied group."""
    u_no, occ = natural_orbital_rotation(dm1_active)
    act = np.asarray(mf.mo_coeff)[:, N_FROZEN:] @ u_no
    strong = occ > threshold
    parts = [_localise(mf.mol, act[:, strong]), _localise(mf.mol, act[:, ~strong])]
    u = _active_rotation_from(mf, np.hstack(parts))
    return _assemble(mf, u), {"orbitals": "split-localised natural orbitals", "n_strong": int(strong.sum()),
                              "occupations": occ.tolist()}
