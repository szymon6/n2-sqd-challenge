"""Active-space Hamiltonian of N2, built exactly like the grader and the lab notebook.

Reference system (the grader's): N2, cc-pVDZ, bond length 2.0 Å, RHF, 2 frozen core
orbitals, 26 active orbitals, 10 active electrons, nelec (5, 5). The grader
(``qfest_grader.challenges.warsaw_2026.lab4b._n2_hamiltonian``) uses the same PySCF calls.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

NORB = 26
NELEC = (5, 5)
N_FROZEN = 2
BOND_LENGTH = 2.0
BASIS = "cc-pvdz"
DEFAULT_CACHE = Path("data/n2_ints.npz")


@dataclasses.dataclass(frozen=True)
class N2Hamiltonian:
    """Active-space integrals plus the constant energy the grader calls ``nuclear_repulsion_energy``."""

    hcore: np.ndarray
    eri: np.ndarray
    core_energy: float
    norb: int = NORB
    nelec: tuple[int, int] = NELEC
    e_hf: float = float("nan")
    mo_coeff: np.ndarray | None = None
    mo_energy: np.ndarray | None = None

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            hcore=self.hcore,
            eri=self.eri,
            core_energy=self.core_energy,
            norb=self.norb,
            nelec=np.array(self.nelec),
            e_hf=self.e_hf,
            mo_coeff=np.zeros(0) if self.mo_coeff is None else self.mo_coeff,
            mo_energy=np.zeros(0) if self.mo_energy is None else self.mo_energy,
        )

    @classmethod
    def load(cls, path: Path | str) -> "N2Hamiltonian":
        d = np.load(path)
        mo_coeff = d["mo_coeff"] if d["mo_coeff"].size else None
        mo_energy = d["mo_energy"] if d["mo_energy"].size else None
        return cls(
            hcore=d["hcore"],
            eri=d["eri"],
            core_energy=float(d["core_energy"]),
            norb=int(d["norb"]),
            nelec=(int(d["nelec"][0]), int(d["nelec"][1])),
            e_hf=float(d["e_hf"]),
            mo_coeff=mo_coeff,
            mo_energy=mo_energy,
        )


def build_molecule():
    """The PySCF molecule of the lab (verbose off)."""
    import pyscf

    mol = pyscf.gto.Mole()
    mol.build(atom=[["N", (0, 0, 0)], ["N", (BOND_LENGTH, 0, 0)]], basis=BASIS, verbose=0)
    return mol


def build_n2_hamiltonian() -> N2Hamiltonian:
    """Build the integrals with the grader's exact sequence of PySCF calls (about 0.2 s)."""
    import pyscf
    import pyscf.mcscf

    mol = build_molecule()
    active_space = range(N_FROZEN, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    nelec = ((n_electrons + mol.spin) // 2, (n_electrons - mol.spin) // 2)
    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)
    hcore, core_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)
    return N2Hamiltonian(
        hcore=np.asarray(hcore),
        eri=np.asarray(eri),
        core_energy=float(core_energy),
        norb=norb,
        nelec=nelec,
        e_hf=float(scf.e_tot),
        mo_coeff=np.asarray(scf.mo_coeff),
        mo_energy=np.asarray(scf.mo_energy),
    )


def load_or_build(cache: Path | str = DEFAULT_CACHE) -> N2Hamiltonian:
    """Load the cached integrals, or build and cache them."""
    cache = Path(cache)
    if cache.exists():
        return N2Hamiltonian.load(cache)
    ham = build_n2_hamiltonian()
    ham.save(cache)
    return ham


if __name__ == "__main__":  # pragma: no cover
    h = load_or_build()
    print(f"norb={h.norb} nelec={h.nelec} core_energy={h.core_energy:.9f} E_HF={h.e_hf:.9f}")
