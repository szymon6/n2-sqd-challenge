"""DMRG reference energy for N2 (10e, 26o) with block2, written to ``results/reference.json``.

FCI (4.3e9 determinants) is out of reach, so the best classical estimate is DMRG: a forward schedule up to
``--max-bond`` followed by a reverse schedule without noise; the energies of the reverse sweeps are
extrapolated linearly in the discarded weight to zero (standard practice, block2 documentation). The
extrapolated value is labelled an estimate; the uncertainty is reported as half the extrapolation distance.

Run: ``python -m n2sqd.reference --max-bond 3000 --threads 8`` (about 30-60 min on an M4 laptop).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from n2sqd.hamiltonian import N2Hamiltonian, load_or_build


def dmrg_schedule(max_bond: int) -> tuple[list[int], list[float], list[float]]:
    """Forward bond dimensions, noises and Davidson thresholds up to ``max_bond``."""
    bond_dims, noises, thrds = [], [], []
    for bd in (250, 500, 1000, 2000, 3000, 4000, 5000):
        if bd > max_bond:
            break
        bond_dims += [bd] * 4
        noises += [1e-4 if bd < 500 else 1e-5 if bd < 2000 else 1e-6] * 4
        thrds += [1e-8 if bd < 1000 else 1e-10] * 4
    bond_dims += [max_bond] * 4
    noises += [0.0] * 4
    thrds += [1e-10] * 4
    return bond_dims, noises, thrds


def reverse_schedule(max_bond: int) -> list[int]:
    """Decreasing bond dimensions for the extrapolation sweeps."""
    return [bd for bd in dict.fromkeys((max_bond, 3000, 2000, 1500, 1000, 750, 500)) if bd <= max_bond]


def extrapolate(dws: np.ndarray, energies: np.ndarray) -> tuple[float, float]:
    """Linear fit E(DW) = E0 + a·DW; returns (E0, uncertainty = |E_min - E0| / 2)."""
    a, e0 = np.polyfit(dws, energies, 1)
    return float(e0), float(abs(energies.min() - e0) / 2)


def last_discarded_weight(results) -> float:
    """Largest discarded weight of the last sweep from ``driver.get_dmrg_results()``.

    block2 returns ``(bond_dims, dws, energies)``; ``dws[-1]`` is the maximal discarded weight of the last sweep.
    """
    try:
        dws = results[1]
        return float(np.max(np.asarray(dws[-1], dtype=float)))
    except (IndexError, TypeError, ValueError):
        return float("nan")


_SWEEP_LINE = re.compile(r"Time elapsed =\s*([0-9.]+) \| E =\s*(-?[0-9.]+) .*?\| DW = ([0-9.eE+-]+)")


def reverse_sweeps_from_log(log_path: Path | str, bond_dims: list[int], sweeps_per_bond: int = 2) -> tuple[list[float], list[float]]:
    """Energies and discarded weights of the reverse schedule, parsed from the block2 log.

    Takes the sweep lines after the last ``resumed MPS`` marker (or after the forward schedule when there is no
    marker: the last ``len(bond_dims) * sweeps_per_bond`` lines) and keeps the last sweep of each bond dimension.
    """
    text = Path(log_path).read_text()
    if "resumed MPS" in text:
        text = text.split("resumed MPS")[-1]
    rows = [(float(e), float(dw)) for _, e, dw in _SWEEP_LINE.findall(text)]
    need = len(bond_dims) * sweeps_per_bond
    rows = rows[-need:] if len(rows) >= need else rows
    energies, dws = [], []
    for i in range(min(len(bond_dims), len(rows) // sweeps_per_bond)):
        e, dw = rows[(i + 1) * sweeps_per_bond - 1]
        energies.append(e)
        dws.append(dw)
    return energies, dws


def run_dmrg(ham: N2Hamiltonian, max_bond: int, threads: int, scratch: Path, resume: bool = False) -> dict:
    """Run block2 DMRG and return the reference table (energies, discarded weights, 1-RDM path).

    With ``resume=True`` the converged MPS ``KET`` is loaded from ``scratch`` and only the reverse schedule runs
    (used after the forward sweeps finished but the bookkeeping failed).
    """
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    t0 = time.time()
    scratch.mkdir(parents=True, exist_ok=True)
    driver = DMRGDriver(scratch=str(scratch), symm_type=SymmetryTypes.SU2, n_threads=threads, stack_mem=4 << 30)
    driver.initialize_system(n_sites=ham.norb, n_elec=sum(ham.nelec), spin=ham.nelec[0] - ham.nelec[1])
    idx = driver.orbital_reordering(ham.hcore, ham.eri)
    h1 = ham.hcore[idx][:, idx]
    g2 = ham.eri[idx][:, idx][:, :, idx][:, :, :, idx]
    mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=ham.core_energy, iprint=1)
    bond_dims, noises, thrds = dmrg_schedule(max_bond)
    if resume:
        ket = driver.load_mps(tag="KET", nroots=1)
        e_fwd = float(driver.expectation(ket, mpo, ket))
        print(f"resumed MPS KET from {scratch}: E = {e_fwd:.10f}", flush=True)
    else:
        ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)
        e_fwd = driver.dmrg(mpo, ket, n_sweeps=len(bond_dims), bond_dims=bond_dims, noises=noises, thrds=thrds,
                            iprint=1, dav_max_iter=100, cutoff=1e-20)
    dm1 = driver.get_1pdm(ket)
    inv = np.argsort(idx)
    dm1 = np.asarray(dm1)[inv][:, inv]
    # reverse schedule for the extrapolation
    rev_bd, rev_e, rev_dw = [], [], []
    for bd in reverse_schedule(max_bond):
        e = driver.dmrg(mpo, ket, n_sweeps=2, bond_dims=[bd] * 2, noises=[0.0] * 2, thrds=[1e-10] * 2, iprint=1,
                        dav_max_iter=100, cutoff=1e-20)
        dw = last_discarded_weight(driver.get_dmrg_results()) if hasattr(driver, "get_dmrg_results") else float("nan")
        rev_bd.append(bd)
        rev_e.append(float(np.ravel(e)[0]))
        rev_dw.append(dw)
        print(f"reverse sweep bond {bd}: E = {rev_e[-1]:.10f}, discarded weight {dw:.3e}", flush=True)
    table = {
        "method": "DMRG (block2, SU(2), Fiedler ordering)",
        "max_bond_dimension": max_bond,
        "e_dmrg": float(np.ravel(e_fwd)[0]),
        "reverse_bond_dims": rev_bd,
        "reverse_energies": rev_e,
        "reverse_discarded_weights": rev_dw,
        "seconds": time.time() - t0,
        "orbital_order": [int(i) for i in idx],
    }
    dws, es = np.array(rev_dw), np.array(rev_e)
    if np.all(np.isfinite(dws)) and len(dws) >= 3:
        e0, unc = extrapolate(dws, es)
        table["e_dmrg_extrapolated"] = e0
        table["e_dmrg_extrapolation_uncertainty"] = unc
    return table, dm1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DMRG reference energy for N2 (10e, 26o).")
    parser.add_argument("--max-bond", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/reference.json"))
    parser.add_argument("--rdm-out", type=Path, default=Path("results/dmrg_rdm1.npy"))
    parser.add_argument("--scratch", type=Path, default=Path("data/dmrg_scratch"))
    parser.add_argument("--resume", action="store_true", help="load the converged MPS from --scratch and run only the reverse schedule")
    parser.add_argument("--from-log", type=Path, default=None,
                        help="do not run DMRG; rebuild the reverse-schedule energies and discarded weights from this log and re-extrapolate")
    args = parser.parse_args(argv)
    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    if args.from_log is not None:
        bds = reverse_schedule(args.max_bond)
        energies, dws = reverse_sweeps_from_log(args.from_log, bds)
        table = {"reverse_bond_dims": bds[: len(energies)], "reverse_energies": energies, "reverse_discarded_weights": dws,
                 "reverse_source": str(args.from_log)}
        if len(energies) >= 3:
            e0, unc = extrapolate(np.array(dws), np.array(energies))
            table["e_dmrg_extrapolated"] = e0
            table["e_dmrg_extrapolation_uncertainty"] = unc
        existing.update(table)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(existing, indent=2))
        print(json.dumps(table, indent=2))
        return 0
    ham = load_or_build()
    table, dm1 = run_dmrg(ham, args.max_bond, args.threads, args.scratch, resume=args.resume)
    table["e_hf"] = ham.e_hf
    existing.update(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(existing, indent=2))
    np.save(args.rdm_out, dm1)
    print(json.dumps({k: v for k, v in table.items() if k != "orbital_order"}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
