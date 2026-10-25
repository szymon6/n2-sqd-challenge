"""Track A runner: sample-based SQD (stage 2).

Strings come only from the logged samples of the lab's LUCJ circuit (Aer MPS, :mod:`n2sqd.sampling`),
processed by ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`` — the official SQD loop
(configuration recovery, batching, carry-over, ``max_dim`` 300, spin symmetrisation, optional pinned lab
reference strings). Optional orbital-optimisation cycles (arXiv:2405.05068 Sec. II A 4) rotate the Hamiltonian
with the configurations fixed, and may re-run the SQD loop in the rotated basis with the current strings
carried over. No classical string proposals (that is Track B, :mod:`n2sqd.trackb`).

    python -m n2sqd.tracka --out output/2-track-a-samples/S1-shots20000 \
        --samples results/samples/shots20000/samples_bd128_s142.npz --max-iterations 10 --oo-cycles 0

Folder contents as in Track B: ``config.json``, ``command.txt``, ``log.txt``, ``metrics.json``, ``result.npz``,
``orbitals.npy``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from n2sqd.hamiltonian import N2Hamiltonian
from n2sqd.lab_sqd import lab_reference_subspace
from n2sqd.orbopt import integrals_for, optimize_orbitals, scf_reference
from n2sqd.result import Result
from n2sqd.sampling import Samples
from n2sqd.select import prune
from n2sqd.solver import solve_subspace
from n2sqd.trackb import RunLog, run_environment, spin_square


@dataclasses.dataclass
class TrackAConfig:
    out: Path
    samples: Path
    max_dim: int = 300
    num_batches: int = 5
    samples_per_batch: int = 1000
    max_iterations: int = 10
    carryover_threshold: float = 1e-4
    symmetrize_spin: bool = True
    pinned_reference: int = 0  # number of lab reference strings pinned (Exercise 4), 0 = none
    oo_cycles: int = 0
    oo_macro: int = 10
    reselect: str = "sqd"  # after each OO cycle: "sqd" = rerun the SQD loop in the rotated basis, "none"
    carry_fraction: float = 0.5  # share of the current strings (by marginal weight) carried into the re-selection loop
    seed: int = 142
    orbitals: str = "rhf"  # rhf | file:<orbitals.npy>
    start: str = "sqd"  # sqd = run the SQD loop first | file:<result.npz> = continue from those strings


def shots_matrix(samples: Samples) -> np.ndarray:
    """Expand the unique bitstrings to one row per shot (what the SQD loop subsamples from)."""
    counts = np.rint(samples.probs * samples.shots).astype(int)
    return np.repeat(samples.bitstrings, counts, axis=0)


def sqd_loop(ham: N2Hamiltonian, bit_array: np.ndarray, config: TrackAConfig, include=None, log=None, seed=None):
    """One run of the official SQD loop; returns (SCIResult, history of total energies per iteration/batch)."""
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    history: list[list[float]] = []

    def callback(results):
        history.append([float(r.energy + ham.core_energy) for r in results])
        if log is not None:
            log(f"  sqd iteration {len(history)}: best batch E = {min(history[-1]):.6f}, dims {[np.prod(r.sci_state.amplitudes.shape) for r in results]}")

    result = diagonalize_fermionic_hamiltonian(
        ham.hcore, ham.eri, bit_array, samples_per_batch=config.samples_per_batch, norb=ham.norb, nelec=ham.nelec,
        num_batches=config.num_batches, max_iterations=config.max_iterations, symmetrize_spin=config.symmetrize_spin,
        max_dim=config.max_dim, include_configurations=include, carryover_threshold=config.carryover_threshold,
        callback=callback, seed=config.seed if seed is None else seed,
    )
    return result, history


def pool_statistics(strings: tuple[np.ndarray, np.ndarray], samples: Samples, norb: int, nelec: tuple[int, int]) -> dict:
    """How many final strings appear verbatim in the valid raw samples (before recovery)."""
    bits = samples.bitstrings
    beta, alpha = bits[:, :norb], bits[:, norb:]
    weights = np.array([1 << i for i in range(norb)][::-1], dtype=np.int64)
    valid = (alpha.sum(axis=1) == nelec[0]) & (beta.sum(axis=1) == nelec[1])
    raw_a = set((alpha[valid].astype(np.int64) * weights).sum(axis=1).tolist())
    raw_b = set((beta[valid].astype(np.int64) * weights).sum(axis=1).tolist())
    raw_any = raw_a | raw_b
    a, b = strings
    return {
        "raw_valid_unique_alpha": len(raw_a), "raw_valid_unique_beta": len(raw_b),
        "final_alpha_in_raw": int(sum(int(s) in raw_any for s in a)), "final_beta_in_raw": int(sum(int(s) in raw_any for s in b)),
    }


def run(config: TrackAConfig, argv: list[str] | None = None) -> dict:
    out = config.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(config).items()},
                                                 **run_environment()}, indent=2))
    command = "python -m n2sqd.tracka " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    (out / "command.txt").write_text(command + "\n")
    log = RunLog(out / "log.txt")

    samples = Samples.load(config.samples)
    bit_array = shots_matrix(samples)
    mf = scf_reference()
    mo = np.load(config.orbitals[5:]) if config.orbitals.startswith("file:") else np.array(mf.mo_coeff, dtype=float)
    ham = integrals_for(mo, mf)
    include = None
    if config.pinned_reference > 0:
        ref = lab_reference_subspace(config.pinned_reference)
        include = (ref[0].tolist(), ref[1].tolist())
    log(f"samples: {samples.shots} shots, {len(samples.probs)} unique, valid fraction {samples.valid_fraction:.3f} ({config.samples})")

    t0 = time.perf_counter()
    if config.start.startswith("file:"):
        previous = Result.load(config.start[5:])
        current = solve_subspace(np.asarray(previous.strings_a, dtype=np.int64), np.asarray(previous.strings_b, dtype=np.int64), ham)
        history = []
        log(f"start from {config.start[5:]}: E = {current.energy:.6f}, dims {current.dims}", energy=current.energy, phase="start")
    else:
        result, history = sqd_loop(ham, bit_array, config, include=include, log=log)
        strings = (np.asarray(result.sci_state.ci_strs_a, dtype=np.int64), np.asarray(result.sci_state.ci_strs_b, dtype=np.int64))
        current = solve_subspace(strings[0], strings[1], ham, ci0=np.asarray(result.sci_state.amplitudes))
        log(f"sqd loop: E = {current.energy:.6f}, dims {current.dims}, {len(history)} iterations, {time.perf_counter() - t0:.0f} s",
            energy=current.energy, phase="sqd-0")
    cycles: list[dict] = [{"phase": "sqd-0", "energy": current.energy, "history": history, "seconds": time.perf_counter() - t0}]

    for cycle in range(1, config.oo_cycles + 1):
        before = current.energy
        opt = optimize_orbitals((current.strings_a, current.strings_b), mo, mf, max_cycle_macro=config.oo_macro)
        mo = opt.mo_coeff
        ham = integrals_for(mo, mf)
        fixed = solve_subspace(current.strings_a, current.strings_b, ham, ci0=current.amplitudes)
        log(f"cycle {cycle} OO (strings fixed): E = {fixed.energy:.6f} (converged={opt.converged}, {opt.seconds:.0f} s)",
            energy=fixed.energy, phase=f"oo-{cycle}")
        current = fixed if fixed.energy < current.energy else current
        entry = {"phase": f"cycle-{cycle}", "energy_after_oo": fixed.energy, "oo_seconds": opt.seconds, "oo_converged": opt.converged}
        if config.reselect == "sqd":
            t1 = time.perf_counter()
            # Carry over only the heaviest strings, so sampled strings can still enter the max_dim budget.
            n_carry = max(1, int(round(config.carry_fraction * config.max_dim)))
            carry_a, carry_b = prune(current, n_carry)
            carry = (carry_a.tolist(), carry_b.tolist())
            if include is not None:
                carry = (sorted(set(carry[0]) | set(include[0])), sorted(set(carry[1]) | set(include[1])))
            log(f"  carrying over {len(carry[0])} alpha / {len(carry[1])} beta strings into the re-selection loop")
            result2, history2 = sqd_loop(ham, bit_array, config, include=carry, log=log, seed=config.seed + cycle)
            s2 = (np.asarray(result2.sci_state.ci_strs_a, dtype=np.int64), np.asarray(result2.sci_state.ci_strs_b, dtype=np.int64))
            cand = solve_subspace(s2[0], s2[1], ham, ci0=np.asarray(result2.sci_state.amplitudes))
            log(f"cycle {cycle} SQD re-selection in the rotated basis: E = {cand.energy:.6f}, dims {cand.dims}, {time.perf_counter() - t1:.0f} s",
                energy=cand.energy, phase=f"resel-{cycle}")
            entry.update({"energy_after_reselect": cand.energy, "reselect_history": history2, "reselect_seconds": time.perf_counter() - t1})
            if cand.energy < current.energy:
                current = cand
        cycles.append(entry)
        log(f"cycle {cycle} done: E = {current.energy:.6f} (gain {1e3 * (before - current.energy):.2f} mHa)", energy=current.energy, phase=f"cycle-{cycle}")
        if before - current.energy < 1e-4:
            log("cycle gain below tolerance, stopping")
            break

    ss = spin_square(current, ham)
    stats = pool_statistics((current.strings_a, current.strings_b), samples, ham.norb, ham.nelec)
    meta = {
        "id": out.name, "track": "A", "stage": out.resolve().parent.name,
        "description": f"SQD loop (qiskit-addon-sqd) on {samples.shots} MPS shots, {config.oo_cycles} OO cycles, reselect={config.reselect}, pinned={config.pinned_reference}",
        "command": command, "config": json.loads((out / "config.json").read_text()), "sampling": samples.meta,
        "pool_statistics": stats, "spin_square": ss, "seconds": time.perf_counter() - log.t0,
    }
    Result(current.strings_a, current.strings_b, ham.hcore, ham.eri, ham.core_energy, current.energy, meta).save(out / "result.npz")
    np.save(out / "orbitals.npy", mo)
    metrics = {"energy": current.energy, "dims": list(current.dims), "spin_square": ss, "seconds": meta["seconds"],
               "pool_statistics": stats, "cycles": cycles, "events": log.events}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"final E = {current.energy:.6f}, dims {current.dims}, S^2 = {ss:.4f}, {stats}, {meta['seconds']:.0f} s")
    return metrics


def parse_args(argv: list[str] | None = None) -> TrackAConfig:
    p = argparse.ArgumentParser(description="Track A: sample-based SQD with optional orbital optimisation.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True, help="samples_*.npz from n2sqd.sampling")
    p.add_argument("--max-dim", type=int, default=300)
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument("--samples-per-batch", type=int, default=1000)
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--carryover-threshold", type=float, default=1e-4)
    p.add_argument("--no-symmetrize-spin", action="store_true")
    p.add_argument("--pinned-reference", type=int, default=0)
    p.add_argument("--oo-cycles", type=int, default=0)
    p.add_argument("--oo-macro", type=int, default=10)
    p.add_argument("--reselect", default="sqd", choices=["sqd", "none"])
    p.add_argument("--seed", type=int, default=142)
    p.add_argument("--orbitals", default="rhf")
    p.add_argument("--start", default="sqd", help="sqd | file:<result.npz> (skip the first SQD loop, continue from those strings)")
    p.add_argument("--carry-fraction", type=float, default=0.5)
    a = p.parse_args(argv)
    return TrackAConfig(out=a.out, samples=a.samples, max_dim=a.max_dim, num_batches=a.num_batches, samples_per_batch=a.samples_per_batch,
                        max_iterations=a.max_iterations, carryover_threshold=a.carryover_threshold, symmetrize_spin=not a.no_symmetrize_spin,
                        pinned_reference=a.pinned_reference, oo_cycles=a.oo_cycles, oo_macro=a.oo_macro, reselect=a.reselect, seed=a.seed,
                        orbitals=a.orbitals, start=a.start, carry_fraction=a.carry_fraction)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    run(parse_args(argv), argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
