"""Track B engine: grow–prune string selection alternated with orbital optimisation (stage 1).

One run = one experiment folder with ``config.json``, ``command.txt``, ``log.txt``, ``metrics.json``,
``result.npz`` (grader inputs) and ``orbitals.npy`` (the AO→MO coefficients that define the basis).

    python -m n2sqd.trackb --out output/1-track-b-engine/E1a --start lab-ref --orbitals rhf \
        --select-cutoff 1e-3 --ci-cutoff 1e-3 --max-new 700 --refine-iterations 6 --oo-cycles 0

Pieces: :mod:`n2sqd.select`, :mod:`n2sqd.orbopt`.
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
from pyscf import fci

from n2sqd import orbitals, select
from n2sqd.hamiltonian import N_FROZEN, N2Hamiltonian
from n2sqd.orbopt import integrals_for, optimize_orbitals, scf_reference
from n2sqd.result import Result
from n2sqd.rotation import natural_orbital_rotation
from n2sqd.solver import SubspaceSolution, solve_subspace
from n2sqd.strings import hf_string, lab_reference_strings


@dataclasses.dataclass
class TrackBConfig:
    out: Path
    start: str = "lab-ref"  # lab-ref | hf
    orbitals: str = "rhf"  # rhf | dmrg-no | pm-local | uhf-no | split-local | no-from:<folder> | file:<orbitals.npy>
    budget: int = 300
    select_cutoff: float = 1e-3
    ci_cutoff: float = 1e-3
    max_new: int = 700
    refine_iterations: int = 6
    refine_tol: float = 1e-4
    oo_cycles: int = 0
    oo_macro: int = 10
    reselect_iterations: int = 2
    cycle_tol: float = 1e-4
    dmrg_rdm: Path = Path("results/dmrg_rdm1.npy")
    rank_new: bool = True  # rank candidate strings by PT2 gain (False = string order, the E1 behaviour)
    prune: str = "weight"  # weight | loss (stage 3)
    asymmetric: bool = False  # allow S_a != S_b when pruning (stage 3)
    prune_steps: int = 1  # gradual prune–resolve rounds from the pool to the budget (stage 6); 1 = one-shot


def starting_orbitals(config: TrackBConfig, mf) -> tuple[np.ndarray, dict]:
    """AO→MO coefficients of the starting basis and a note on how they were made."""
    if config.orbitals.startswith("file:"):
        path = config.orbitals[5:]
        return np.load(path), {"orbitals": f"from {path}"}
    if config.orbitals.startswith("no-from:"):
        folder = Path(config.orbitals[8:])
        return natural_orbitals_of_run(folder, mf)
    if config.orbitals == "pm-local":
        return orbitals.pipek_mezey(mf)
    if config.orbitals == "uhf-no":
        return orbitals.uhf_natural(mf)
    if config.orbitals == "split-local":
        return orbitals.split_localised(mf, np.load(config.dmrg_rdm))
    if config.orbitals not in ("rhf", "dmrg-no"):
        raise ValueError(f"unknown orbitals option {config.orbitals!r}")
    mo = np.array(mf.mo_coeff, dtype=float)
    if config.orbitals == "rhf":
        return mo, {"orbitals": "canonical RHF"}
    dm1 = np.load(config.dmrg_rdm)
    u, occ = natural_orbital_rotation(dm1)
    mo[:, N_FROZEN:] = mo[:, N_FROZEN:] @ u
    return mo, {"orbitals": "DMRG natural orbitals", "occupations": occ.tolist(), "rdm": str(config.dmrg_rdm)}


def natural_orbitals_of_run(folder: Path, mf) -> tuple[np.ndarray, dict]:
    """Natural orbitals of the CI state saved in an experiment folder (``result.npz`` + ``orbitals.npy``)."""
    result = Result.load(folder / "result.npz")
    mo = np.load(folder / "orbitals.npy")
    ham = N2Hamiltonian(result.hcore, result.eri, result.core_energy)
    myci = fci.selected_ci.SelectedCI()
    myci.verbose = 0
    _, vec = fci.selected_ci.kernel_fixed_space(myci, ham.hcore, ham.eri, ham.norb, ham.nelec,
                                                ci_strs=(result.strings_a, result.strings_b))
    dm1 = myci.make_rdm1(vec, ham.norb, ham.nelec)
    u, occ = natural_orbital_rotation(dm1)
    mo = np.array(mo, dtype=float)
    mo[:, N_FROZEN:] = mo[:, N_FROZEN:] @ u
    return mo, {"orbitals": f"natural orbitals of {folder}", "occupations": occ.tolist()}


def starting_strings(config: TrackBConfig, ham: N2Hamiltonian) -> tuple[np.ndarray, np.ndarray]:
    """``(strings_a, strings_b)`` of the starting subspace."""
    if config.start == "lab-ref":
        s = lab_reference_strings(config.budget, ham.norb, ham.nelec[0])
        return s, s.copy()
    if config.start == "hf":
        s = np.array([hf_string(ham.nelec[0])], dtype=np.int64)
        return s, s.copy()
    if config.start.startswith("file:"):
        r = Result.load(config.start[5:])
        return np.asarray(r.strings_a, dtype=np.int64), np.asarray(r.strings_b, dtype=np.int64)
    raise ValueError(f"unknown start option {config.start!r}")


def spin_square(solution: SubspaceSolution, ham: N2Hamiltonian) -> float:
    vec = fci.selected_ci._as_SCIvector(np.asarray(solution.amplitudes, dtype=float), (solution.strings_a, solution.strings_b))
    ss, _ = fci.selected_ci.SelectedCI().spin_square(vec, ham.norb, ham.nelec)
    return float(ss)


def run_environment() -> dict:
    """Thread count and code revision, recorded in every experiment's ``config.json`` (review finding, stage 2)."""
    import os
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git available
        commit = None
    return {"omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "git_commit": commit}


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.t0 = time.perf_counter()
        self.events: list[dict] = []
        path.write_text("")

    def __call__(self, message: str, **fields) -> None:
        t = time.perf_counter() - self.t0
        line = f"[{t:8.1f} s] {message}"
        with self.path.open("a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
        if fields:
            self.events.append({"t": t, "message": message, **fields})


def run(config: TrackBConfig, argv: list[str] | None = None) -> dict:
    """Run the engine and write the experiment folder. Returns the metrics dict."""
    out = config.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(config).items()},
                                                 **run_environment()}, indent=2))
    command = "python -m n2sqd.trackb " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    (out / "command.txt").write_text(command + "\n")
    log = RunLog(out / "log.txt")

    mf = scf_reference()
    mo, orbital_note = starting_orbitals(config, mf)
    ham = integrals_for(mo, mf)
    strings = starting_strings(config, ham)
    log(f"start: {config.start} ({strings[0].size} x {strings[1].size} strings), {orbital_note['orbitals']}")
    history: list[dict] = []

    current, steps = select.refine(strings, ham, config.budget, config.refine_iterations,
                                   config.select_cutoff, config.ci_cutoff, config.max_new, config.refine_tol, log=log,
                                   rank=config.rank_new, symmetric=not config.asymmetric, criterion=config.prune,
                                   prune_steps=config.prune_steps)
    history.append({"phase": "refine-0", "energy": current.energy, "steps": [dataclasses.asdict(s) for s in steps]})
    log(f"after initial grow-prune: E = {current.energy:.6f}", energy=current.energy, phase="refine-0")

    for cycle in range(1, config.oo_cycles + 1):
        before = current.energy
        opt = optimize_orbitals((current.strings_a, current.strings_b), mo, mf, max_cycle_macro=config.oo_macro)
        mo = opt.mo_coeff
        ham = integrals_for(mo, mf)
        rotated = solve_subspace(current.strings_a, current.strings_b, ham, ci0=current.amplitudes)
        log(f"cycle {cycle} OO: E = {rotated.energy:.6f} (casscf {opt.energy:.6f}, converged={opt.converged}, {opt.seconds:.0f} s)",
            energy=rotated.energy, phase=f"oo-{cycle}", casscf_energy=opt.energy, converged=opt.converged, seconds=opt.seconds)
        current = rotated if rotated.energy < current.energy else current
        reselected, steps = select.refine((current.strings_a, current.strings_b), ham, config.budget, config.reselect_iterations,
                                          config.select_cutoff, config.ci_cutoff, config.max_new, config.refine_tol, log=log,
                                          rank=config.rank_new, ci0=current.amplitudes, symmetric=not config.asymmetric,
                                          criterion=config.prune, prune_steps=config.prune_steps)
        if reselected.energy < current.energy:
            current = reselected
        history.append({"phase": f"cycle-{cycle}", "energy_after_oo": rotated.energy, "energy_after_reselect": current.energy,
                        "oo_seconds": opt.seconds, "oo_converged": opt.converged,
                        "reselect_steps": [dataclasses.asdict(s) for s in steps]})
        log(f"cycle {cycle} done: E = {current.energy:.6f} (gain {1e3 * (before - current.energy):.2f} mHa)", energy=current.energy, phase=f"cycle-{cycle}")
        save_outputs(out, config, command, current, ham, mo, orbital_note, history, log, checkpoint=cycle)
        if before - current.energy < config.cycle_tol:
            log("cycle gain below tolerance, stopping")
            break

    metrics = save_outputs(out, config, command, current, ham, mo, orbital_note, history, log, checkpoint=None)
    log(f"final E = {current.energy:.6f}, dims {current.dims}, S^2 = {metrics['spin_square']:.4f}, {metrics['seconds']:.0f} s")
    return metrics


def save_outputs(out: Path, config: TrackBConfig, command: str, current: SubspaceSolution, ham: N2Hamiltonian, mo: np.ndarray,
                 orbital_note: dict, history: list[dict], log: RunLog, checkpoint: int | None) -> dict:
    """Write ``result.npz``, ``orbitals.npy`` and ``metrics.json`` for the current state.

    Called after every OO cycle (``checkpoint`` = cycle number, stage 6: a chain stopped at the budget keeps its last
    state) and once more at the end (``checkpoint=None``); the final write overwrites the checkpoints.
    """
    ss = spin_square(current, ham)
    meta = {
        "id": out.name, "track": "B", "stage": out.resolve().parent.name, "description": f"grow-prune + OO ({config.oo_cycles} cycles), start {config.start}, {orbital_note['orbitals']}",
        "command": command, "config": json.loads((out / "config.json").read_text()), "spin_square": ss,
        "seconds": time.perf_counter() - log.t0, "orbital_note": orbital_note, "checkpoint_cycle": checkpoint,
    }
    Result(current.strings_a, current.strings_b, ham.hcore, ham.eri, ham.core_energy, current.energy, meta).save(out / "result.npz")
    np.save(out / "orbitals.npy", mo)
    metrics = {"energy": current.energy, "dims": list(current.dims), "spin_square": ss, "seconds": meta["seconds"],
               "checkpoint_cycle": checkpoint, "history": history, "events": log.events}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args(argv: list[str] | None = None) -> TrackBConfig:
    p = argparse.ArgumentParser(description="Track B engine: grow-prune + orbital optimisation.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default="lab-ref", help="lab-ref | hf | file:<result.npz>")
    p.add_argument("--orbitals", default="rhf", help="rhf | dmrg-no | pm-local | uhf-no | split-local | no-from:<folder> | file:<orbitals.npy>")
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--select-cutoff", type=float, default=1e-3)
    p.add_argument("--ci-cutoff", type=float, default=1e-3)
    p.add_argument("--max-new", type=int, default=700)
    p.add_argument("--refine-iterations", type=int, default=6)
    p.add_argument("--refine-tol", type=float, default=1e-4)
    p.add_argument("--oo-cycles", type=int, default=0)
    p.add_argument("--oo-macro", type=int, default=10)
    p.add_argument("--reselect-iterations", type=int, default=2)
    p.add_argument("--cycle-tol", type=float, default=1e-4)
    p.add_argument("--dmrg-rdm", type=Path, default=Path("results/dmrg_rdm1.npy"))
    p.add_argument("--no-rank", action="store_true", help="cap new strings in string order instead of by PT2 gain")
    p.add_argument("--prune", default="weight", choices=["weight", "loss"], help="prune criterion (stage 3)")
    p.add_argument("--asymmetric", action="store_true", help="allow different alpha and beta string sets (stage 3)")
    p.add_argument("--prune-steps", type=int, default=1, help="gradual prune-resolve rounds from pool to budget (stage 6)")
    a = p.parse_args(argv)
    return TrackBConfig(out=a.out, start=a.start, orbitals=a.orbitals, budget=a.budget, select_cutoff=a.select_cutoff,
                        ci_cutoff=a.ci_cutoff, max_new=a.max_new, refine_iterations=a.refine_iterations, refine_tol=a.refine_tol,
                        oo_cycles=a.oo_cycles, oo_macro=a.oo_macro, reselect_iterations=a.reselect_iterations,
                        cycle_tol=a.cycle_tol, dmrg_rdm=a.dmrg_rdm, rank_new=not a.no_rank, prune=a.prune, asymmetric=a.asymmetric,
                        prune_steps=a.prune_steps)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    run(parse_args(argv), argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
