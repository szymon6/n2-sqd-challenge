"""Challenge baselines, each saved as a scorable result artifact.

- ``hf``: the 1 × 1 Hartree–Fock subspace.
- ``lab-reference``: the lab's 300 lowest-excitation-rank strings (Exercise 4).
- ``lab-sqd``: the lab's SQD loop on simulated LUCJ samples, as given (seed 142, 5 × 5 × 500, max_dim 300).
- ``lab-sqd-augmented``: the same loop with the first 100 reference strings pinned.

Run::

    python -m n2sqd.baselines --out results/baselines --bond-dim 128 --shots 2000 --seeds 142 1 2

Samples are cached in ``<out>/samples_bd<bond_dim>_s<seed>.npz``; one artifact per baseline and seed is
written next to them, plus ``summary.json`` with the energies and run times. Score any artifact with
``python -m eval.score <artifact>``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from n2sqd.hamiltonian import N2Hamiltonian, load_or_build
from n2sqd.lab_sqd import LAB_PINNED, lab_reference_subspace, run_lab_sqd
from n2sqd.result import Result
from n2sqd.solver import grader_energy
from n2sqd.strings import hf_string, lab_reference_strings


def hf_result(ham: N2Hamiltonian) -> Result:
    hf = np.array([hf_string(ham.nelec[0])], dtype=np.int64)
    energy, seconds = grader_energy(hf, hf, ham)
    return Result(hf, hf, ham.hcore, ham.eri, ham.core_energy, energy,
                  {"id": "hf", "track": "baseline", "description": "Hartree-Fock determinant only",
                   "command": "python -m n2sqd.baselines --only hf", "seconds": seconds})


def lab_reference_result(ham: N2Hamiltonian, n: int = 300) -> Result:
    strings = lab_reference_strings(n, ham.norb, ham.nelec[0])
    energy, seconds = grader_energy(strings, strings, ham)
    return Result(strings, strings, ham.hcore, ham.eri, ham.core_energy, energy,
                  {"id": f"lab-reference-{n}", "track": "baseline",
                   "description": f"lab Exercise 4: first {n} lowest-excitation-rank strings per spin",
                   "command": f"python -m n2sqd.baselines --only lab-reference --ref-size {n}", "seconds": seconds})


def get_samples(ham: N2Hamiltonian, out: Path, bond_dim: int, shots: int, seed: int):
    """Cached MPS samples of the lab circuit."""
    from n2sqd.sampling import Samples, lucj_circuit, sample_mps

    path = out / f"samples_bd{bond_dim}_s{seed}.npz"
    if path.exists():
        return Samples.load(path)
    circuit = lucj_circuit(ham)
    samples = sample_mps(circuit, shots=shots, bond_dim=bond_dim, seed=seed, norb=ham.norb, nelec=ham.nelec)
    samples.save(path)
    return samples


def lab_sqd_result(ham: N2Hamiltonian, samples, seed: int, augmented: bool, bond_dim: int, shots: int) -> tuple[Result, dict]:
    include = lab_reference_subspace()[:, :LAB_PINNED] if augmented else None
    t0 = time.perf_counter()
    run = run_lab_sqd(samples.bitstrings, samples.probs, ham, seed=seed, include=include)
    seconds = time.perf_counter() - t0
    name = "lab-sqd-augmented" if augmented else "lab-sqd"
    a, b = run.best_subspace
    meta = {
        "id": f"{name}-bd{bond_dim}-s{seed}", "track": "baseline",
        "description": f"lab SQD loop as given ({'100 reference strings pinned, ' if augmented else ''}"
                       f"MPS samples bond_dim={bond_dim}, {shots} shots), seed {seed}",
        "command": f"python -m n2sqd.baselines --only {name} --bond-dim {bond_dim} --shots {shots} --seeds {seed}",
        "seed": seed, "bond_dim": bond_dim, "shots": shots, "valid_fraction": samples.valid_fraction,
        "history": run.history, "seconds": seconds, "sampling_seconds": samples.seconds,
    }
    return Result(a, b, ham.hcore, ham.eri, ham.core_energy, run.best_energy, meta), meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the baselines and save scorable artifacts.")
    parser.add_argument("--out", type=Path, default=Path("results/baselines"))
    parser.add_argument("--only", nargs="*", default=["hf", "lab-reference", "lab-sqd", "lab-sqd-augmented"])
    parser.add_argument("--ref-size", type=int, default=300)
    parser.add_argument("--bond-dim", type=int, default=128)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="*", default=[142])
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ham = load_or_build()
    summary: dict[str, dict] = {}
    if "hf" in args.only:
        r = hf_result(ham)
        r.save(args.out / "hf.npz")
        summary["hf"] = {"energy": r.claimed_energy, "seconds": r.meta["seconds"]}
    if "lab-reference" in args.only:
        r = lab_reference_result(ham, args.ref_size)
        r.save(args.out / f"lab-reference-{args.ref_size}.npz")
        summary[r.meta["id"]] = {"energy": r.claimed_energy, "seconds": r.meta["seconds"]}
    for seed in args.seeds:
        if not ({"lab-sqd", "lab-sqd-augmented"} & set(args.only)):
            break
        samples = get_samples(ham, args.out, args.bond_dim, args.shots, seed)
        print(f"samples seed={seed}: {len(samples.probs)} unique, valid fraction {samples.valid_fraction:.3f}, "
              f"{samples.seconds:.0f} s", flush=True)
        for augmented in (False, True):
            name = "lab-sqd-augmented" if augmented else "lab-sqd"
            if name not in args.only:
                continue
            r, meta = lab_sqd_result(ham, samples, seed, augmented, args.bond_dim, args.shots)
            r.save(args.out / f"{meta['id']}.npz")
            summary[meta["id"]] = {"energy": r.claimed_energy, "seconds": meta["seconds"], "history": meta["history"],
                                   "valid_fraction": samples.valid_fraction, "sampling_seconds": samples.seconds}
            print(f"{meta['id']}: E = {r.claimed_energy:.6f} ({meta['seconds']:.0f} s)", flush=True)
            _write_summary(args.out / "summary.json", summary)
    _write_summary(args.out / "summary.json", summary)
    print(json.dumps({k: v.get("energy") for k, v in summary.items()}, indent=2))
    return 0


def _write_summary(path: Path, summary: dict) -> None:
    """Merge into the summary file after every result, so a long run leaves partial results behind."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(summary)
    path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
