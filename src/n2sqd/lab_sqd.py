"""The lab's SQD workflow (QGSS 2026 Lab 4c, Exercises 2–4 and the loop of cells 61–75), as given.

This is the Lab 4b baseline. Hardware shots are replaced by the samples a
caller passes in (``n2sqd.sampling`` provides a matrix-product-state simulation of the lab's LUCJ
circuit). Everything else follows the notebook: Hamming-weight recovery with ``exp(occ) - 1`` flip
weights, ``subsample`` into batches, ``build_subspace`` with optional pinned strings and ``max_dim``,
``solve_sci_batch`` with default options, occupancies fed back, best energy kept.
"""

from __future__ import annotations

import copy
import dataclasses
from collections import defaultdict

import numpy as np

from n2sqd.hamiltonian import N2Hamiltonian
from n2sqd.strings import lab_reference_strings

# Lab defaults (cell 61).
LAB_SEED = 142
LAB_MAX_ITERATIONS = 5
LAB_NUM_BATCHES = 5
LAB_SAMPLES_PER_BATCH = 500
LAB_MAX_DIM = (300, 300)
LAB_PINNED = 100


def reshape_bitstring(bitstrings: np.ndarray, num_orbitals: int) -> np.ndarray:
    """(n, 2·norb) → (n, 2, norb) with block 0 = alpha (Exercise 2a).

    Qiskit strings are little-endian: the right half holds qubits 0..norb-1 (alpha), so the two halves
    are swapped after the reshape.
    """
    reshaped = np.asarray(bitstrings, dtype=bool).reshape(-1, 2, num_orbitals)
    return reshaped[:, ::-1, :]


def hamming_weight(bitstrings: np.ndarray) -> np.ndarray:
    """Electrons per spin sector (Exercise 2b)."""
    return np.asarray(bitstrings, dtype=bool).sum(axis=-1)


def weight_flip_0_to_1(occupancy: np.ndarray) -> np.ndarray:
    """``exp(occ) - 1`` (Exercise 3a)."""
    return np.exp(np.asarray(occupancy, dtype=float)) - 1.0


def weight_flip_1_to_0(occupancy: np.ndarray) -> np.ndarray:
    return weight_flip_0_to_1(1.0 - np.asarray(occupancy, dtype=float))


def _normalise(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0:
        return np.full(weights.shape, 1.0 / weights.size)
    return weights / total


def recover_configurations(
    bitstrings: np.ndarray,
    probs: np.ndarray,
    occupancy: np.ndarray,
    num_orbitals: int,
    num_elec: tuple[int, int],
    rand_seed: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip the fewest bits so every sample has the right electron count per spin (Exercise 3b)."""
    corrected: defaultdict[str, float] = defaultdict(float)
    for bitstring, freq, weight in zip(bitstrings, probs, hamming_weight(bitstrings)):
        bs = np.array(bitstring, dtype=bool)
        for i in range(2):
            if weight[i] < num_elec[i]:
                candidates = np.where(~bs[i])[0]
                p = _normalise(weight_flip_0_to_1(occupancy[i][candidates]))
                flip = rand_seed.choice(candidates, size=num_elec[i] - weight[i], replace=False, p=p)
                bs[i, flip] = True
            elif weight[i] > num_elec[i]:
                candidates = np.where(bs[i])[0]
                p = _normalise(weight_flip_1_to_0(occupancy[i][candidates]))
                flip = rand_seed.choice(candidates, size=weight[i] - num_elec[i], replace=False, p=p)
                bs[i, flip] = False
        corrected["".join("1" if b else "0" for b in bs.flatten())] += freq
    bs_out = np.array([[c == "1" for c in key] for key in corrected]).reshape(-1, 2, num_orbitals)
    freqs = np.abs(np.array(list(corrected.values())))
    return bs_out, freqs / freqs.sum()


def build_subspace(ci_strs: np.ndarray, include: np.ndarray | None = None, max_dim=(None, None)) -> list[list[np.ndarray]]:
    """Per batch and spin: pinned strings first, then sampled strings by descending count, cut at ``max_dim``, sorted."""
    include = np.empty((2, 0), dtype=np.int64) if include is None else np.asarray(include, dtype=np.int64)
    subspaces = []
    for samples in ci_strs:
        pair = []
        for i in range(2):
            strings, counts = np.unique(samples[i], return_counts=True)
            strings = strings[np.argsort(counts, kind="stable")[::-1]]
            merged = np.concatenate((include[i], strings))
            _, first = np.unique(merged, return_index=True)
            first.sort()
            merged = merged[first][: max_dim[i]]
            merged.sort()
            pair.append(merged.astype(np.int64))
        subspaces.append(pair)
    return subspaces


def initial_occupancy(num_orbitals: int, nelec: tuple[int, int]) -> np.ndarray:
    """HF-like prior: the highest-index orbitals occupied (lab cell 47)."""
    occ = np.zeros((2, num_orbitals))
    occ[0, -nelec[0]:] = 1.0
    occ[1, -nelec[1]:] = 1.0
    return occ


@dataclasses.dataclass
class LabSQDRun:
    """Everything the lab plots: best result, per-iteration energies and the subspace it submits."""

    best_energy: float
    best_subspace: tuple[np.ndarray, np.ndarray]
    history: list[list[float]]  # history[iteration][batch] total energies
    best_occupancies: tuple[np.ndarray, np.ndarray]


def run_lab_sqd(
    raw_bitstrings: np.ndarray,
    raw_probs: np.ndarray,
    ham: N2Hamiltonian,
    seed: int = LAB_SEED,
    max_iterations: int = LAB_MAX_ITERATIONS,
    num_batches: int = LAB_NUM_BATCHES,
    samples_per_batch: int = LAB_SAMPLES_PER_BATCH,
    max_dim: tuple[int, int] = LAB_MAX_DIM,
    include: np.ndarray | None = None,
    sci_solver=None,
    callback=None,
) -> LabSQDRun:
    """The lab's self-consistent loop (cells 63 and 74). ``include`` pins strings (``(2, n)`` array)."""
    from qiskit_addon_sqd.counts import bitstring_matrix_to_integers
    from qiskit_addon_sqd.fermion import solve_sci_batch
    from qiskit_addon_sqd.subsampling import subsample

    solver = sci_solver or solve_sci_batch
    rng = np.random.default_rng(seed)
    norb, nelec = ham.norb, ham.nelec
    occupancies = initial_occupancy(norb, nelec)
    best = None
    history: list[list[float]] = []
    for _ in range(max_iterations):
        reshaped = reshape_bitstring(raw_bitstrings, norb)
        bitstrings, probs = recover_configurations(reshaped, raw_probs, occupancies, norb, nelec, rand_seed=rng)
        batches = np.array(
            subsample(bitstrings.reshape(-1, 2 * norb), probs, samples_per_batch=samples_per_batch, num_batches=num_batches, rand_seed=rng)
        )
        batches = batches.reshape(*batches.shape[:-1], 2, -1)
        ci_strs = np.array([bitstring_matrix_to_integers(samples[:, i]) for samples in batches for i in range(2)])
        ci_strs = ci_strs.reshape(-1, 2, ci_strs.shape[-1])
        subspaces = build_subspace(ci_strs, include, max_dim=max_dim)
        results = solver(subspaces, ham.hcore, ham.eri, norb, nelec)
        history.append([float(r.energy + ham.core_energy) for r in results])
        if callback is not None:
            callback(results)
        best_in_batch = min(results, key=lambda r: r.energy)
        if best is None or best_in_batch.energy < best.energy:
            best = copy.deepcopy(best_in_batch)
        occupancies = np.clip(np.array(best_in_batch.orbital_occupancies), 0.0, 1.0)
    assert best is not None
    return LabSQDRun(
        best_energy=float(best.energy + ham.core_energy),
        best_subspace=(np.array(best.sci_state.ci_strs_a), np.array(best.sci_state.ci_strs_b)),
        history=history,
        best_occupancies=(np.array(best.orbital_occupancies[0]), np.array(best.orbital_occupancies[1])),
    )


def lab_reference_subspace(n: int = 300) -> np.ndarray:
    """Exercise 4: ``(2, n)`` array of the lab's reference strings for both spins."""
    strings = lab_reference_strings(n)
    return np.stack([strings, strings])
