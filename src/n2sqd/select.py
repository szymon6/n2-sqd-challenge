"""String selection for a product subspace with a fixed budget per spin.

Grow–prune loop (classical selected CI with a *string* budget, the shape the grader scores):

1. ``grow``: PySCF's heat-bath style ``enlarge_space`` proposes new alpha/beta strings connected to the
   current eigenvector by large ``|H_ij c_j|`` (``select_cutoff``), starting from strings with
   ``|c| > ci_coeff_cutoff``.
2. diagonalise in the enlarged product space;
3. ``prune``: keep the ``budget`` strings per spin with the largest marginal weight Σ_b |C_ab|²; for a
   singlet the alpha and beta weights are averaged so both spins keep the same set;
4. diagonalise again (warm start) and repeat while the energy drops.

Sample-based variants pass the sampled strings as the candidate pool instead of ``enlarge_space``
(Track A).
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from pyscf import fci

from n2sqd.hamiltonian import N2Hamiltonian
from n2sqd.solver import SubspaceSolution, marginal_weights, solve_subspace


def embed(amplitudes: np.ndarray, old: tuple[np.ndarray, np.ndarray], new: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Place the CI matrix of the ``old`` product space into the ``new`` one (zeros elsewhere), normalised."""
    out = np.zeros((len(new[0]), len(new[1])))
    ia = np.searchsorted(new[0], old[0])
    ib = np.searchsorted(new[1], old[1])
    keep_a = (ia < len(new[0])) & (new[0][np.minimum(ia, len(new[0]) - 1)] == old[0])
    keep_b = (ib < len(new[1])) & (new[1][np.minimum(ib, len(new[1]) - 1)] == old[1])
    sub = amplitudes[np.ix_(keep_a, keep_b)]
    out[np.ix_(ia[keep_a], ib[keep_b])] = sub
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def grow(solution: SubspaceSolution, ham: N2Hamiltonian, select_cutoff: float = 1e-3, ci_coeff_cutoff: float = 1e-3,
         max_new: int | None = None, rank: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Strings of the enlarged space proposed by PySCF's heat-bath selection (current strings included).

    With ``max_new`` set and ``rank=True`` the new strings are ranked by their second-order energy gain
    (:func:`importance`) and only the best ``max_new`` per spin are kept; ``rank=False`` keeps the first
    ``max_new`` in string order (the stage-1 E1 behaviour, kept for reproducibility).
    """
    myci = fci.selected_ci.SelectedCI()
    myci.select_cutoff = select_cutoff
    myci.ci_coeff_cutoff = ci_coeff_cutoff
    civec = fci.selected_ci._as_SCIvector(np.asarray(solution.amplitudes, dtype=float),
                                          (solution.strings_a, solution.strings_b))
    enlarged = fci.selected_ci.enlarge_space(myci, civec, ham.eri, ham.norb, ham.nelec)
    strs_a, strs_b = np.asarray(enlarged._strs[0], dtype=np.int64), np.asarray(enlarged._strs[1], dtype=np.int64)
    if max_new is None:
        return strs_a, strs_b
    if not rank:
        return _cap_new(strs_a, solution.strings_a, max_new), _cap_new(strs_b, solution.strings_b, max_new)
    new_a = np.setdiff1d(strs_a, solution.strings_a)
    new_b = np.setdiff1d(strs_b, solution.strings_b)
    symmetric = np.array_equal(solution.strings_a, solution.strings_b) and np.array_equal(new_a, new_b)
    gain_a = importance(solution, new_a, ham, spin=0) if new_a.size > max_new else None
    keep_a = new_a if gain_a is None else np.sort(new_a[np.argsort(-gain_a, kind="stable")[:max_new]])
    if symmetric:
        keep_b = keep_a.copy()
    else:
        gain_b = importance(solution, new_b, ham, spin=1) if new_b.size > max_new else None
        keep_b = new_b if gain_b is None else np.sort(new_b[np.argsort(-gain_b, kind="stable")[:max_new]])
    return np.union1d(solution.strings_a, keep_a), np.union1d(solution.strings_b, keep_b)


def importance(solution: SubspaceSolution, candidates: np.ndarray, ham: N2Hamiltonian, spin: int = 0) -> np.ndarray:
    """Epstein–Nesbet second-order gain of adding each candidate string (with the whole other-spin set).

    gain(a') = Σ_b |⟨a' b|H|Ψ⟩|² / (E_Ψ − ⟨a' b|H|a' b⟩), computed with one PySCF contraction in the space
    (S_a ∪ candidates) × S_b (or the mirror for ``spin=1``). Larger is more important.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0:
        return np.zeros(0)
    cur = (solution.strings_a, solution.strings_b)
    ext = list(cur)
    ext[spin] = np.union1d(cur[spin], candidates)
    ext = (ext[0], ext[1])
    amps = embed(solution.amplitudes, cur, ext)
    myci = fci.selected_ci.SelectedCI()
    h2e = myci.absorb_h1e(ham.hcore, ham.eri, ham.norb, ham.nelec, 0.5)
    vec = fci.selected_ci._as_SCIvector(amps, ext)
    hc = np.asarray(myci.contract_2e(h2e, vec, ham.norb, ham.nelec))
    hdiag = np.asarray(myci.make_hdiag(ham.hcore, ham.eri, ext, ham.norb, ham.nelec)).reshape(len(ext[0]), len(ext[1]))
    e_el = solution.energy - ham.core_energy
    idx = np.searchsorted(ext[spin], candidates)
    if spin == 0:
        r, d = hc[idx, :], hdiag[idx, :]
    else:
        r, d = hc[:, idx].T, hdiag[:, idx].T
    denom = e_el - d
    denom = np.where(np.abs(denom) < 1e-8, -1e-8, denom)
    return -(r * r / denom).sum(axis=1)  # denom < 0 for strings above the ground state → positive gain


def _cap_new(enlarged: np.ndarray, current: np.ndarray, max_new: int) -> np.ndarray:
    new = np.setdiff1d(enlarged, current)
    if new.size <= max_new:
        return enlarged
    return np.union1d(current, new[:max_new])


def removal_losses(solution: SubspaceSolution, ham: N2Hamiltonian) -> tuple[np.ndarray, np.ndarray]:
    """First-order energy increase from removing each alpha (beta) string's row (column) of the CI matrix.

    ΔE(a) ≈ Σ_b C_ab² (⟨ab|H|ab⟩ − E) / (1 − Σ_b C_ab²): drop the row, renormalise, keep the diagonal of the
    removed block. Larger means more important.
    """
    myci = fci.selected_ci.SelectedCI()
    strs = (solution.strings_a, solution.strings_b)
    hdiag = np.asarray(myci.make_hdiag(ham.hcore, ham.eri, strs, ham.norb, ham.nelec)).reshape(len(strs[0]), len(strs[1]))
    c2 = np.abs(solution.amplitudes) ** 2
    excess = hdiag - (solution.energy - ham.core_energy)
    wa, wb = c2.sum(axis=1), c2.sum(axis=0)
    loss_a = (c2 * excess).sum(axis=1) / np.maximum(1.0 - wa, 1e-12)
    loss_b = (c2 * excess).sum(axis=0) / np.maximum(1.0 - wb, 1e-12)
    return loss_a, loss_b


def prune(solution: SubspaceSolution, budget: int, symmetric: bool = True, criterion: str = "weight",
          ham: N2Hamiltonian | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The ``budget`` strings per spin with the largest marginal weight (or removal loss, ``criterion="loss"``)."""
    if criterion == "loss":
        if ham is None:
            raise ValueError("criterion='loss' needs the Hamiltonian")
        wa, wb = removal_losses(solution, ham)
    elif criterion == "weight":
        wa, wb = marginal_weights(solution.amplitudes)
    else:
        raise ValueError(f"unknown prune criterion {criterion!r}")
    if symmetric and np.array_equal(solution.strings_a, solution.strings_b):
        w = 0.5 * (wa + wb)
        keep = np.sort(solution.strings_a[np.argsort(-w, kind="stable")[:budget]])
        return keep, keep.copy()
    keep_a = np.sort(solution.strings_a[np.argsort(-wa, kind="stable")[:budget]])
    keep_b = np.sort(solution.strings_b[np.argsort(-wb, kind="stable")[:budget]])
    return keep_a, keep_b


def prune_schedule(start: int, budget: int, steps: int) -> list[int]:
    """Geometric sizes from ``start`` down to ``budget`` in ``steps`` prune–resolve rounds (last entry = budget)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if start <= budget or steps == 1:
        return [budget]
    ratio = (budget / start) ** (1.0 / steps)
    sizes = [int(round(start * ratio**k)) for k in range(1, steps)]
    sizes = [s for s in sizes if budget < s < start]
    return sorted(set(sizes), reverse=True) + [budget]


def select_from_pool(solution: SubspaceSolution, pool: tuple[np.ndarray, np.ndarray], budget: int,
                     ham: N2Hamiltonian, symmetric: bool = True, criterion: str = "weight",
                     prune_steps: int = 1) -> SubspaceSolution:
    """Diagonalise in ``pool`` (warm-started from ``solution``) and prune to ``budget``; returns the pruned solution.

    ``prune_steps > 1`` prunes gradually (stage 6): the pool shrinks geometrically to ``budget`` and the
    eigenvector is recomputed after every cut, so the weights redistribute before the next cut. One step is the
    one-shot prune of stages 1–5.
    """
    pool = (np.union1d(pool[0], solution.strings_a), np.union1d(pool[1], solution.strings_b))
    ci0 = embed(solution.amplitudes, (solution.strings_a, solution.strings_b), pool)
    big = solve_subspace(pool[0], pool[1], ham, ci0=ci0)
    for size in prune_schedule(max(len(pool[0]), len(pool[1])), budget, prune_steps):
        keep = prune(big, size, symmetric=symmetric, criterion=criterion, ham=ham)
        ci0 = embed(big.amplitudes, (big.strings_a, big.strings_b), keep)
        big = solve_subspace(keep[0], keep[1], ham, ci0=ci0)
    return big


@dataclasses.dataclass
class RefineStep:
    energy: float
    enlarged_dims: tuple[int, int]
    seconds: float


def refine(strings: tuple[np.ndarray, np.ndarray], ham: N2Hamiltonian, budget: int = 300, iterations: int = 10,
           select_cutoff: float = 1e-3, ci_coeff_cutoff: float = 1e-3, max_new: int | None = 700,
           tol: float = 1e-6, log=print, rank: bool = True, ci0: np.ndarray | None = None, symmetric: bool = True,
           criterion: str = "weight", prune_steps: int = 1) -> tuple[SubspaceSolution, list[RefineStep]]:
    """Grow–prune until the energy stops dropping by more than ``tol`` or ``iterations`` are used."""
    current = solve_subspace(strings[0], strings[1], ham, ci0=ci0)
    steps: list[RefineStep] = []
    for it in range(iterations):
        t0 = time.perf_counter()
        pool = grow(current, ham, select_cutoff, ci_coeff_cutoff, max_new, rank=rank)
        candidate = select_from_pool(current, pool, budget, ham, symmetric=symmetric, criterion=criterion,
                                     prune_steps=prune_steps)
        steps.append(RefineStep(candidate.energy, (len(pool[0]), len(pool[1])), time.perf_counter() - t0))
        if log:
            schedule = prune_schedule(max(len(pool[0]), len(pool[1])), budget, prune_steps)
            log(f"refine {it}: pool {len(pool[0])}x{len(pool[1])} -> E = {candidate.energy:.6f} "
                f"(prev {current.energy:.6f}, {steps[-1].seconds:.0f} s; prune {' -> '.join(str(n) for n in schedule)})")
        if candidate.energy < current.energy - tol:
            current = candidate
        else:
            if candidate.energy < current.energy:
                current = candidate
            break
    return current, steps
