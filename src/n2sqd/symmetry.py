"""Point-group classes of a product subspace, and the kick that lets the engine leave a symmetric point.

Stage 6 observation: the converged Track B orbitals are still
eigenfunctions of a C2v subgroup of D∞h (the two-fold axis along the bond plus two mirror planes through it,
rotated about the bond by an arbitrary angle). Every orbital then carries one of four irreps, every CI string
the product of its orbitals' irreps, and the lowest state of a symmetric product subspace is totally symmetric,
so amplitudes between strings of different irreps vanish exactly: in the stage 4 best 59 % of the 300 × 300
determinants are dead. The orbital gradient along symmetry-breaking rotations is exactly zero for a symmetric
state, so the CASSCF step of the engine (:mod:`n2sqd.orbopt`) can never leave that manifold by itself.

Everything here is read off the integrals of an artifact, without a point-group frame: orbital classes are the
connected components of the one-electron coupling pattern (``|h_ij| + max_k |(ik|jk)| + max_k |(ij|kk)|``), and the
group multiplication (needed for the string irreps) is recovered by matching bit labels to the four-index
integrals' selection rule ``Γi ⊗ Γj = Γk ⊗ Γl``.

    python -m n2sqd.symmetry report results/best/track-b
    python -m n2sqd.symmetry kick --folder results/best/track-b --out kicked.npy --scale 0.05 --seed 1
    python -m n2sqd.symmetry probe --folder results/best/track-b --out output/6-symmetry-breaking/U0 --theta 0.05
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse.csgraph import connected_components

from n2sqd.hamiltonian import N_FROZEN, N2Hamiltonian
from n2sqd.result import Result
from n2sqd.rotation import rotate_integrals
from n2sqd.solver import solve_subspace
from n2sqd.strings import to_bits

DEAD_TOL = 1e-10  # |C_ab| below this counts as an exact zero


def coupling_matrix(ham: N2Hamiltonian) -> np.ndarray:
    """``|h_ij| + max_k |(ik|jk)| + max_k |(ij|kk)|``: zero exactly when orbitals i and j belong to different irreps."""
    h = np.abs(ham.hcore)
    exch = np.abs(np.einsum("ikjk->ijk", ham.eri)).max(axis=2)
    coul = np.abs(np.einsum("ijkk->ijk", ham.eri)).max(axis=2)
    return h + exch + coul


def orbital_classes(ham: N2Hamiltonian, tol: float = 1e-6) -> np.ndarray:
    """Class index per active orbital (connected components of :func:`coupling_matrix` above ``tol``).

    One class for every orbital irrep of the residual point group; a single class means the basis has no
    symmetry the integrals can see.
    """
    m = coupling_matrix(ham)
    n_classes, labels = connected_components(m > tol, directed=False)
    # relabel so that classes are numbered by their first orbital
    order = {old: new for new, old in enumerate(dict.fromkeys(labels.tolist()))}
    return np.array([order[int(x)] for x in labels], dtype=int)


def irrep_bits(ham: N2Hamiltonian, classes: np.ndarray, tol: float = 1e-6) -> np.ndarray | None:
    """Bit labels (Z2^k elements) per class so that ``(ij|kl) ≠ 0 ⟹ bits_i ^ bits_j == bits_k ^ bits_l``.

    Returns ``None`` when the class count is not a power of two or no assignment satisfies the rule (then the
    classes are not the irreps of an abelian group with self-inverse irreps, or the coupling pattern split one).
    """
    n = int(classes.max()) + 1
    if n == 1:
        return np.zeros(1, dtype=int)
    k = int(np.log2(n))
    if 2**k != n:
        return None
    nz = np.abs(ham.eri) > tol
    # class-level nonzero pattern: present[p, q, r, s] = any (ij|kl) above tol with those classes
    present = np.zeros((n, n, n, n), dtype=bool)
    idx = np.nonzero(nz)
    present[classes[idx[0]], classes[idx[1]], classes[idx[2]], classes[idx[3]]] = True
    p, q, r, s = np.meshgrid(*(np.arange(n),) * 4, indexing="ij")
    for perm in itertools.permutations(range(n)):
        bits = np.array(perm)
        rule = (bits[p] ^ bits[q]) == (bits[r] ^ bits[s])
        if np.all(rule[present]):
            return bits
    return None


def string_irreps(strings: np.ndarray, orbital_bits: np.ndarray, norb: int) -> np.ndarray:
    """XOR of the orbital bit labels over the occupied orbitals of every string."""
    bits = to_bits(np.asarray(strings, dtype=np.int64), norb)
    return np.bitwise_xor.reduce(np.where(bits, orbital_bits[None, :], 0), axis=1)


def report(result: Result, amplitudes: np.ndarray | None = None, tol: float = 1e-6) -> dict:
    """Symmetry structure of an artifact's basis and product subspace.

    Keys: ``n_classes``, ``class_sizes``, ``cross_class_pairs`` (orbital pairs the gradient ignores), and when
    the group structure is recoverable ``string_irreps_a/b`` (counts per irrep) and ``allowed_fraction`` (pairs
    ``(a, b)`` with equal irreps). With ``amplitudes``: ``dead_fraction`` (``|C_ab| < 1e-10``) and
    ``weight_in_allowed_block``.
    """
    ham = N2Hamiltonian(result.hcore, result.eri, result.core_energy)
    classes = orbital_classes(ham, tol)
    n = int(classes.max()) + 1
    sizes = np.bincount(classes, minlength=n).tolist()
    cross = int(sum(sizes[i] * sizes[j] for i in range(n) for j in range(i + 1, n)))
    out: dict = {"n_classes": n, "class_sizes": sizes, "orbital_classes": classes.tolist(), "cross_class_pairs": cross}
    bits = irrep_bits(ham, classes, tol)
    if bits is not None:
        orbital_bits = bits[classes]
        ga = string_irreps(result.strings_a, orbital_bits, ham.norb)
        gb = string_irreps(result.strings_b, orbital_bits, ham.norb)
        out["string_irreps_a"] = np.bincount(ga, minlength=n).tolist()
        out["string_irreps_b"] = np.bincount(gb, minlength=n).tolist()
        allowed = ga[:, None] == gb[None, :]
        out["allowed_fraction"] = float(allowed.mean())
        if amplitudes is not None:
            out["weight_in_allowed_block"] = float((np.asarray(amplitudes) ** 2)[allowed].sum())
    if amplitudes is not None:
        out["dead_fraction"] = float((np.abs(amplitudes) < DEAD_TOL).mean())
    return out


def kick_rotation(classes: np.ndarray, scale: float, seed: int, mode: str = "cross",
                  pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    """``U = exp(K)`` with random antisymmetric ``K`` (entries ~ N(0, scale)) on cross-class pairs (``mode="cross"``)
    or on every pair (``mode="all"``). ``mode="cross"`` moves only along directions whose gradient vanishes by
    symmetry at a symmetric point. ``pairs`` restricts the generator to the listed orbital pairs (targeted kick,
    e.g. the pairs the curvature probe flags); the sign of each entry is then random, its size ``scale``."""
    if mode not in ("cross", "all"):
        raise ValueError(f"unknown kick mode {mode!r}")
    n = classes.size
    rng = np.random.default_rng(seed)
    k = np.zeros((n, n))
    if pairs is not None:
        for i, j in pairs:
            i, j = min(i, j), max(i, j)
            k[i, j] = scale * rng.choice([-1.0, 1.0])
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if mode == "all" or classes[i] != classes[j]:
                    k[i, j] = rng.normal(0.0, scale)
    k -= k.T
    return expm(k)


def kicked_orbitals(mo_coeff: np.ndarray, u: np.ndarray) -> np.ndarray:
    """AO→MO coefficients with the active columns rotated by ``u`` (core columns untouched)."""
    mo = np.array(mo_coeff, dtype=float)
    mo[:, N_FROZEN:] = mo[:, N_FROZEN:] @ u
    return mo


def givens(norb: int, i: int, j: int, theta: float) -> np.ndarray:
    u = np.eye(norb)
    c, s = np.cos(theta), np.sin(theta)
    u[i, i] = u[j, j] = c
    u[i, j], u[j, i] = -s, s
    return u


def curvature_probe(result: Result, pairs: list[tuple[int, int]], theta: float = 0.05, log=None,
                    one_sided: bool = False) -> list[dict]:
    """Energy of the fixed string set after rotating orbital pairs by ``±theta`` (warm-started solves).

    Returns one dict per pair: ``i, j, class_i, class_j, e_plus, e_minus, curvature`` with
    ``curvature = (E(+θ) + E(−θ) − 2 E0) / θ²``; negative curvature means the symmetric point is a saddle along
    that rotation for this string set. ``one_sided`` computes ``E(+θ)`` only and uses ``2 (E(+θ) − E0) / θ²``,
    exact for cross-class pairs at a symmetric point (``E(θ) = E(−θ)`` by symmetry) and half the cost.
    """
    ham = N2Hamiltonian(result.hcore, result.eri, result.core_energy)
    classes = orbital_classes(ham)
    base = solve_subspace(result.strings_a, result.strings_b, ham)
    if log:
        log(f"base E = {base.energy:.9f} ({base.seconds:.1f} s), {len(pairs)} pairs, theta = {theta}")
    rows: list[dict] = []
    for i, j in pairs:
        t0 = time.perf_counter()
        energies = []
        for sign in ((1.0,) if one_sided else (1.0, -1.0)):
            h, g = rotate_integrals(ham.hcore, ham.eri, givens(ham.norb, i, j, sign * theta))
            rotated = N2Hamiltonian(h, g, ham.core_energy)
            energies.append(solve_subspace(result.strings_a, result.strings_b, rotated, ci0=base.amplitudes).energy)
        e_minus = energies[0] if one_sided else energies[1]
        curv = (energies[0] + e_minus - 2 * base.energy) / theta**2
        row = {"i": i, "j": j, "class_i": int(classes[i]), "class_j": int(classes[j]), "e_plus": energies[0],
               "e_minus": None if one_sided else e_minus, "curvature": curv, "seconds": time.perf_counter() - t0}
        rows.append(row)
        if log:
            log(f"pair ({i:2d},{j:2d}) classes ({classes[i]},{classes[j]}): E(+) {energies[0]:.9f} "
                + ("" if one_sided else f"E(-) {e_minus:.9f} ")
                + f"curvature {curv:+.5f} Ha/rad^2 ({row['seconds']:.0f} s)")
    return rows


def class_pairs(classes: np.ndarray, which: str = "cross") -> list[tuple[int, int]]:
    """Orbital pairs: ``cross`` (different classes), ``same`` or ``all``."""
    n = classes.size
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            different = classes[i] != classes[j]
            if which == "all" or (which == "cross" and different) or (which == "same" and not different):
                out.append((i, j))
    return out


def _load(folder: Path) -> tuple[Result, np.ndarray]:
    return Result.load(folder / "result.npz"), np.load(folder / "orbitals.npy")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description="Symmetry structure of a product subspace; kicks and curvature probes.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="classes, string irreps, dead fraction of an artifact folder")
    r.add_argument("folder", type=Path)
    r.add_argument("--no-solve", action="store_true", help="skip the diagonalisation (no dead fraction)")
    k = sub.add_parser("kick", help="write symmetry-broken starting orbitals for n2sqd.trackb --orbitals file:<out>")
    k.add_argument("--folder", type=Path, required=True, help="artifact folder (result.npz + orbitals.npy)")
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--scale", type=float, default=0.05)
    k.add_argument("--seed", type=int, default=1)
    k.add_argument("--mode", default="cross", choices=["cross", "all"])
    k.add_argument("--pairs", default=None, help="targeted kick on these orbital pairs only, e.g. '8-21,8-22'")
    q = sub.add_parser("probe", help="finite-difference curvature along orbital-pair rotations for the fixed strings")
    q.add_argument("--folder", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--theta", type=float, default=0.05)
    q.add_argument("--pairs", default="cross", choices=["cross", "same", "all"])
    q.add_argument("--slice", default=None, help="start:stop over the pair list (to split the work)")
    q.add_argument("--one-sided", action="store_true", help="E(+theta) only (exact for cross pairs at a symmetric point)")
    a = p.parse_args(argv)

    if a.cmd == "report":
        result, _ = _load(a.folder)
        amps = None
        if not a.no_solve:
            ham = N2Hamiltonian(result.hcore, result.eri, result.core_energy)
            amps = solve_subspace(result.strings_a, result.strings_b, ham).amplitudes
        print(json.dumps(report(result, amps), indent=2))
        return 0
    if a.cmd == "kick":
        result, mo = _load(a.folder)
        ham = N2Hamiltonian(result.hcore, result.eri, result.core_energy)
        classes = orbital_classes(ham)
        pairs = None
        if a.pairs:
            pairs = [tuple(int(x) for x in item.split("-")) for item in a.pairs.split(",")]
        u = kick_rotation(classes, a.scale, a.seed, a.mode, pairs=pairs)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(a.out, kicked_orbitals(mo, u))
        angle = float(np.degrees(np.arccos(np.clip(np.diag(u), -1, 1))).max())
        print(json.dumps({"classes": classes.tolist(), "mode": a.mode, "scale": a.scale, "seed": a.seed, "pairs": pairs,
                          "max_orbital_rotation_deg": angle, "out": str(a.out)}))
        return 0
    # probe
    result, _ = _load(a.folder)
    ham = N2Hamiltonian(result.hcore, result.eri, result.core_energy)
    classes = orbital_classes(ham)
    pairs = class_pairs(classes, a.pairs)
    if a.slice:
        start, stop = (int(x) if x else None for x in a.slice.split(":"))
        pairs = pairs[start:stop]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "command.txt").write_text("python -m n2sqd.symmetry " + " ".join(shlex.quote(x) for x in argv) + "\n")
    logf = a.out / "log.txt"
    logf.write_text("")

    def log(msg: str) -> None:
        with logf.open("a") as fh:
            fh.write(msg + "\n")
        print(msg, flush=True)

    rows = curvature_probe(result, pairs, a.theta, log=log, one_sided=a.one_sided)
    tag = f"-{a.slice.replace(':', '-')}" if a.slice else ""
    (a.out / f"curvature{tag}.json").write_text(json.dumps({"theta": a.theta, "pairs": a.pairs, "one_sided": a.one_sided,
                                                           "classes": classes.tolist(), "rows": rows}, indent=2))
    neg = [r_ for r_ in rows if r_["curvature"] < 0]
    log(f"done: {len(rows)} pairs, {len(neg)} with negative curvature"
        + (f" (most negative {min(r_['curvature'] for r_ in rows):+.4f})" if rows else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
