"""CI-string helpers.

A CI string is an integer whose bit ``p`` is 1 when spatial orbital ``p`` is occupied by an electron of
that spin. The grader wants, per spin, a strictly ascending ``int64`` array of such strings with exactly
``nelec`` bits set and values below ``2**norb``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from n2sqd.hamiltonian import NELEC, NORB


def hf_string(nocc: int = NELEC[0]) -> int:
    """Hartree–Fock string: the ``nocc`` lowest orbitals occupied (``0b11111`` = 31 for N2)."""
    return (1 << nocc) - 1


def popcount(strings: np.ndarray) -> np.ndarray:
    """Number of set bits of every string (vectorised)."""
    s = np.asarray(strings, dtype=np.uint64)
    out = np.zeros(s.shape, dtype=np.int64)
    while np.any(s):
        out += (s & np.uint64(1)).astype(np.int64)
        s = s >> np.uint64(1)
    return out


def to_bits(strings: np.ndarray, norb: int = NORB) -> np.ndarray:
    """Boolean matrix (n_strings, norb); column ``p`` is the occupation of orbital ``p``."""
    s = np.asarray(strings, dtype=np.int64)[:, None]
    return ((s >> np.arange(norb, dtype=np.int64)) & 1).astype(bool)


def from_bits(bits: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_bits`."""
    b = np.asarray(bits, dtype=np.int64)
    return (b << np.arange(b.shape[-1], dtype=np.int64)).sum(axis=-1)


def all_strings(norb: int = NORB, nocc: int = NELEC[0]) -> np.ndarray:
    """Every string with ``nocc`` of ``norb`` orbitals occupied, ascending (65 780 for N2)."""
    out = [sum(1 << p for p in occ) for occ in combinations(range(norb), nocc)]
    return np.array(sorted(out), dtype=np.int64)


def excitations(reference: int, norb: int, max_rank: int) -> list[list[int]]:
    """Strings grouped by excitation rank from ``reference``: rank 0 (the reference), 1, ..., ``max_rank``.

    Within a rank, holes and particles are enumerated with :func:`itertools.combinations` in index order,
    which is the lab's "fill the lowest orbitals first" enumeration.
    """
    occ = [p for p in range(norb) if reference >> p & 1]
    virt = [p for p in range(norb) if not reference >> p & 1]
    groups: list[list[int]] = [[reference]]
    for rank in range(1, max_rank + 1):
        group = []
        for holes in combinations(occ, rank):
            for parts in combinations(virt, rank):
                s = reference
                for h in holes:
                    s ^= 1 << h
                for p in parts:
                    s |= 1 << p
                group.append(s)
        groups.append(group)
    return groups


def lab_reference_strings(n: int = 300, norb: int = NORB, nocc: int = NELEC[0]) -> np.ndarray:
    """The lab's Exercise 4 reference subspace: the first ``n`` lowest-excitation-rank strings, sorted.

    Rank 0 then singles (105 for N2) then doubles in enumeration order, cut at ``n`` before sorting.
    """
    flat: list[int] = []
    for group in excitations(hf_string(nocc), norb, max_rank=nocc):
        flat.extend(group)
        if len(flat) >= n:
            break
    return np.array(sorted(set(flat[:n])), dtype=np.int64)


def canonical(strings) -> np.ndarray:
    """Unique, ascending ``int64`` array, as the grader requires."""
    return np.unique(np.asarray(strings, dtype=np.int64))
