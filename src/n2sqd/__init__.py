"""N2 (10e, 26o) ground-state energy with sample-based quantum diagonalization.

Layout (top-down):
- :mod:`n2sqd.hamiltonian` builds the active-space integrals exactly as the grader does.
- :mod:`n2sqd.strings` holds CI-string helpers (HF string, excitation enumeration, bit tools).
- :mod:`n2sqd.rotation` rotates the integrals into another orbital basis.
- :mod:`n2sqd.solver` diagonalises H in a product subspace (grader-identical path and a warm-started path).
- :mod:`n2sqd.result` saves and loads result artifacts that :mod:`eval` scores.

See ``README.md`` for the problem, the results and how to reproduce them.
"""

from n2sqd.hamiltonian import N2Hamiltonian, build_n2_hamiltonian, load_or_build

__all__ = ["N2Hamiltonian", "build_n2_hamiltonian", "load_or_build"]
