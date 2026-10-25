"""Reference energies the gap is measured against.

``results/reference.json`` is written by ``python -m n2sqd.reference`` (DMRG) and by the baseline
script; this module only reads it and provides the fixed classical numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PATH = Path("results/reference.json")
CHEMICAL_ACCURACY_HA = 1.6e-3

# Computed in phase 1 with pyscf 2.14.0 in the (10e, 26o) active space.
E_HF = -108.330582753660
E_CISD = -108.779320811350
E_CCSD = -108.909520658608
E_CCSD_T = -109.012859740371


def load_reference(path: Path | str = DEFAULT_PATH) -> dict:
    """The reference table, or the fixed classical numbers when no DMRG file exists yet."""
    table = {"e_hf": E_HF, "e_cisd": E_CISD, "e_ccsd": E_CCSD, "e_ccsd_t": E_CCSD_T}
    path = Path(path)
    if path.exists():
        table.update(json.loads(path.read_text()))
    return table


def best_estimate(table: dict) -> tuple[float | None, str]:
    """The reference energy used for the gap: the DMRG estimate when present, else none."""
    if "e_dmrg_extrapolated" in table:
        return float(table["e_dmrg_extrapolated"]), "DMRG extrapolated (estimate)"
    if "e_dmrg" in table:
        return float(table["e_dmrg"]), "DMRG largest bond dimension (estimate, variational)"
    return None, "none"
