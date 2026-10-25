from __future__ import annotations

import pytest

from n2sqd.hamiltonian import build_n2_hamiltonian


@pytest.fixture(scope="session")
def ham():
    return build_n2_hamiltonian()
