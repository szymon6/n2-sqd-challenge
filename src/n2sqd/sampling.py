"""Samples of the lab's LUCJ circuit without a quantum computer.

The lab samples 2 000 shots of a one-layer LUCJ circuit (CCSD amplitudes, Nighthawk interaction pairs) on
``ibm_miami``. Here the same circuit is simulated with Qiskit Aer's matrix-product-state method. An exact
statevector of the (5, 5) sector would need 4.3e9 amplitudes (69 GB), so the MPS bond dimension is capped;
the truncation breaks particle-number conservation in a fraction of the shots, much like hardware noise,
and the lab's configuration recovery repairs them.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from n2sqd.hamiltonian import N2Hamiltonian, build_molecule

NIGHTHAWK_PAIRS = 24  # the lab's recommended truncation of the alpha-beta interaction pairs


def interaction_pairs(norb: int = 26, num_ab_pairs: int = NIGHTHAWK_PAIRS):
    """Same-spin nearest neighbours and the first ``num_ab_pairs`` (p, p) cross-spin pairs (Exercise 1a)."""
    aa = [(p, p + 1) for p in range(norb - 1)]
    ab = [(p, p) for p in range(num_ab_pairs)]
    return aa, ab


def ccsd_amplitudes(ham: N2Hamiltonian):
    """CCSD t1 and t2 with the two core orbitals frozen (lab cell 15)."""
    import pyscf
    import pyscf.cc

    mol = build_molecule()
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf, frozen=[0, 1]).run(max_cycle=1000)
    return ccsd.t1, ccsd.t2, float(ccsd.e_tot)


def lucj_circuit(ham: N2Hamiltonian, n_reps: int = 1, num_ab_pairs: int = NIGHTHAWK_PAIRS, maxiter: int = 20):
    """The lab's LUCJ circuit with final measurements (cells 15, 27)."""
    import ffsim
    from qiskit.circuit import QuantumCircuit, QuantumRegister

    t1, t2, _ = ccsd_amplitudes(ham)
    pairs = interaction_pairs(ham.norb, num_ab_pairs)
    ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t1=t1, t2=t2, n_reps=n_reps, interaction_pairs=pairs, optimize=True, options=dict(maxiter=maxiter)
    )
    qubits = QuantumRegister(2 * ham.norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(ham.norb, ham.nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj), qubits)
    circuit.measure_all()
    return circuit


@dataclasses.dataclass
class Samples:
    """Unique bitstrings (n, 2·norb) as booleans in Qiskit string order, their probabilities, and run facts."""

    bitstrings: np.ndarray
    probs: np.ndarray
    shots: int
    valid_fraction: float  # shots with exactly nelec electrons per spin before recovery
    seconds: float
    meta: dict

    def save(self, path) -> None:
        np.savez_compressed(path, bitstrings=self.bitstrings, probs=self.probs, shots=self.shots,
                            valid_fraction=self.valid_fraction, seconds=self.seconds, meta=str(self.meta))

    @classmethod
    def load(cls, path) -> "Samples":
        d = np.load(path, allow_pickle=False)
        import ast

        return cls(d["bitstrings"], d["probs"], int(d["shots"]), float(d["valid_fraction"]), float(d["seconds"]),
                   ast.literal_eval(str(d["meta"])))


def counts_to_samples(counts: dict[str, int], norb: int, nelec: tuple[int, int], seconds: float, meta: dict) -> Samples:
    """Convert a Qiskit counts dictionary into :class:`Samples` (same layout as ``bit_array_to_arrays``)."""
    keys = list(counts)
    bits = np.array([[c == "1" for c in key.replace(" ", "")] for key in keys], dtype=bool)
    freqs = np.array([counts[k] for k in keys], dtype=float)
    shots = int(freqs.sum())
    beta, alpha = bits[:, :norb], bits[:, norb:]
    valid = (alpha.sum(axis=1) == nelec[0]) & (beta.sum(axis=1) == nelec[1])
    return Samples(bits, freqs / shots, shots, float(freqs[valid].sum() / shots), seconds, meta)


def sample_mps(circuit, shots: int, bond_dim: int, seed: int, norb: int, nelec: tuple[int, int],
               truncation_threshold: float = 1e-10) -> Samples:
    """Simulate ``circuit`` with Aer's matrix-product-state method and sample ``shots`` bitstrings."""
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    import ffsim

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
        matrix_product_state_truncation_threshold=truncation_threshold,
        max_parallel_shots=1,
        max_parallel_experiments=1,
        max_memory_mb=400_000,  # Aer's estimate assumes the worst-case bond dimension; the real MPS is small
    )
    pm = generate_preset_pass_manager(optimization_level=1, backend=sim)
    pm.pre_init = ffsim.qiskit.PRE_INIT
    isa = pm.run(circuit)
    t0 = time.perf_counter()
    counts = sim.run(isa, shots=shots, seed_simulator=seed).result().get_counts()
    seconds = time.perf_counter() - t0
    meta = {"simulator": "aer matrix_product_state", "bond_dim": bond_dim, "truncation_threshold": truncation_threshold,
            "seed": seed, "shots": shots, "ops": {k: int(v) for k, v in isa.count_ops().items()}, "depth": isa.depth()}
    return counts_to_samples(counts, norb, nelec, seconds, meta)
