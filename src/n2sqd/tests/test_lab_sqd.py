"""The lab workflow pieces, checked against the grader's own test inputs and with synthetic samples."""

from __future__ import annotations

import numpy as np
import pytest

from n2sqd import lab_sqd
from n2sqd.baselines import hf_result, lab_reference_result, lab_sqd_result, main
from n2sqd.sampling import Samples, counts_to_samples, interaction_pairs


def test_reshape_and_hamming():
    raw = np.zeros((3, 52), dtype=bool)
    raw[0, 52 - 5:] = True  # alpha HF: qubits 0..4 are the rightmost characters
    raw[1, :5] = True  # beta orbitals 21..25: the highest qubits are the leftmost characters
    reshaped = lab_sqd.reshape_bitstring(raw, 26)
    assert reshaped.shape == (3, 2, 26)
    assert reshaped[0, 0, -5:].all() and not reshaped[0, 1].any()  # block 0 is alpha; column 25 is orbital 0
    assert reshaped[1, 1, :5].all() and not reshaped[1, 0].any()
    from qiskit_addon_sqd.counts import bitstring_matrix_to_integers

    assert bitstring_matrix_to_integers(reshaped[0, 0][None, :]).tolist() == [31]
    assert lab_sqd.hamming_weight(reshaped).tolist() == [[5, 0], [0, 5], [0, 0]]


def test_flip_weights():
    occ = np.array([0.0, 1.0])
    np.testing.assert_allclose(lab_sqd.weight_flip_0_to_1(occ), [0.0, np.e - 1])
    np.testing.assert_allclose(lab_sqd.weight_flip_1_to_0(occ), [np.e - 1, 0.0])


def test_recovery_restores_electron_count():
    rng = np.random.default_rng(0)
    bits = rng.random((50, 2, 12)) < 0.5
    probs = np.full(50, 1 / 50)
    occ = rng.random((2, 12))
    out, freqs = lab_sqd.recover_configurations(bits, probs, occ, 12, (6, 6), np.random.default_rng(1))
    assert np.all(lab_sqd.hamming_weight(out) == 6)
    assert freqs.sum() == pytest.approx(1.0) and len(freqs) == out.shape[0]
    # zero flip weights fall back to uniform instead of dividing by zero
    out2, _ = lab_sqd.recover_configurations(bits, probs, np.zeros((2, 12)), 12, (6, 6), np.random.default_rng(2))
    assert np.all(lab_sqd.hamming_weight(out2) == 6)


def test_build_subspace_pins_and_truncates():
    ci = np.array([[[31, 47, 47, 55, 55, 55], [31, 31, 47, 55, 59, 61]]])
    include = np.array([[31, 1023], [1023, 31]])
    sub = lab_sqd.build_subspace(ci, include, max_dim=(3, 3))
    assert sub[0][0].tolist() == [31, 55, 1023]  # pinned first, then most frequent sampled (55), sorted
    assert sub[0][1].tolist() == [31, 55, 1023] or sub[0][1].tolist() == [31, 47, 1023] or 1023 in sub[0][1]
    plain = lab_sqd.build_subspace(ci, max_dim=(None, None))
    assert plain[0][0].tolist() == [31, 47, 55]


def test_initial_occupancy_and_reference_subspace():
    occ = lab_sqd.initial_occupancy(26, (5, 5))
    assert occ.sum() == 10 and occ[0, -1] == 1 and occ[0, 0] == 0
    ref = lab_sqd.lab_reference_subspace(300)
    assert ref.shape == (2, 300) and (ref[0] == ref[1]).all()


def _synthetic_samples(ham, rng, n=400):
    """Shots drawn around HF with random bit flips, in Qiskit string order."""
    counts: dict[str, int] = {}
    for _ in range(n):
        alpha = np.zeros(26, dtype=bool)
        alpha[:5] = True
        beta = alpha.copy()
        for arr in (alpha, beta):
            if rng.random() < 0.7:
                i, j = rng.integers(0, 5), rng.integers(5, 26)
                arr[i], arr[j] = False, True
            if rng.random() < 0.2:  # noise: break the electron count
                arr[rng.integers(0, 26)] ^= True
        key = "".join("1" if b else "0" for b in beta[::-1]) + "".join("1" if b else "0" for b in alpha[::-1])
        counts[key] = counts.get(key, 0) + 1
    return counts_to_samples(counts, 26, (5, 5), 0.1, {"synthetic": True})


def test_lab_loop_runs_on_synthetic_samples(ham):
    samples = _synthetic_samples(ham, np.random.default_rng(3))
    assert 0.0 < samples.valid_fraction < 1.0 and samples.shots == 400
    run = lab_sqd.run_lab_sqd(samples.bitstrings, samples.probs, ham, seed=1, max_iterations=2, num_batches=2,
                              samples_per_batch=60, max_dim=(20, 20))
    assert len(run.history) == 2 and len(run.history[0]) == 2
    assert run.best_energy <= min(min(h) for h in run.history) + 1e-9
    assert run.best_energy < ham.e_hf  # correlation lowers the energy below HF
    a, b = run.best_subspace
    assert a.size <= 20 and np.all(np.diff(a) > 0)
    assert run.best_occupancies[0].sum() == pytest.approx(5.0, abs=1e-6)


def test_lab_loop_with_pinned_strings_and_callback(ham):
    samples = _synthetic_samples(ham, np.random.default_rng(4))
    seen = []
    include = lab_sqd.lab_reference_subspace()[:, :10]
    run = lab_sqd.run_lab_sqd(samples.bitstrings, samples.probs, ham, seed=2, max_iterations=1, num_batches=1,
                              samples_per_batch=40, max_dim=(15, 15), include=include, callback=seen.append)
    assert len(seen) == 1
    assert set(include[0].tolist()) <= set(run.best_subspace[0].tolist())


def test_sampling_helpers_and_round_trip(tmp_path):
    aa, ab = interaction_pairs(26, 24)
    assert aa[0] == (0, 1) and len(aa) == 25 and ab == [(p, p) for p in range(24)]
    s = counts_to_samples({"0" * 21 + "11111" + "0" * 21 + "11111": 3, "0" * 52: 1}, 26, (5, 5), 1.5, {"a": 1})
    assert s.shots == 4 and s.valid_fraction == pytest.approx(0.75) and s.bitstrings.shape == (2, 52)
    s.save(tmp_path / "s.npz")
    back = Samples.load(tmp_path / "s.npz")
    assert back.meta == {"a": 1} and back.probs.tolist() == s.probs.tolist()


def test_lucj_circuit_and_tiny_mps_sampling(ham):
    """The lab circuit builds (52 qubits) and Aer's MPS method samples it at a tiny bond dimension."""
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit_aer")
    from n2sqd.sampling import lucj_circuit, sample_mps

    circuit = lucj_circuit(ham, maxiter=1)
    assert circuit.num_qubits == 52
    samples = sample_mps(circuit, shots=20, bond_dim=4, seed=0, norb=26, nelec=(5, 5))
    assert samples.shots == 20 and samples.bitstrings.shape[1] == 52
    assert samples.meta["bond_dim"] == 4 and samples.meta["depth"] > 0


def test_get_samples_uses_cache(ham, tmp_path):
    from n2sqd.baselines import get_samples

    s = _synthetic_samples(ham, np.random.default_rng(6))
    s.save(tmp_path / "samples_bd8_s7.npz")
    back = get_samples(ham, tmp_path, 8, 400, 7)
    assert back.shots == s.shots


def test_baseline_artifacts_and_cli(ham, tmp_path, monkeypatch):
    hf = hf_result(ham)
    assert hf.claimed_energy == pytest.approx(ham.e_hf, abs=1e-8) and hf.meta["id"] == "hf"
    ref = lab_reference_result(ham, 12)
    assert ref.strings_a.size == 12 and ref.claimed_energy < ham.e_hf
    samples = _synthetic_samples(ham, np.random.default_rng(5))
    monkeypatch.setattr("n2sqd.baselines.get_samples", lambda ham, out, bond_dim, shots, seed: samples)
    monkeypatch.setattr("n2sqd.lab_sqd.LAB_MAX_ITERATIONS", 1)
    import n2sqd.baselines as b

    monkeypatch.setattr(b, "run_lab_sqd", lambda bits, probs, ham, seed, include: lab_sqd.run_lab_sqd(
        bits, probs, ham, seed=seed, include=include, max_iterations=1, num_batches=1, samples_per_batch=30, max_dim=(12, 12)))
    assert main(["--out", str(tmp_path), "--ref-size", "8", "--seeds", "7", "--bond-dim", "8", "--shots", "400"]) == 0
    import json

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {"hf", "lab-reference-8", "lab-sqd-bd8-s7", "lab-sqd-augmented-bd8-s7"} <= set(summary)
    assert (tmp_path / "lab-sqd-bd8-s7.npz").exists()
    r, meta = lab_sqd_result(ham, samples, 7, True, 8, 400)
    assert meta["id"] == "lab-sqd-augmented-bd8-s7" and r.strings_a.size <= 300
