"""Track A runner on synthetic samples with tiny dimensions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval.score import score_result
from n2sqd import tracka
from n2sqd.result import Result
from n2sqd.tests.test_lab_sqd import _synthetic_samples


def test_shots_matrix_and_pool_statistics(ham):
    samples = _synthetic_samples(ham, np.random.default_rng(8), n=300)
    m = tracka.shots_matrix(samples)
    assert m.shape == (300, 52) and m.dtype == bool
    stats = tracka.pool_statistics((np.array([31]), np.array([31])), samples, 26, (5, 5))
    assert stats["final_alpha_in_raw"] == 1 and stats["raw_valid_unique_alpha"] >= 1


def test_tracka_run_with_oo_and_pins(ham, tmp_path):
    samples = _synthetic_samples(ham, np.random.default_rng(9), n=600)
    path = tmp_path / "samples.npz"
    samples.save(path)
    out = tmp_path / "S0"
    argv = ["--out", str(out), "--samples", str(path), "--max-dim", "12", "--num-batches", "2", "--samples-per-batch", "60",
            "--max-iterations", "2", "--pinned-reference", "4", "--oo-cycles", "1", "--oo-macro", "1", "--seed", "3"]
    metrics = tracka.run(tracka.parse_args(argv), argv)
    assert metrics["dims"][0] <= 12 and metrics["dims"][1] <= 12
    assert metrics["energy"] < ham.e_hf
    assert metrics["cycles"][0]["phase"] == "sqd-0" and metrics["cycles"][1]["phase"] == "cycle-1"
    assert "energy_after_reselect" in metrics["cycles"][1]
    result = Result.load(out / "result.npz")
    assert set(np.array([31]).tolist()) <= set(result.strings_a.tolist())  # pinned HF string present
    score = score_result(result, ham, run_grader_code=False)
    assert score.matches and score.energy == pytest.approx(metrics["energy"], abs=2e-6)
    assert result.meta["track"] == "A" and result.meta["sampling"] == {"synthetic": True}
    for name in ("config.json", "command.txt", "log.txt", "metrics.json", "orbitals.npy"):
        assert (out / name).exists()
    data = json.loads((out / "metrics.json").read_text())
    assert data["pool_statistics"]["final_alpha_in_raw"] >= 0


def test_tracka_no_reselect_and_file_orbitals(ham, tmp_path):
    samples = _synthetic_samples(ham, np.random.default_rng(10), n=400)
    path = tmp_path / "samples.npz"
    samples.save(path)
    out = tmp_path / "S1"
    argv = ["--out", str(out), "--samples", str(path), "--max-dim", "8", "--num-batches", "1", "--samples-per-batch", "40",
            "--max-iterations", "1", "--oo-cycles", "1", "--oo-macro", "1", "--reselect", "none", "--no-symmetrize-spin"]
    m = tracka.run(tracka.parse_args(argv), argv)
    assert "energy_after_reselect" not in m["cycles"][1]
    out2 = tmp_path / "S2"
    argv2 = ["--out", str(out2), "--samples", str(path), "--max-dim", "8", "--num-batches", "1", "--samples-per-batch", "40",
             "--max-iterations", "1", "--orbitals", f"file:{out / 'orbitals.npy'}", "--start", f"file:{out / 'result.npz'}"]
    assert tracka.main(argv2) == 0
    r1, r2 = Result.load(out / "result.npz"), Result.load(out2 / "result.npz")
    assert np.array_equal(r1.strings_a, r2.strings_a) and r2.claimed_energy == pytest.approx(r1.claimed_energy, abs=1e-6)
