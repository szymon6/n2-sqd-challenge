"""Scoring reproduces the grader's energy and refuses invalid results."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from eval.reference import E_CCSD, E_HF, best_estimate, load_reference
from eval.score import grader_facts, main, score_result
from eval.validate import ValidationError
from n2sqd.rotation import rotate_integrals, rotation_from_params


def test_hf_subspace_scores_hf_energy(hf_result, ham):
    score = score_result(hf_result, ham, run_grader_code=False, reference_table={"e_hf": E_HF, "e_ccsd": E_CCSD})
    assert score.energy == pytest.approx(-108.330583, abs=1e-6)
    assert score.dims == (1, 1)
    assert score.matches
    assert score.gap_to_reference_mha is None and score.within_chemical_accuracy is None
    assert score.below_hf_mha == pytest.approx(0.0, abs=1e-3)


def test_offline_grader_agrees(small_result, ham):
    pytest.importorskip("qfest_grader")
    score = score_result(dataclasses.replace(small_result, claimed_energy=-108.0), ham)
    assert score.grader_energy is not None
    assert score.grader_agrees
    assert score.grader_energy == score.energy
    facts = grader_facts(small_result)
    assert facts["dims"] == [12, 12]


def test_claim_too_low_does_not_match(small_result, ham):
    score = score_result(dataclasses.replace(small_result, claimed_energy=-200.0), ham, run_grader_code=False)
    assert not score.matches


def test_gap_uses_reference_table(hf_result, ham):
    table = {"e_hf": E_HF, "e_ccsd": E_CCSD, "e_dmrg_extrapolated": -108.98}
    score = score_result(hf_result, ham, run_grader_code=False, reference_table=table)
    assert score.gap_to_reference_mha == pytest.approx((-108.330583 + 108.98) * 1e3, abs=1e-2)
    assert score.within_chemical_accuracy is False
    assert score.reference_label.startswith("DMRG extrapolated")


def test_rotated_result_scores_in_its_own_basis(small_result, ham):
    """Same strings, rotated integrals: a different (valid) energy, like the grader's own test."""
    u = rotation_from_params(np.random.default_rng(1).normal(scale=0.2, size=325), 26)
    h, g = rotate_integrals(ham.hcore, ham.eri, u)
    rotated = dataclasses.replace(small_result, hcore=h, eri=g)
    s0 = score_result(small_result, ham, run_grader_code=False)
    s1 = score_result(rotated, ham, run_grader_code=False)
    assert abs(s0.energy - s1.energy) > 1e-4
    assert s1.energy >= -108.98  # never below the exact energy region


def test_invalid_result_is_not_scored(hf_result, ham):
    bad = dataclasses.replace(hf_result, strings_a=np.array([7]))
    with pytest.raises(ValidationError):
        score_result(bad, ham, run_grader_code=False)


def test_reference_helpers(tmp_path):
    table = load_reference(tmp_path / "missing.json")
    assert table["e_hf"] == E_HF
    assert best_estimate(table) == (None, "none")
    (tmp_path / "ref.json").write_text(json.dumps({"e_dmrg": -108.98}))
    table = load_reference(tmp_path / "ref.json")
    assert best_estimate(table)[0] == -108.98
    assert best_estimate({"e_dmrg_extrapolated": -108.981})[0] == -108.981


def test_cli_writes_json(hf_result, tmp_path, capsys):
    path = hf_result.save(tmp_path / "r.npz")
    out = tmp_path / "score.json"
    assert main([str(path), "--no-grader", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["energy"] == pytest.approx(-108.330583, abs=1e-6)
    assert data["meta"] == {"id": "hf"}
    assert "seconds_total" in data
    assert "-108.330583" in capsys.readouterr().out
