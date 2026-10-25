"""The submission tool is a dry run by default and never contacts the server without --submit."""

from __future__ import annotations

import numpy as np
import pytest

from n2sqd import submit
from n2sqd.result import Result


def test_default_is_dry_run_and_sends_nothing(ham, tmp_path, capsys, monkeypatch):
    hf = np.array([31])
    path = Result(hf, hf, ham.hcore, ham.eri, ham.core_energy, ham.e_hf, {"id": "hf"}).save(tmp_path / "r.npz")
    monkeypatch.setattr(submit, "score_result", lambda result, run_grader_code=True: _FakeScore(True, True))
    assert submit.main(["--artifact", str(path)]) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing sent" in out and "matches=True" in out


def test_refuses_mismatching_claim(ham, tmp_path, monkeypatch):
    hf = np.array([31])
    path = Result(hf, hf, ham.hcore, ham.eri, ham.core_energy, -200.0, {"id": "bad"}).save(tmp_path / "r.npz")
    monkeypatch.setattr(submit, "score_result", lambda result, run_grader_code=True: _FakeScore(False, True))
    assert submit.main(["--artifact", str(path)]) == 2


def test_argument_errors():
    with pytest.raises(SystemExit):
        submit.main([])  # neither --track nor --artifact
    with pytest.raises(SystemExit):
        submit.main(["--track", "b", "--submit"])  # real submission without --email
    with pytest.raises(SystemExit):
        submit.main(["--track", "b", "--email", "team@example.org"])  # email without --submit


class _FakeScore:
    def __init__(self, matches, agrees):
        self.dims = (1, 1)
        self.claimed_energy = -108.330583
        self.energy = -108.330583
        self.matches = matches
        self.grader_agrees = agrees
        self.gap_to_reference_mha = 651.4
        self.below_ccsd_mha = -578.9
