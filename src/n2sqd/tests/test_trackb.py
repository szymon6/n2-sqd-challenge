"""Track B runner on a tiny budget: folder contents, artifact validity, natural-orbital start."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval.score import score_result
from n2sqd import trackb
from n2sqd.result import Result


def test_trackb_tiny_run_writes_everything(ham, tmp_path):
    out = tmp_path / "E0"
    argv = ["--out", str(out), "--start", "hf", "--budget", "8", "--refine-iterations", "1", "--max-new", "30",
            "--select-cutoff", "1e-2", "--ci-cutoff", "1e-2", "--oo-cycles", "1", "--oo-macro", "1", "--reselect-iterations", "1",
            "--prune-steps", "2"]
    metrics = trackb.run(trackb.parse_args(argv), argv)
    assert metrics["dims"] == [8, 8] and metrics["energy"] < ham.e_hf
    assert 0 <= metrics["spin_square"] < 2.1
    for name in ("config.json", "command.txt", "log.txt", "metrics.json", "result.npz", "orbitals.npy"):
        assert (out / name).exists()
    assert "python -m n2sqd.trackb --out" in (out / "command.txt").read_text()
    result = Result.load(out / "result.npz")
    score = score_result(result, ham, run_grader_code=False)
    assert score.matches and score.energy == pytest.approx(metrics["energy"], abs=2e-6)
    assert result.meta["track"] == "B" and result.meta["config"]["budget"] == 8
    metrics_file = json.loads((out / "metrics.json").read_text())
    history = metrics_file["history"]
    assert history[0]["phase"] == "refine-0" and history[1]["phase"] == "cycle-1"
    assert metrics_file["checkpoint_cycle"] is None and result.meta["checkpoint_cycle"] is None  # final write
    # a per-cycle checkpoint is written before the final write (stage 6): same files, checkpoint_cycle = cycle
    ckpt = trackb.save_outputs(out, trackb.parse_args(argv), "cmd", trackb.solve_subspace(result.strings_a, result.strings_b,
                               trackb.N2Hamiltonian(result.hcore, result.eri, result.core_energy)),
                               trackb.N2Hamiltonian(result.hcore, result.eri, result.core_energy), np.load(out / "orbitals.npy"),
                               {"orbitals": "x"}, history, trackb.RunLog(out / "log2.txt"), checkpoint=1)
    assert ckpt["checkpoint_cycle"] == 1 and Result.load(out / "result.npz").meta["checkpoint_cycle"] == 1


def test_trackb_natural_orbital_and_file_starts(ham, tmp_path):
    rdm = tmp_path / "rdm.npy"
    occ = np.zeros(26)
    occ[:5] = 1.9
    occ[5:7] = 0.25
    np.save(rdm, np.diag(occ)[::-1, ::-1])  # reversed order: natural orbitals must re-sort by occupation
    out = tmp_path / "E3"
    argv = ["--out", str(out), "--start", "hf", "--orbitals", "dmrg-no", "--dmrg-rdm", str(rdm), "--budget", "4",
            "--refine-iterations", "1", "--max-new", "10", "--select-cutoff", "1e-2", "--ci-cutoff", "1e-2"]
    metrics = trackb.run(trackb.parse_args(argv), argv)
    assert metrics["dims"] == [4, 4]
    meta = Result.load(out / "result.npz").meta
    assert meta["orbital_note"]["orbitals"] == "DMRG natural orbitals" and meta["orbital_note"]["occupations"][0] == pytest.approx(1.9)
    out2 = tmp_path / "E3b"
    argv2 = ["--out", str(out2), "--start", f"file:{out / 'result.npz'}", "--orbitals", f"file:{out / 'orbitals.npy'}",
             "--budget", "4", "--refine-iterations", "0", "--no-rank"]
    m2 = trackb.run(trackb.parse_args(argv2), argv2)
    assert m2["energy"] == pytest.approx(metrics["energy"], abs=1e-6)  # same strings, same orbitals
    assert Result.load(out2 / "result.npz").meta["orbital_note"]["orbitals"].startswith("from ")
    out3 = tmp_path / "E3c"
    argv3 = ["--out", str(out3), "--start", "hf", "--orbitals", f"no-from:{out}", "--budget", "4", "--refine-iterations", "0"]
    trackb.run(trackb.parse_args(argv3), argv3)
    note = Result.load(out3 / "result.npz").meta["orbital_note"]
    assert note["orbitals"].startswith("natural orbitals of") and sum(note["occupations"]) == pytest.approx(10.0, abs=1e-6)
    with pytest.raises(ValueError):
        trackb.starting_orbitals(trackb.TrackBConfig(out=tmp_path, orbitals="nope"), None)
    with pytest.raises(ValueError):
        trackb.starting_strings(trackb.TrackBConfig(out=tmp_path, start="nope"), ham)
    assert trackb.main(["--out", str(tmp_path / "E4"), "--start", "hf", "--budget", "2", "--refine-iterations", "0"]) == 0
    out5 = tmp_path / "E5"
    argv5 = ["--out", str(out5), "--start", "hf", "--budget", "6", "--refine-iterations", "1", "--max-new", "20",
             "--select-cutoff", "1e-2", "--ci-cutoff", "1e-2", "--prune", "loss", "--asymmetric"]
    m5 = trackb.run(trackb.parse_args(argv5), argv5)
    assert m5["dims"] == [6, 6] and m5["energy"] < ham.e_hf
    cfg = json.loads((out5 / "config.json").read_text())
    assert cfg["prune"] == "loss" and cfg["asymmetric"] is True and "git_commit" in cfg
