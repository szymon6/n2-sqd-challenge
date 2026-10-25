"""DMRG reference module: schedules, extrapolation and the CLI with a mocked DMRG run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from n2sqd import reference


def test_schedules():
    bds, noises, thrds = reference.dmrg_schedule(1000)
    assert bds[:4] == [250] * 4 and bds[-4:] == [1000] * 4 and bds.count(1000) == 8
    assert len(bds) == len(noises) == len(thrds)
    assert noises[-1] == 0.0 and noises[0] == 1e-4
    assert reference.reverse_schedule(2000) == [2000, 1500, 1000, 750, 500]
    assert reference.reverse_schedule(3000)[:2] == [3000, 2000]


def test_last_discarded_weight_handles_scalars_arrays_and_garbage():
    # block2 layout: (bond_dims, dws, energies)
    assert reference.last_discarded_weight(([1000, 1000], [2e-6, 3e-6], [[-108.9], [-108.98]])) == pytest.approx(3e-6)
    assert reference.last_discarded_weight(([1000], [np.array([1e-6, 4e-6])], [[-108.9]])) == pytest.approx(4e-6)
    assert np.isnan(reference.last_discarded_weight(([],)))
    assert np.isnan(reference.last_discarded_weight(([1], ["x"], [2])))


def test_reverse_sweeps_from_log_and_from_log_cli(tmp_path):
    log = tmp_path / "dmrg.log"
    lines = ["Time elapsed =      8.277 | E =    -108.3950182801 | DW = 5.54850e-10"] * 3
    lines.append("resumed MPS KET from data/dmrg_scratch: E = -108.9816588972")
    for bd, e, dw in ((1000, -108.98171, 8.5e-6), (1000, -108.981714, 8.56e-6), (750, -108.9815, 2.0e-5), (750, -108.98152, 2.1e-5),
                      (500, -108.9804, 6.0e-5), (500, -108.98041, 6.1e-5)):
        lines.append(f"Time elapsed =   2264.513 | E =    {e:.10f} | DE = -5.47e-08 | DW = {dw:.5e}")
    log.write_text("\n".join(lines) + "\n")
    energies, dws = reference.reverse_sweeps_from_log(log, [1000, 750, 500])
    assert energies == pytest.approx([-108.981714, -108.98152, -108.98041]) and dws == pytest.approx([8.56e-6, 2.1e-5, 6.1e-5])
    out = tmp_path / "ref.json"
    out.write_text(json.dumps({"e_dmrg": -108.981714}))
    assert reference.main(["--max-bond", "1000", "--from-log", str(log), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["e_dmrg"] == -108.981714 and data["reverse_bond_dims"] == [1000, 750, 500]
    assert data["e_dmrg_extrapolated"] < -108.9817 and data["e_dmrg_extrapolation_uncertainty"] > 0
    # too few sweeps: no extrapolation
    short = tmp_path / "short.log"
    short.write_text("resumed MPS\n" + lines[-2] + "\n" + lines[-1] + "\n")
    e2, d2 = reference.reverse_sweeps_from_log(short, [1000, 750, 500])
    assert len(e2) == 1


def test_main_passes_resume(tmp_path, monkeypatch):
    seen = {}

    def fake(ham, max_bond, threads, scratch, resume=False):
        seen["resume"] = resume
        return {"method": "mock", "e_dmrg": -108.0, "orbital_order": [0]}, np.eye(26)

    monkeypatch.setattr(reference, "run_dmrg", fake)
    monkeypatch.setattr(reference, "load_or_build", lambda: _FakeHam())
    assert reference.main(["--resume", "--out", str(tmp_path / "r.json"), "--rdm-out", str(tmp_path / "d.npy")]) == 0
    assert seen["resume"] is True


def test_extrapolate_is_linear_in_discarded_weight():
    dws = np.array([1e-5, 5e-6, 2e-6])
    es = -108.98 + 3.0 * dws
    e0, unc = reference.extrapolate(dws, es)
    assert e0 == pytest.approx(-108.98, abs=1e-9)
    assert unc == pytest.approx(3.0 * 2e-6 / 2, abs=1e-9)


def test_main_writes_table_with_mocked_dmrg(tmp_path, monkeypatch, capsys):
    table = {"method": "mock", "e_dmrg": -108.98, "e_dmrg_extrapolated": -108.981, "orbital_order": [0, 1]}
    dm1 = np.eye(26) * (10 / 26)
    monkeypatch.setattr(reference, "run_dmrg", lambda ham, max_bond, threads, scratch, resume=False: (table, dm1))
    monkeypatch.setattr(reference, "load_or_build", lambda: _FakeHam())
    out = tmp_path / "reference.json"
    out.write_text(json.dumps({"e_ccsd": -108.9095}))
    rdm = tmp_path / "rdm.npy"
    assert reference.main(["--max-bond", "500", "--out", str(out), "--rdm-out", str(rdm)]) == 0
    data = json.loads(out.read_text())
    assert data["e_ccsd"] == -108.9095 and data["e_dmrg"] == -108.98 and data["e_hf"] == -108.33
    assert np.load(rdm).shape == (26, 26)
    assert "orbital_order" not in capsys.readouterr().out


class _FakeHam:
    e_hf = -108.33
