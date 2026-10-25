# Best result — Track B (classical string selection + orbital optimisation)

`result.npz` — 300 × 300 CI strings and the integrals in the optimised orbital basis (`orbitals.npy` holds the
AO→MO coefficients; the two core columns are the RHF core orbitals, so the core energy is the lab's).

- Grader energy: **−108.949402 Ha** (`score.json`; offline grader code agrees; `matches` true; dry run of
  `python -m n2sqd.submit --track b` recomputes the same number)
- Strings per spin: 300 / 300; ⟨S²⟩ of the lowest eigenvector 0.92 (no spin constraint in the grader)
- Verified: a fresh run of the final command reproduced strings, integrals (to 8e-15) and energy; a rerun of the upstream 6.4 h chain U6
  reproduced its grow–prune rounds before the team stopped it
- Basin: this chain starts from the DMRG natural orbitals (`results/dmrg_rdm1.npy`, stage 2 reference run) and ends
  0.27 mHa below the RHF-derived basin of stages 1–6 (−108.949135, run U7 in `docs/results.md`,
  itself verified with every upstream step)

How it was reached (two steps, about 6.5 h on the 12-core Linux box with other jobs running):

```bash
# stage 6/7: full engine (ranked grow-prune with gradual prune, converged OO <-> re-selection) from the HF string
# in the DMRG natural orbitals
python -m n2sqd.trackb --out output/solution-b/chain --start hf --orbitals dmrg-no \
    --select-cutoff 5e-4 --ci-cutoff 5e-4 --max-new 700 --refine-iterations 6 --prune-steps 4 \
    --oo-cycles 6 --oo-macro 80 --reselect-iterations 2
# stage 7: polish with the 0.01 mHa cycle tolerance
python -m n2sqd.trackb --out output/solution-b/polish \
    --start file:output/solution-b/chain/result.npz \
    --orbitals file:output/solution-b/chain/orbitals.npy \
    --refine-iterations 0 --oo-cycles 6 --oo-macro 80 --reselect-iterations 2 --max-new 700 --prune-steps 4 \
    --select-cutoff 5e-4 --ci-cutoff 5e-4 --cycle-tol 1e-5
python -m eval.score output/solution-b/polish/result.npz
```

The DMRG 1-RDM comes from `python -m n2sqd.reference --max-bond 1000 --threads 4` (stage 2; `results/dmrg_rdm1.npy`
is committed).

Submission: the team decided on 2026-10-25 to submit Track B; `python -m n2sqd.submit --track b --submit --email "<team email>"`.
