# Best result — Track A (sample-based SQD)

`result.npz` — 300 × 300 CI strings produced by the official SQD loop of `qiskit-addon-sqd`
(`diagonalize_fermionic_hamiltonian`: configuration recovery, batching, carry-over, spin symmetrisation,
`max_dim` 300) on 100 000 Aer-MPS shots of the lab's LUCJ circuit, with the lab's 100 reference strings
pinned (Exercise 4), followed by three cycles of orbital optimisation of the fixed configurations (10 CASSCF
macro iterations) and five cycles with converged orbital optimisation (60 macro iterations), each followed by a
re-selection from the sampled pool in the rotated basis (half the strings carried over). The integrals are in
the optimised orbital basis (`orbitals.npy`).

- Grader energy: **−108.902027 Ha** (`score.json`; offline grader code agrees; `matches` true)
- Strings per spin: 300 / 300; 298 of them appear verbatim (as the alpha or beta half of a valid raw shot) in the
  raw samples, the other 2 are pinned lab reference strings; ⟨S²⟩ = 0.22
- Verified: a fresh run of the last step reproduced strings, integrals and energy exactly; the upstream step S5 was
  verified the same way
- Caveat: after orbital optimisation the strings are the sampled configurations as labels in the rotated basis
  (the SQD paper's convention, arXiv:2405.05068 Sec. II A 4)

Reproduce (about 3.5 h on an M4 laptop; the sampling step is cached in `results/samples/shots100000/`):

```bash
python -c "from pathlib import Path; from n2sqd.hamiltonian import load_or_build; from n2sqd.baselines import get_samples; \
get_samples(load_or_build(), Path('results/samples/shots100000'), 128, 100000, 142)"
python -m n2sqd.tracka --out output/solution-a/sqd \
    --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 \
    --max-iterations 10 --pinned-reference 100 --oo-cycles 0
python -m n2sqd.tracka --out output/solution-a/oo \
    --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 \
    --max-iterations 6 --start file:output/solution-a/sqd/result.npz \
    --pinned-reference 100 --oo-cycles 3 --oo-macro 10 --reselect sqd --carry-fraction 0.5
python -m n2sqd.tracka --out output/solution-a/deep-oo \
    --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 \
    --max-iterations 6 --start file:output/solution-a/oo/result.npz \
    --orbitals file:output/solution-a/oo/orbitals.npy \
    --pinned-reference 100 --oo-cycles 6 --oo-macro 60 --reselect sqd --carry-fraction 0.5 --seed 342
python -m eval.score output/solution-a/deep-oo/result.npz
```
