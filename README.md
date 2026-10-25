# N₂ ground-state energy with SQD

Qiskit Fall Fest 2026 Warsaw [Winner](https://qfest2026.com/verify/x3bgrq52e2)

**Result: −108.902027 Ha** for N₂ (cc-pVDZ, 2.0 Å, 2 frozen core orbitals, 10 electrons in 26 orbitals),
with 300 strings per spin.

|                                                       | energy / Ha             | gap to exact / mHa |
| ----------------------------------------------------- | ----------------------- | ------------------ |
| SQD with orbital optimization                         | −108.902027            | 80.0               |
| CCSD (classical reference)                            | −108.909521            | 72.5               |
| Lab 4b SQD workflow (best of 3 seeds, pinned strings) | −108.784924            | 197.1              |
| Hartree–Fock                                         | −108.330583            | 651.4              |
| Exact (DMRG, extrapolated)                            | −108.982028 ± 0.00016 | —                 |

## Quick start

1. Create the environment:

   ```bash
   uv venv --python 3.12 .venv
   source .venv/bin/activate          # fish: source .venv/bin/activate.fish
   uv pip install -e ".[dev,sim,notebooks]"
   ```

   Without `uv`: `python3.12 -m venv .venv && pip install -e ".[dev,sim,notebooks]"`.
   Versions that matter: `pyscf 2.14.0`, `qiskit 2.6.1`, `qiskit-addon-sqd 0.14.0`, `qfest-grader 0.3.0`.
2. Score the result the way the grader does (offline, no account needed, under 2 minutes):

   ```bash
   python -m eval.score results/best/track-a/result.npz
   ```
3. Run the tests:

   ```bash
   python -m pytest -q
   ```

## How it works

The grader takes two lists of at most 300 occupation strings (one per spin) plus the integrals. It diagonalizes
the Hamiltonian in the 300 × 300 product space with `qiskit_addon_sqd.fermion.solve_sci`. It accepts integrals in
any orbital basis that is a rotation of the reference one. So the score depends on two choices: which strings
and which orbitals.

The strings come from the official SQD loop on simulated LUCJ-circuit samples (matrix-product-state
simulator, 100 000 shots), with configuration recovery, carry-over and the lab's 100 reference strings pinned.
A CASSCF orbital optimization rotates the basis for the fixed string set.

## Reproduce

Runs write to `output/`. The logged metrics of the three runs behind the result are in `results/best/track-a/runs/`.

**Result** (about 3.5 h on an M4 laptop; the 100 000 samples are cached in
`results/samples/shots100000/`, so the first command returns at once):

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

**Baselines** (Hartree–Fock, the lab reference subspace, the lab's SQD loop on simulated LUCJ samples):

```bash
python -m n2sqd.baselines --out results/baselines --bond-dim 128 --shots 2000 --seeds 142 1 2
```

**Exact reference** (DMRG, several hours; needs `uv pip install -e ".[ref]"`):

```bash
python -m n2sqd.reference --max-bond 1000 --threads 4
```

## Notebooks

- [`solution-a.ipynb`](notebooks/solution-a.ipynb) — the result (−108.902027 Ha): scores it, plots its convergence and reproduces it from scratch.

## License

MIT, see [`LICENSE`](LICENSE).
