# Results

## Final result (updated 2026-10-25, stage 7; Track B is the submission, team decision)

| | energy / Ha | vs best baseline | vs CCSD | vs exact (DMRG estimate −108.982028 ± 0.16 mHa) | status |
|---|---|---|---|---|---|
| **Track B** (classical string selection + orbital optimisation; chain from DMRG natural orbitals) | **−108.949402** | −164.5 mHa | −39.9 mHa | +32.6 mHa | verified (stage 7), `results/best/track-b/` |
| **Track A** (official SQD loop on simulated LUCJ samples + pinned lab strings + converged orbital optimisation) | **−108.902027** | −117.1 mHa | +7.5 mHa | +80.0 mHa | verified (stage 5), `results/best/track-a/` |
| best baseline (lab workflow with pinned reference strings, best of 3 seeds) | −108.784924 | — | +124.6 mHa | +197.1 mHa | `results/best/baseline/` |

Both tracks use 300 strings per spin; every energy is the grader's own computation (offline cross-check agreed);
each best artifact was reproduced by a fresh run of its recorded command (strings identical, integrals to ≤ 1.3e-7). The team chose Track B as
the submission on 2026-10-25; Track A stays as the sample-based reference. Chemical accuracy (1.6 mHa) was not reached by either track.

Every row is scored by the harness (`python -m eval.score <artifact>`); `E` is the
grader energy (6 decimals) and the offline grader cross-check agreed on every row (baseline score files in
`results/scores/`, stage score files as `score.json` next to each experiment's artifact).
Gap = `E − E_ref` in mHa with `E_ref` = −108.982028 Ha, the DMRG estimate extrapolated to zero discarded weight (`results/reference.json`, uncertainty 0.16 mHa). Spread =
min–max over seeds where applicable. Cost = wall time on the M4 laptop (stages 1–5) or the 12-core Linux machine (stages 6–7). Tracks: `baseline`, `A`
(sample-based SQD), `B` (best energy).

## Official grader

Rule: the official grader runs as a **dry run** only; nothing is submitted until the team explicitly
decides to. `python -m n2sqd.submit --track <a|b>` runs the grader's own energy code locally and
sends nothing; `--submit --email <team email>` is the only path that contacts the server.

**Dry run** (outputs in `results/scores/submit-dry-run-track-{a,b}.txt`; Track B re-run on 2026-10-25 for the stage 7 artifact, Track A on 2026-10-24):

| track | artifact | strings per spin | claimed | recomputed by grader code | matches | offline grader agrees |
|---|---|---|---|---|---|---|
| B | `results/best/track-b/result.npz` | 300 / 300 | −108.949402 | −108.949402 | yes | yes |
| A | `results/best/track-a/result.npz` | 300 / 300 | −108.902027 | −108.902027 | yes | yes |

**Submission:** none yet. The grader's output goes here once the team submits.

## Reference energies (not submittable)

| id | method | E / Ha | note |
|---|---|---|---|
| e_hf | RHF | −108.330583 | 1 × 1 subspace, string 31 |
| e_cisd | CISD, frozen core | −108.779321 | PySCF |
| e_ccsd | CCSD, frozen core | −108.909521 | PySCF, the lab's "hard to beat" line |
| e_ccsd_t | CCSD(T), frozen core | −109.012860 | not variational; ~30 mHa *below* exact here |
| e_dmrg | DMRG block2, SU(2), bond dim 1000 | −108.981659 (variational, `e_dmrg`) | `python -m n2sqd.reference --max-bond 1000 --threads 4` (+ `--resume`, `--from-log` after the bookkeeping crash); `results/reference.json` |
| **e_ref** | DMRG extrapolated to zero discarded weight (bond dims 1000/750/500) | **−108.982028 ± 0.16 mHa** | the reference for the gap column; literature estimate −108.983 ± 0.3 mHa agrees within 1 mHa |

## Current best per track

| track | id | E / Ha | artifact |
|---|---|---|---|
| baseline | lab-sqd-augmented-bd128-s1 | −108.784924 | `results/best/baseline/result.npz` (lowest of the three seeds) |
| A | T1-deep-oo-from-S5 (stage 5) | **−108.902027** (verified: fresh rerun identical; upstream S5 verified too) | `results/best/track-a/result.npz` |
| B | V1-polish-U6-dmrg-no-chain (stage 7) | **−108.949402** (verified: fresh rerun identical; upstream chain U6 rerun stopped by the team after the grow–prune rounds reproduced) | `results/best/track-b/result.npz` |

## Results table

| id | stage | track | description | E / Ha | gap / mHa | strings per spin | spread | cost | command |
|---|---|---|---|---|---|---|---|---|---|
| hf | 0 | baseline | Hartree–Fock determinant only | −108.330583 | +651.4 | 1 | — | 0.01 s | `python -m n2sqd.baselines --only hf` |
| lab-reference-300 | 0 | baseline | lab Exercise 4: first 300 lowest-excitation-rank strings | −108.757711 | +224.3 | 300 | — | 22 s | `python -m n2sqd.baselines --only lab-reference` |
| lab-sqd-bd128-s142 | 0 | baseline | lab SQD loop as given, MPS samples (bond dim 128, 2 000 shots, 33–35 % of shots with the right electron count) | −108.725897 | +256.1 | 300 | seeds 142/1/2: −108.725897 / −108.731079 / −108.717547 | 287–313 s sampling + 794–860 s loop | `python -m n2sqd.baselines --only lab-sqd --bond-dim 128 --shots 2000 --seeds 142 1 2` |
| lab-sqd-augmented-bd128-s142 | 0 | baseline | same loop, first 100 reference strings pinned | **−108.783919** | +198.1 | 300 | seeds 142/1/2: −108.783919 / −108.784924 / −108.779533 | 652–954 s loop | `python -m n2sqd.baselines --only lab-sqd-augmented --bond-dim 128 --shots 2000 --seeds 142 1 2` |

| E1a-labref-rhf-c1e-3 | 1 | B | grow–prune (unranked cap 700) from the lab reference subspace, RHF orbitals | −108.894811 | +87.2 | 300 | — | 767 s | `python -m n2sqd.trackb --out output/1-track-b-engine/E1a-labref-rhf-c1e-3 --start lab-ref --orbitals rhf --select-cutoff 1e-3 --ci-cutoff 1e-3 --max-new 700 --refine-iterations 6 --oo-cycles 0 --no-rank` |
| E1b-hf-rhf-c1e-3 | 1 | B | grow–prune (unranked cap 700) from the HF string, RHF orbitals | −108.900130 | +81.9 | 300 | — | 647 s | same with `--out output/1-track-b-engine/E1b-hf-rhf-c1e-3 --start hf` |
| E1c-ranked-from-E1b | 1 | B | grow–prune with PT2-ranked candidates, from the E1b strings, RHF orbitals | −108.918736 | +63.3 | 300 | — | 1120 s | `python -m n2sqd.trackb --out output/1-track-b-engine/E1c-ranked-from-E1b --start file:output/1-track-b-engine/E1b-hf-rhf-c1e-3/result.npz --orbitals rhf --refine-iterations 6 --max-new 700 --oo-cycles 0` |
| E3a-no-from-E1c-hf | 1 | B | ranked grow–prune from the HF string in the natural orbitals of the E1c state | −108.928320 | +53.7 | 300 | — | 1279 s | `python -m n2sqd.trackb --out output/1-track-b-engine/E3a-no-from-E1c-hf --start hf --orbitals no-from:output/1-track-b-engine/E1c-ranked-from-E1b --select-cutoff 1e-3 --ci-cutoff 1e-3 --max-new 700 --refine-iterations 6 --oo-cycles 0` |
| E2a-oo-from-E1b | 1 | B | 5 × (CASSCF orbital optimisation ↔ ranked reselection) from the E1b strings, RHF start | **−108.937535** | +44.5 | 300 | fresh rerun identical (E2a-oo-from-E1b-verify, 2485 s) | 3959 s (contended) | `python -m n2sqd.trackb --out output/1-track-b-engine/E2a-oo-from-E1b --start file:output/1-track-b-engine/E1b-hf-rhf-c1e-3/result.npz --orbitals rhf --refine-iterations 0 --oo-cycles 5 --oo-macro 10 --reselect-iterations 2 --max-new 700` |
| E2b-oo-from-E3a | 1 | B | 3 × (orbital optimisation ↔ ranked reselection) from the E3a natural-orbital state | −108.937096 | +44.9 | 300 | two independent starts (E2a, E2b) agree within 0.44 mHa | 1610 s | `python -m n2sqd.trackb --out output/1-track-b-engine/E2b-oo-from-E3a --start file:output/1-track-b-engine/E3a-no-from-E1c-hf/result.npz --orbitals file:output/1-track-b-engine/E3a-no-from-E1c-hf/orbitals.npy --refine-iterations 0 --oo-cycles 4 --oo-macro 10 --reselect-iterations 2 --max-new 700` |

| S1-shots2000 | 2 | A | qiskit-addon-sqd loop (carry-over, spin symmetrised, max_dim 300) on the 2 000 MPS shots of the baseline | −108.808809 | +173.2 | 300 | — | 1164 s | `python -m n2sqd.tracka --out output/2-track-a-samples/S1-shots2000 --samples results/baselines/samples_bd128_s142.npz --samples-per-batch 500 --num-batches 5 --max-iterations 10 --oo-cycles 0` |
| S1-shots20000 | 2 | A | same loop, 20 000 shots | −108.824519 | +157.5 | 300 | — | 721 s (+ 801 s sampling) | `python -m n2sqd.tracka --out output/2-track-a-samples/S1-shots20000 --samples results/samples/shots20000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 --max-iterations 10 --oo-cycles 0` |
| S1-shots100000 | 2 | A | same loop, 100 000 shots | −108.825812 | +156.2 | 300 | — | 796 s (+ 1674 s sampling) | `python -m n2sqd.tracka --out output/2-track-a-samples/S1-shots100000 --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 --max-iterations 10 --oo-cycles 0` |
| S2-oo-fixed-from-S1-2000 | 2 | A | one orbital-optimisation step on the fixed S1-shots2000 strings | −108.853592 | +128.4 | 300 | — | 491 s | `python -m n2sqd.tracka --out output/2-track-a-samples/S2-oo-fixed-from-S1-2000 --samples results/baselines/samples_bd128_s142.npz --start file:output/2-track-a-samples/S1-shots2000/result.npz --oo-cycles 1 --oo-macro 10 --reselect none` |
| S3-oo-sqd-from-S1-100000 | 2 | A | 3 × (OO on fixed strings → SQD re-selection in the rotated basis, 150 strings carried over), from S1-shots100000, no pins | −108.885272 | +96.8 | 300 | — | 2934 s | `python -m n2sqd.tracka --out output/2-track-a-samples/S3-oo-sqd-from-S1-100000 --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 --max-iterations 6 --start file:output/2-track-a-samples/S1-shots100000/result.npz --oo-cycles 3 --oo-macro 10 --reselect sqd --carry-fraction 0.5` |
| S4-pinned100-shots100000 | 2 | A | SQD loop with the lab's 100 reference strings pinned, 100 000 shots | −108.865799 | +116.2 | 300 | — | 384 s | `python -m n2sqd.tracka --out output/2-track-a-samples/S4-pinned100-shots100000 --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 --max-iterations 10 --pinned-reference 100 --oo-cycles 0` |
| S6a / S6b | 2 | A | SQD loop with 100 pinned strings, 20 000 shots, sampler seed 142 / 1 | −108.863885 / −108.864446 | +118.1 | 300 | 0.56 mHa between seeds | 460 / 463 s | `python -m n2sqd.tracka --out output/2-track-a-samples/S6a-pinned100-shots20000-seed142 --samples results/samples/shots20000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 --max-iterations 10 --pinned-reference 100 --oo-cycles 0` (S6b: `..._s1.npz --seed 1`) |
| S5-pinned100-oo-sqd-from-S4 | 2 | A | S4 strings + 3 × (OO → SQD re-selection with pins) | **−108.894649** | +87.4 | 300 | fresh rerun identical (S5-pinned100-oo-sqd-from-S4-verify, 2486 s) | 2544 s | `python -m n2sqd.tracka --out output/2-track-a-samples/S5-pinned100-oo-sqd-from-S4 --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 --max-iterations 6 --start file:output/2-track-a-samples/S4-pinned100-shots100000/result.npz --pinned-reference 100 --oo-cycles 3 --oo-macro 10 --reselect sqd --carry-fraction 0.5` |

| S5b-continue-3-more-cycles | 2 | A | S5 strings and orbitals + 3 more (OO → SQD re-selection) cycles | −108.899833 | +82.2 | 300 | fresh rerun identical (stage 5, `output/5-track-a-deep-oo/S5b-verify`, 2209 s) | 2469 s | `python -m n2sqd.tracka --out output/2-track-a-samples/S5b-continue-3-more-cycles --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 --max-iterations 6 --start file:output/2-track-a-samples/S5-pinned100-oo-sqd-from-S4/result.npz --orbitals file:output/2-track-a-samples/S5-pinned100-oo-sqd-from-S4/orbitals.npy --pinned-reference 100 --oo-cycles 3 --oo-macro 10 --reselect sqd --carry-fraction 0.5 --seed 242` |

| P0-control | 3 | B | stage 1 best + 3 more cycles with stage 1 settings (control) | −108.937596 | +44.4 | 300 | — | 486 s | `python -m n2sqd.trackb --out output/3-track-b-pool-and-prune/P0-control --start file:results/best/track-b/result.npz --orbitals file:results/best/track-b/orbitals.npy --refine-iterations 0 --oo-cycles 3 --oo-macro 10 --reselect-iterations 2 --max-new 700` |
| P1-pool1500 | 3 | B | as P0 with a 1500-string ranked pool | −108.937541 | +44.5 | 300 | — | 881 s | `python -m n2sqd.trackb --out output/3-track-b-pool-and-prune/P1-pool1500 --start file:results/best/track-b/result.npz --orbitals file:results/best/track-b/orbitals.npy --refine-iterations 0 --oo-cycles 3 --oo-macro 10 --reselect-iterations 2 --max-new 1500` |
| P2-prune-loss | 3 | B | as P0 with the removal-loss prune criterion | −108.937541 | +44.5 | 300 | — | 496 s | `python -m n2sqd.trackb --out output/3-track-b-pool-and-prune/P2-prune-loss --start file:results/best/track-b/result.npz --orbitals file:results/best/track-b/orbitals.npy --refine-iterations 0 --oo-cycles 3 --oo-macro 10 --reselect-iterations 2 --max-new 700 --prune loss` |
| P3-asymmetric | 3 | B | as P0 with independent alpha/beta sets | −108.937596 | +44.4 | 300 | — | 504 s | `python -m n2sqd.trackb --out output/3-track-b-pool-and-prune/P3-asymmetric --start file:results/best/track-b/result.npz --orbitals file:results/best/track-b/orbitals.npy --refine-iterations 0 --oo-cycles 3 --oo-macro 10 --reselect-iterations 2 --max-new 700 --asymmetric` |
| P4-cutoffs5e-4 | 3 | B | as P0 with heat-bath cutoffs 5e-4 / 5e-4 | −108.937818 | +44.2 | 300 | fresh rerun identical | 941 s | `python -m n2sqd.trackb --out output/3-track-b-pool-and-prune/P4-cutoffs5e-4 --start file:results/best/track-b/result.npz --orbitals file:results/best/track-b/orbitals.npy --refine-iterations 0 --oo-cycles 3 --oo-macro 10 --reselect-iterations 2 --max-new 700 --select-cutoff 5e-4 --ci-cutoff 5e-4` |
| P5-oomacro40-from-P4 | 3 | B | P4 state + 3 cycles with 40 CASSCF macro iterations per OO step | **−108.948837** | +33.2 | 300 | fresh rerun identical (P5-oomacro40-from-P4-verify, 2759 s) | 2908 s | `python -m n2sqd.trackb --out output/3-track-b-pool-and-prune/P5-oomacro40-from-P4 --start file:output/3-track-b-pool-and-prune/P4-cutoffs5e-4/result.npz --orbitals file:output/3-track-b-pool-and-prune/P4-cutoffs5e-4/orbitals.npy --refine-iterations 0 --oo-cycles 3 --oo-macro 40 --reselect-iterations 2 --max-new 700 --select-cutoff 5e-4 --ci-cutoff 5e-4` |

| Q1-deep-oo-from-P5 | 4 | B | P5 state + converged OO (80 macro) ↔ re-selection until the gain < 0.1 mHa | **−108.948892** | +33.1 | 300 | fresh rerun identical (Q1-deep-oo-from-P5-verify, 522 s) | 511 s | `python -m n2sqd.trackb --out output/4-oo-depth/Q1-deep-oo-from-P5 --start file:output/3-track-b-pool-and-prune/P5-oomacro40-from-P4/result.npz --orbitals file:output/3-track-b-pool-and-prune/P5-oomacro40-from-P4/orbitals.npy --refine-iterations 0 --oo-cycles 8 --oo-macro 80 --reselect-iterations 2 --max-new 700 --select-cutoff 5e-4 --ci-cutoff 5e-4` |

| T1-deep-oo-from-S5 | 5 | A | S5 state + cycles of converged OO (60 macro) ↔ SQD re-selection with pins, until the gain < 0.1 mHa | **−108.902027** | +80.0 | 300 | fresh rerun identical (T1-deep-oo-from-S5-verify, 6007 s) | 6016 s | `python -m n2sqd.tracka --out output/5-track-a-deep-oo/T1-deep-oo-from-S5 --samples results/samples/shots100000/samples_bd128_s142.npz --samples-per-batch 1000 --num-batches 5 --max-iterations 6 --start file:output/2-track-a-samples/S5-pinned100-oo-sqd-from-S4/result.npz --orbitals file:output/2-track-a-samples/S5-pinned100-oo-sqd-from-S4/orbitals.npy --pinned-reference 100 --oo-cycles 6 --oo-macro 60 --reselect sqd --carry-fraction 0.5 --seed 342` |

| U1-gradual-prune | 6 | B | stage 4 best + re-selection in the same (converged) basis with a gradual prune (1000-string pool, 4 prune–resolve rounds) | −108.949031 | +33.0 | 300 | fresh rerun identical (U1-gradual-prune-verify, 3172 s) | 3034 s (12-core Linux box, 4 threads, two other jobs running) | `python -m n2sqd.trackb --out output/6-symmetry-breaking/U1-gradual-prune --start file:results/best/track-b/result.npz --orbitals file:results/best/track-b/orbitals.npy --refine-iterations 3 --oo-cycles 0 --prune-steps 4 --max-new 700 --select-cutoff 5e-4 --ci-cutoff 5e-4` |
| U1b-gradual-prune-pool1500 | 6 | B | U1 state + re-selection from a 1500-string ranked pool (1800 × 1800) pruned in six rounds | −108.949031 | +33.0 | 300 | — | 5034 s | `python -m n2sqd.trackb --out output/6-symmetry-breaking/U1b-gradual-prune-pool1500 --start file:output/6-symmetry-breaking/U1-gradual-prune/result.npz --orbitals file:output/6-symmetry-breaking/U1-gradual-prune/orbitals.npy --refine-iterations 2 --oo-cycles 0 --prune-steps 6 --max-new 1500 --select-cutoff 5e-4 --ci-cutoff 5e-4` |
| U2a-kick-cross-0.05-s1 | 6 | B | stage 4 strings in a randomly kicked basis (cross-class rotation, ≤ 16.6° per orbital, +233 mHa) + converged OO ↔ re-selection | −108.948893 | +33.1 | 300 | — | 4387 s | `python -m n2sqd.symmetry kick --folder results/best/track-b --out output/6-symmetry-breaking/U2a-kick-cross-0.05-s1/kicked.npy --scale 0.05 --seed 1 --mode cross` then `python -m n2sqd.trackb --out output/6-symmetry-breaking/U2a-kick-cross-0.05-s1 --start file:results/best/track-b/result.npz --orbitals file:output/6-symmetry-breaking/U2a-kick-cross-0.05-s1/kicked.npy --refine-iterations 0 --oo-cycles 8 --oo-macro 80 --reselect-iterations 2 --max-new 700 --select-cutoff 5e-4 --ci-cutoff 5e-4` (the `results/best/track-b/` start was the Q1 artifact at the time) |
| U3-deep-oo-gradual-from-U1 | 6 | B | U1 state + converged OO (80 macro) ↔ gradual-prune re-selection until the cycle gain < 0.1 mHa | **−108.949119** | +32.9 | 300 | fresh rerun identical (U3-deep-oo-gradual-from-U1-verify, 2560 s) | 3431 s (one cycle, 4 threads, three other jobs running) | `python -m n2sqd.trackb --out output/6-symmetry-breaking/U3-deep-oo-gradual-from-U1 --start file:output/6-symmetry-breaking/U1-gradual-prune/result.npz --orbitals file:output/6-symmetry-breaking/U1-gradual-prune/orbitals.npy --refine-iterations 0 --oo-cycles 6 --oo-macro 80 --reselect-iterations 2 --max-new 700 --prune-steps 4 --select-cutoff 5e-4 --ci-cutoff 5e-4` |
| U5-pm-local-chain | 6 | B | full engine (ranked grow–prune with gradual prune, then converged OO ↔ re-selection) from the HF string in Pipek–Mezey localised RHF orbitals: a different basin | -108.918851 | +63.2 | 300 | — | 9737 s | `python -m n2sqd.trackb --out output/6-symmetry-breaking/U5-pm-local-chain --start hf --orbitals pm-local --select-cutoff 5e-4 --ci-cutoff 5e-4 --max-new 700 --refine-iterations 6 --prune-steps 4 --oo-cycles 6 --oo-macro 80 --reselect-iterations 2` |
| U7-polish-from-U3 | 6 | B | U3 state + the same cycles with a tighter stop rule (0.01 mHa per cycle) | **-108.949135** | +32.9 | 300 | fresh rerun identical (U7-polish-from-U3-verify, 4012 s) | 3982 s | `python -m n2sqd.trackb --out output/6-symmetry-breaking/U7-polish-from-U3 --start file:output/6-symmetry-breaking/U3-deep-oo-gradual-from-U1/result.npz --orbitals file:output/6-symmetry-breaking/U3-deep-oo-gradual-from-U1/orbitals.npy --refine-iterations 0 --oo-cycles 6 --oo-macro 80 --reselect-iterations 2 --max-new 700 --prune-steps 4 --select-cutoff 5e-4 --ci-cutoff 5e-4 --cycle-tol 1e-5` |

| U6-dmrg-no-chain | 7 | B | full engine (ranked grow–prune with gradual prune, converged OO ↔ re-selection) from the HF string in the DMRG natural orbitals: a different basin | **−108.949401** | +32.6 | 300 | rerun stopped by the team after the grow–prune rounds reproduced (U6-dmrg-no-chain-verify) | 23166 s (6.4 h, shared machine) | `python -m n2sqd.trackb --out output/6-symmetry-breaking/U6-dmrg-no-chain --start hf --orbitals dmrg-no --select-cutoff 5e-4 --ci-cutoff 5e-4 --max-new 700 --refine-iterations 6 --prune-steps 4 --oo-cycles 6 --oo-macro 80 --reselect-iterations 2` |
| U4-uhf-no-chain | 7 | B | same engine from the HF string in the natural orbitals of the spin-polarised UHF | −108.949091 | +32.9 | 300 | — | 32796 s (9.1 h, no CASSCF step converged) | `python -m n2sqd.trackb --out output/6-symmetry-breaking/U4-uhf-no-chain --start hf --orbitals uhf-no --select-cutoff 5e-4 --ci-cutoff 5e-4 --max-new 700 --refine-iterations 6 --prune-steps 4 --oo-cycles 6 --oo-macro 80 --reselect-iterations 2` |
| V1-polish-U6-dmrg-no-chain | 7 | B | U6 state + polish (gradual prune, 0.01 mHa cycle tolerance) | **−108.949402** | +32.6 | 300 | fresh rerun identical (V1-polish-U6-dmrg-no-chain-verify, 861 s) | 963 s | `python -m n2sqd.trackb --out output/7-other-basins/V1-polish-U6-dmrg-no-chain --start file:output/6-symmetry-breaking/U6-dmrg-no-chain/result.npz --orbitals file:output/6-symmetry-breaking/U6-dmrg-no-chain/orbitals.npy --refine-iterations 0 --oo-cycles 6 --oo-macro 80 --reselect-iterations 2 --max-new 700 --prune-steps 4 --select-cutoff 5e-4 --ci-cutoff 5e-4 --cycle-tol 1e-5` |

Notebooks for the two best results: `notebooks/solution-a.ipynb` (Track A) and `notebooks/solution-b.ipynb` (Track B).
