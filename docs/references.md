# References (reading list)

Source · title · why it matters · where the project used it (phase or stage).

## SQD method and variants
- arXiv:2405.05068 · Robledo-Moreno et al., *Chemistry beyond exact solutions on a quantum-centric supercomputer* · the SQD method: configuration recovery, batching, orbital optimisation (Supp. 2.1/4.3: gradient of the fixed-amplitude subspace energy w.r.t. the Thouless rotation, biggest gains at stretched geometries); N₂ (10e, 26o) cc-pVDZ is their benchmark · phase 3
- arXiv:2411.00468 · Barison et al., *Ext-SQD* (excited states, singles+doubles augmentation of SQD subspaces) · augmenting sampled strings with excitations halves the subspace for the same accuracy on N₂ (10e, 26o) · phase 3
- arXiv:2302.11320 · Kanno et al., *QSCI* · the same idea as SQD, variational by construction · phase 3
- arXiv:2501.09702, arXiv:2508.02578 · Yu et al.; Kirby et al., *sample-based Krylov QD / SqDRIFT* · time-evolved samplers; HF orbitals far more compact than localised ones; S/D augmentation worth tens of mHa · phase 3
- arXiv:2412.07218, arXiv:2412.13839, arXiv:2509.02525 · HSB-QSCI / TE-QSCI / TE-QSCI+PT2 · time-evolution samplers; occupancy-guided S/D proposals; HCI still most compact · phase 3
- arXiv:2603.13536 · *Active-sampling SQD* · Epstein–Nesbet PT2 acquisition score to add H-connected configurations (CIPSI seeded by samples) · phase 3, stage 1 design
- arXiv:2603.09346 · *Cluster-specific SQD* · recovery with cluster-specific occupancies; stretched N₂ (10e, 26o) gains 1–16 mHa over SQD at d = 2000 · phase 3
- arXiv:2605.23697 · noise and recovery study on N₂ (10e, 26o) · random strings + HF + recovery match QPU samples by iteration 2: recovery, not circuit quality, carries SQD · phase 3 (Track A design)
- arXiv:2608.11569 · benchmark hygiene for SQD · fix the diagonalisation dimension when comparing; uniform random reproduces inflated SQD claims · phase 3
- arXiv:2602.01165 · HCI-inspired sample selection (HCI-HSQD) · sub-mHa on N₂ (10e, 26o) with subspaces 39 % smaller than HCI · phase 3
- arXiv:2609.01264 · DF-SQD (double-factorised) · N₂ 6-31G 45× more accurate at 11 % smaller subspace · phase 3
- arXiv:2608.05314 · ML for SQD review · stretched N₂ CAS(10, 12): 99.9 % of the weight in ≈190 of 792 α-strings; α|β Schmidt rank bounds the strings any spin-collinear basis needs · phase 3 (how many strings 300 can be worth)
- arXiv:2503.05967 · SQD + AFQMC · not usable: the grader recomputes plain SCI · phase 3
- arXiv:2601.16169, arXiv:2601.16637 · GPU Davidson for SQD · irrelevant at 90 k determinants · phase 3

- arXiv:2505.00914 · Nogaki et al., *Symmetry-adapted sample-based quantum diagonalization* · embedding space-group symmetry improves the compactness of (free) sampled subspaces; the counter-argument to breaking symmetry in a product space · stage 6
- PySCF `fci.selected_ci.kernel_fixed_space` (2.14.0) · uses `ci0` only when it is an `SCIvector` · the warm-start bug of stages 1–5 · stage 6

## Critiques
- arXiv:2501.07231 · Reinholdt et al., *Critical limitations of SQD/QSCI* · with ideal sampling N₂ (10e, 22o) needs 1e8–1e9 samples for mHa accuracy; at equal determinant count QSCI ≈ HCI; notes PySCF `kernel_fixed_space` is a product space (quadratically more determinants than individually selected ones) · phase 3
- arXiv:2605.02494 · Gaberle & Jattana, *critical assessment for Heisenberg/Hubbard* · configurations for fixed accuracy grow exponentially even with optimal ordering · phase 3
- arXiv:2603.03496 · Kirby et al. · prior SQD demos "have not surpassed SCI"; SKQD wins only on a Hamiltonian built to defeat perturbative selection · phase 3

## Orbital bases and compact CI
- arXiv:2104.02587 · SHCI orbital optimisation · natural orbitals then Newton-optimised orbitals give much more compact variational wavefunctions · phase 3, stage 1
- arXiv:2308.08056 · WAHTOR orbitals = natural orbitals; maximally sparse mutual information · phase 3
- Bytautas, Ivanic, Ruedenberg, J. Chem. Phys. 119, 8217 (2003) · split-localized orbitals converge CI faster than natural orbitals for diatomics (own knowledge of the agent; not re-read) · phase 3
- arXiv:2605.28040 · filter-assisted SQD (unitary filter concentrating ground-state weight) · phase 3

## Reference energies
- arXiv:0705.2827 (Table II) · DMRG N₂ cc-pVDZ all-electron (Chan, Kállay, Gauss 2004 values): 3.0 bohr −109.089380, 3.6 bohr −108.998052, 4.2 bohr −108.970090 Ha · sanity check of our DMRG (frozen core, 3.78 bohr) · phase 2/3
- arXiv:1902.04592 (Table 5) · CDFCI N₂ cc-pVDZ: 3.6 bohr −108.998083, 4.2 bohr −108.970132 Ha · same · phase 3
- https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/tree/main/data/variational-problems · N₂ cc-pVDZ (14e, 28o) FCIDUMPs 0.8–3.0 Å incl. 2.00 Å; HCI −108.9614962 at 3.0 Å · phase 3

## Tools and prior results on this lab
- https://github.com/qiskit-community/qgss-2026/tree/main/lab-4 · official QGSS 2026 Lab 4c notebook (the Warsaw Lab 4b) · baseline definition · phase 1/2
- https://github.com/wolfomania/qfest-grader · `qfest-grader` source (lab4b.py, tests) · grader rules; rotated integrals accepted · phase 1/2
- https://github.com/qiskit-community/Quantum-Challenge-Grader/blob/main/qc_grader/challenges/qgss_2026/lab4c.py · QGSS grader client: scoring tiers live on the server, not public · phase 3
- https://github.com/afrah/qgss-2026 · solutions fork: reference-augmented SQD −108.769682 scored 0 ("too high, beat classical brute force"); plain SQD −108.609120; own reference subspace −108.798250 · phase 3 (what others reached)
- https://github.com/Iqumnov/qgss-2026 · claims −108.805 Ha with score 100 (FakeMiami MPS samples, 100 pinned strings); outputs cleared, unverifiable · phase 3
- https://github.com/Ambaaq-Ajibike/QGSS26/blob/main/QGSS_Lab4.ipynb · bonus score 38, no energy shown · phase 3
- https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization · IBM SQD tutorial, N₂ cc-pVDZ 1.0 Å, max_dim 1000, 1e5 shots: SQD 31.9 mHa above SCI reference · phase 3
- https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-sqd/guides/optimize-orbitals · orbital-optimisation guide (N₂ 6-31g): −108.9847 → −109.0118 after 10 iterations, exact −109.0467; `optimize_orbitals` deprecated for `ffsim.optimize_orbitals` · phase 3
- https://github.com/Qiskit/qiskit-addon-sqd/tree/main/docs/guides · `TrimPolicy` (0.14) guide: N₂ 6-31g, merged 148 × 148 subspace −109.034369 vs exact −109.046672 · phase 3
- https://github.com/Qiskit/qiskit-aer (issues #2056, #2246, #2286, #2380, #1305) · MPS memory estimate assumes worst-case bond dimension; `max_memory_mb` bypass; bond cap is a process-wide static · phase 2
- https://pypi.org/project/ffsim/ (0.0.84) · `UCJOpSpinBalanced.from_t_amplitudes` options; `optimize=True` fits the t2 factorisation, not the energy; no MPS backend · phase 2/3
