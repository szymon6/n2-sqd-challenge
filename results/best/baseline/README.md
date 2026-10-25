# Best baseline

`result.npz` — lab SQD loop as given, with the first 100 reference strings pinned (lab Exercise 4 +
reference-augmented loop), on simulated LUCJ samples (Aer MPS, bond dimension 128, 2 000 shots), **seed 1**
(the lowest of seeds 142 / 1 / 2: −108.783919 / −108.784924 / −108.779533).

- Grader energy: **−108.784924 Ha** (`score.json`, grader cross-check agrees)
- Strings per spin: 300 / 300

Reproduce (about 20 min on an M4 laptop: 5 min sampling, 15 min loop):

```bash
python -m n2sqd.baselines --out results/baselines --only lab-sqd-augmented --bond-dim 128 --shots 2000 --seeds 1
python -m eval.score results/baselines/lab-sqd-augmented-bd128-s1.npz
```
