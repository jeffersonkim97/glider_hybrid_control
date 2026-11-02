# Phase 7 Chebyshev-Neighborhood Sweep

Controlled comparison at `dx=25 m`, `dpsi=10 deg`, unseen final-ID Terrain 001.
The same frozen seed-0, episode-10,000 G2 checkpoint is used for every radius.

Negative signed J_D gap means the DQN-selected sensor has a higher exact defender payoff than the Bellman local-search outcome.

| r | max neighbors | Bellman evals | DQN evals | Bellman first | DQN first | DQN faster | J_A error | signed J_D gap | certificate | action match |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|
| 1 | 8 | 29 | 9 | 177.604 s | 52.302 s | yes | 3.88% | 58.29% | FAIL | no |
| 2 | 24 | 53 | 97 | 347.631 s | 401.687 s | no | 4.23% | -279.91% | PASS | no |
| 3 | 48 | 88 | 128 | 542.570 s | 485.677 s | yes | 4.82% | 56.43% | FAIL | no |
| 4 | 80 | 131 | 147 | 1029.280 s | 803.918 s | yes | 4.23% | -279.91% | PASS | no |
| 5 | 120 | 237 | 190 | 1448.504 s | 786.613 s | yes | 4.23% | 0.00% | PASS | yes |
