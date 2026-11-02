# Phase 7 Chebyshev-Neighborhood Sweep

Controlled comparison at `dx=25 m`, `dpsi=5 deg`, unseen final-ID Terrain 001.
The same frozen seed-0, episode-10,000 G2 checkpoint is used for every radius.

Negative signed J_D gap means the DQN-selected sensor has a higher exact defender payoff than the Bellman local-search outcome.

| r | max neighbors | Bellman evals | DQN evals | Bellman first | DQN first | DQN faster | J_A error | signed J_D gap | certificate | action match |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|
| 1 | 8 | 44 | 12 | 961.422 s | 116.917 s | yes | 8.23% | 90.42% | FAIL | no |
| 2 | 24 | 73 | 46 | 1379.353 s | 246.830 s | yes | 8.69% | 92.26% | FAIL | no |
| 3 | 48 | 102 | 114 | 1637.406 s | 675.706 s | yes | 9.92% | -314.99% | PASS | no |
| 4 | 80 | 147 | 226 | 2566.386 s | 1441.131 s | yes | 9.92% | 0.00% | PASS | yes |
| 5 | 120 | 190 | 190 | 3368.593 s | 1156.165 s | yes | 9.92% | 0.00% | PASS | yes |
