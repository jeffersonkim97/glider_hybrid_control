# G2 Final-ID Aggregate: Terrains 000-003

The bank contains four sealed unseen random-box terrains and sixteen generated sensor scenarios per resolution. DQN time is warm steady-state time; setup, checkpoint load, CUDA initialization, and cache warm-up are excluded.

| dx | Feasible | Policy success | Median error | P90 error | Max error | Candidates | Bellman total | DQN warm total | Speedup | Terrain passes | Bank pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 100 m | 14/16 | 14/14 | 4.77% | 17.48% | 23.35% | 1938 | 9.044 s | 0.376 s | 24.07x | 2/4 | yes |
| 50 m | 14/16 | 14/14 | 9.64% | 25.10% | 38.83% | 4785 | 55.526 s | 1.614 s | 34.41x | 1/4 | yes |
| 25 m | 14/16 | 14/14 | 8.00% | 17.37% | 38.75% | 12047 | 291.188 s | 27.378 s | 10.64x | 1/4 | yes |
| 10 m | 14/16 | 14/14 | 7.61% | 25.18% | 44.26% | 28149 | 2318.193 s | 396.302 s | 5.85x | 2/4 | yes |

## Per-terrain diagnostic

| Terrain | dx | Feasible | Success | Median error | P90 error | Candidates | Bellman | DQN warm | Pass |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 000 | 100 m | 3/4 | 3/3 | 12.15% | 23.35% | 157 | 1.261 s | 0.053 s | no |
| 000 | 50 m | 3/4 | 3/3 | 10.03% | 38.83% | 225 | 6.583 s | 0.143 s | no |
| 000 | 25 m | 3/4 | 3/3 | 8.19% | 38.75% | 354 | 30.859 s | 0.332 s | no |
| 000 | 10 m | 3/4 | 3/3 | 25.18% | 44.26% | 408 | 510.057 s | 4.986 s | no |
| 001 | 100 m | 4/4 | 4/4 | 2.15% | 5.23% | 323 | 1.532 s | 0.064 s | yes |
| 001 | 50 m | 4/4 | 4/4 | 4.39% | 7.99% | 796 | 9.762 s | 0.273 s | yes |
| 001 | 25 m | 4/4 | 4/4 | 4.16% | 7.81% | 2209 | 51.260 s | 4.339 s | yes |
| 001 | 10 m | 4/4 | 4/4 | 2.23% | 4.41% | 6705 | 530.163 s | 100.810 s | yes |
| 002 | 100 m | 3/4 | 3/3 | 11.63% | 17.48% | 731 | 2.043 s | 0.117 s | no |
| 002 | 50 m | 3/4 | 3/3 | 10.50% | 12.71% | 1990 | 15.228 s | 0.555 s | no |
| 002 | 25 m | 3/4 | 3/3 | 10.98% | 11.28% | 5235 | 74.221 s | 10.067 s | no |
| 002 | 10 m | 3/4 | 3/3 | 8.67% | 13.18% | 12597 | 424.960 s | 132.609 s | yes |
| 003 | 100 m | 4/4 | 4/4 | 4.71% | 11.15% | 727 | 4.208 s | 0.143 s | yes |
| 003 | 50 m | 4/4 | 4/4 | 15.36% | 25.10% | 1774 | 23.954 s | 0.643 s | no |
| 003 | 25 m | 4/4 | 4/4 | 13.71% | 17.37% | 4249 | 134.849 s | 12.640 s | no |
| 003 | 10 m | 4/4 | 4/4 | 13.66% | 16.36% | 8439 | 853.014 s | 157.897 s | no |
