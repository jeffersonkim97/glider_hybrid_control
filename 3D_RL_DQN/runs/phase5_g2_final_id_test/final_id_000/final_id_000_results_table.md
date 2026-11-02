# G2 Final-ID Terrain 000 Results

DQN time is warm steady-state computation time; setup, checkpoint load, CUDA initialization, and cache warm-up are excluded.

| dx | Mission feasible | Policy success | Median error | P90 error | Max error | Bellman | DQN warm | Speedup | Accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 100 m | 3/4 | 3/3 | 12.15% | 23.35% | 23.35% | 1.261 s | 0.053 s | 23.97x | no |
| 50 m | 3/4 | 3/3 | 10.03% | 38.83% | 38.83% | 6.583 s | 0.143 s | 46.12x | no |
| 25 m | 3/4 | 3/3 | 8.19% | 38.75% | 38.75% | 30.859 s | 0.332 s | 92.94x | no |
| 10 m | 3/4 | 3/3 | 25.18% | 44.26% | 44.26% | 510.057 s | 4.986 s | 102.30x | no |

## Per-case accuracy

| dx | Case | Bellman J_A | DQN J_A | Relative error | Status |
|---:|:---|---:|---:|---:|:---|
| 100 m | Low | 0.185648 | 0.187187 | 0.83% | success |
| 100 m | Mixed A | 0.186385 | 0.229912 | 23.35% | success |
| 100 m | Mixed B | 0.201222 | 0.225667 | 12.15% | success |
| 100 m | High | -- | -- | -- | mission infeasible |
| 50 m | Low | 0.178664 | 0.196584 | 10.03% | success |
| 50 m | Mixed A | 0.178749 | 0.248162 | 38.83% | success |
| 50 m | Mixed B | 0.191454 | 0.209175 | 9.26% | success |
| 50 m | High | -- | -- | -- | mission infeasible |
| 25 m | Low | 0.175320 | 0.185758 | 5.95% | success |
| 25 m | Mixed A | 0.179126 | 0.248529 | 38.75% | success |
| 25 m | Mixed B | 0.189162 | 0.204655 | 8.19% | success |
| 25 m | High | -- | -- | -- | mission infeasible |
| 10 m | Low | 0.173201 | 0.178121 | 2.84% | success |
| 10 m | Mixed A | 0.176019 | 0.253917 | 44.26% | success |
| 10 m | Mixed B | 0.191677 | 0.239945 | 25.18% | success |
| 10 m | High | -- | -- | -- | mission infeasible |
