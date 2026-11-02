# G2 final-id-002 Results

DQN time is warm steady-state computation time; setup, checkpoint load, CUDA initialization, and cache warm-up are excluded.

| dx | Mission feasible | Policy success | Median error | P90 error | Max error | Bellman | DQN warm | Speedup | Accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 100 m | 3/4 | 3/3 | 11.63% | 17.48% | 17.48% | 2.043 s | 0.117 s | 17.53x | no |
| 50 m | 3/4 | 3/3 | 10.50% | 12.71% | 12.71% | 15.228 s | 0.555 s | 27.44x | no |
| 25 m | 3/4 | 3/3 | 10.98% | 11.28% | 11.28% | 74.221 s | 10.067 s | 7.37x | no |
| 10 m | 3/4 | 3/3 | 8.67% | 13.18% | 13.18% | 424.960 s | 132.609 s | 3.20x | yes |

## Per-case accuracy

| dx | Case | Bellman J_A | DQN J_A | Relative error | Status |
|---:|:---|---:|---:|---:|:---|
| 100 m | Mixed A | 0.171409 | 0.171409 | 0.00% | success |
| 100 m | Mixed B | 0.460690 | 0.541218 | 17.48% | success |
| 100 m | High | 0.162468 | 0.181362 | 11.63% | success |
| 100 m | Low | -- | -- | -- | mission infeasible |
| 50 m | Mixed A | 0.156742 | 0.161311 | 2.92% | success |
| 50 m | Mixed B | 0.227165 | 0.251009 | 10.50% | success |
| 50 m | High | 0.153979 | 0.173557 | 12.71% | success |
| 50 m | Low | -- | -- | -- | mission infeasible |
| 25 m | Mixed A | 0.155827 | 0.158586 | 1.77% | success |
| 25 m | Mixed B | 0.232390 | 0.257896 | 10.98% | success |
| 25 m | High | 0.153148 | 0.170418 | 11.28% | success |
| 25 m | Low | -- | -- | -- | mission infeasible |
| 10 m | Mixed A | 0.154794 | 0.158234 | 2.22% | success |
| 10 m | Mixed B | 0.221332 | 0.250495 | 13.18% | success |
| 10 m | High | 0.152309 | 0.165519 | 8.67% | success |
| 10 m | Low | -- | -- | -- | mission infeasible |
