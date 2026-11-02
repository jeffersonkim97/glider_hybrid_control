# G2 final-id-001 Results

DQN time is warm steady-state computation time; setup, checkpoint load, CUDA initialization, and cache warm-up are excluded.

| dx | Mission feasible | Policy success | Median error | P90 error | Max error | Bellman | DQN warm | Speedup | Accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 100 m | 4/4 | 4/4 | 2.15% | 5.23% | 5.23% | 1.532 s | 0.064 s | 23.97x | yes |
| 50 m | 4/4 | 4/4 | 4.39% | 7.99% | 7.99% | 9.762 s | 0.273 s | 35.75x | yes |
| 25 m | 4/4 | 4/4 | 4.16% | 7.81% | 7.81% | 51.260 s | 4.339 s | 11.81x | yes |
| 10 m | 4/4 | 4/4 | 2.23% | 4.41% | 4.41% | 530.163 s | 100.810 s | 5.26x | yes |

## Per-case accuracy

| dx | Case | Bellman J_A | DQN J_A | Relative error | Status |
|---:|:---|---:|---:|---:|:---|
| 100 m | Low | 0.174184 | 0.174184 | 0.00% | success |
| 100 m | Mixed A | 0.175092 | 0.175092 | 0.00% | success |
| 100 m | Mixed B | 0.177034 | 0.184657 | 4.31% | success |
| 100 m | High | 0.187049 | 0.196841 | 5.23% | success |
| 50 m | Low | 0.168287 | 0.171433 | 1.87% | success |
| 50 m | Mixed A | 0.167938 | 0.172245 | 2.56% | success |
| 50 m | Mixed B | 0.171425 | 0.185123 | 7.99% | success |
| 50 m | High | 0.177763 | 0.188812 | 6.22% | success |
| 25 m | Low | 0.166550 | 0.171337 | 2.87% | success |
| 25 m | Mixed A | 0.166286 | 0.170283 | 2.40% | success |
| 25 m | Mixed B | 0.170465 | 0.183772 | 7.81% | success |
| 25 m | High | 0.177816 | 0.187508 | 5.45% | success |
| 10 m | Low | 0.164885 | 0.166926 | 1.24% | success |
| 10 m | Mixed A | 0.165624 | 0.168678 | 1.84% | success |
| 10 m | Mixed B | 0.171306 | 0.178857 | 4.41% | success |
| 10 m | High | 0.177874 | 0.182516 | 2.61% | success |
