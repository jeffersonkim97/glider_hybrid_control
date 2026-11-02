# G2 final-id-003 Results

DQN time is warm steady-state computation time; setup, checkpoint load, CUDA initialization, and cache warm-up are excluded.

| dx | Mission feasible | Policy success | Median error | P90 error | Max error | Bellman | DQN warm | Speedup | Accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 100 m | 4/4 | 4/4 | 4.71% | 11.15% | 11.15% | 4.208 s | 0.143 s | 29.51x | yes |
| 50 m | 4/4 | 4/4 | 15.36% | 25.10% | 25.10% | 23.954 s | 0.643 s | 37.26x | no |
| 25 m | 4/4 | 4/4 | 13.71% | 17.37% | 17.37% | 134.849 s | 12.640 s | 10.67x | no |
| 10 m | 4/4 | 4/4 | 13.66% | 16.36% | 16.36% | 853.014 s | 157.897 s | 5.40x | no |

## Per-case accuracy

| dx | Case | Bellman J_A | DQN J_A | Relative error | Status |
|---:|:---|---:|---:|---:|:---|
| 100 m | Low | 0.184641 | 0.184641 | 0.00% | success |
| 100 m | Mixed A | 0.181234 | 0.201446 | 11.15% | success |
| 100 m | Mixed B | 0.373550 | 0.373550 | 0.00% | success |
| 100 m | High | 0.189466 | 0.207322 | 9.42% | success |
| 50 m | Low | 0.173346 | 0.201620 | 16.31% | success |
| 50 m | Mixed A | 0.169924 | 0.212582 | 25.10% | success |
| 50 m | Mixed B | 0.218560 | 0.236293 | 8.11% | success |
| 50 m | High | 0.172957 | 0.197877 | 14.41% | success |
| 25 m | Low | 0.170412 | 0.187989 | 10.31% | success |
| 25 m | Mixed A | 0.168684 | 0.197983 | 17.37% | success |
| 25 m | Mixed B | 0.216200 | 0.253181 | 17.10% | success |
| 25 m | High | 0.169128 | 0.176698 | 4.48% | success |
| 10 m | Low | 0.172941 | 0.196149 | 13.42% | success |
| 10 m | Mixed A | 0.177364 | 0.202034 | 13.91% | success |
| 10 m | Mixed B | 0.204617 | 0.238094 | 16.36% | success |
| 10 m | High | 0.164929 | 0.175719 | 6.54% | success |
