# Phase 7 Angular-Discretization Sweep

Controlled comparison at `dx=25 m`, `r=1`, unseen final-ID Terrain 001.
Each angular condition uses its own seed-0, episode-10,000 G2 model.

## Training and validation

| dpsi | actions | training | validation median | validation P90 | frozen-model gate |
|---:|---:|---:|---:|---:|:---:|
| 10 deg | 36 | 32.20 min | 3.88% | 8.58% | PASS |
| 5 deg | 72 | 48.67 min | 6.75% | 35.52% | FAIL |
| 2.5 deg | 144 | 47.60 min | 5.22% | 60.28% | FAIL |

## Unseen-terrain Local-SSE

| dpsi | Bellman first | DQN first | J_A error | J_D regret | exact certificate |
|---:|---:|---:|---:|---:|:---:|
| 10 deg | 177.604 s | 52.302 s | 3.88% | 58.29% | FAIL |
| 5 deg | 961.422 s | 116.917 s | 8.23% | 90.42% | FAIL |
| 2.5 deg | 981.267 s | 214.095 s | 10.87% | 45.16% | FAIL |
