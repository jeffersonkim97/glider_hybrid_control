# Phase-7 neighborhood sweep G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `5 deg`, Chebyshev `r=3`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 1637.406 s | 102 | 232 | `(5.0, 0.0, 0.0)` | 0.17879089 | 0.02252130 |
| Frozen DQN | 516.787 s | 114 | 449 | `(6.75, 0.0, 0.0)` | 0.24047693 | 0.09319379 |

- Selected action match: `False`
- Selected sensor distance: `175.000 m`
- Bellman/DQN runtime ratio: `3.168x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.02169770`
- Exact-baseline minus DQN-selected exact `J_D`: `-0.07094081`
- DQN-selected sensor exact local certificate: `True`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
