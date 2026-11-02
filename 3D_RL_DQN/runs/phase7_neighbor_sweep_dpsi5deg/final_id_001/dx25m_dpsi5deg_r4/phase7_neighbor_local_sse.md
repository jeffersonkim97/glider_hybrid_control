# Phase-7 neighborhood sweep G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `5 deg`, Chebyshev `r=4`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 2566.386 s | 147 | 449 | `(6.75, 0.0, 0.0)` | 0.21877923 | 0.09346211 |
| Frozen DQN | 1307.503 s | 226 | 449 | `(6.75, 0.0, 0.0)` | 0.24047693 | 0.09319379 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN runtime ratio: `1.963x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.02169770`
- Exact-baseline minus DQN-selected exact `J_D`: `0.00000000`
- DQN-selected sensor exact local certificate: `True`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
