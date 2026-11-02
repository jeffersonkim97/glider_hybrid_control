# Phase-7 angular sweep G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `10 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 177.604 s | 29 | 169 | `(4.5, -0.25, 0.0)` | 0.18007672 | 0.02438421 |
| Frozen DQN | 2.721 s | 9 | 195 | `(4.75, -1.5, 0.0)` | 0.17918301 | 0.01202357 |

- Selected action match: `False`
- Selected sensor distance: `127.475 m`
- Bellman/DQN runtime ratio: `65.267x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.00669578`
- Exact-baseline minus DQN-selected exact `J_D`: `0.01421446`
- DQN-selected sensor exact local certificate: `False`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
