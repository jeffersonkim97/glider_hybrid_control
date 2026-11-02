# Phase-7 angular sweep G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `2.5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 981.267 s | 20 | 229 | `(5.0, -0.75, 0.0)` | 0.17844013 | 0.02647307 |
| Frozen DQN | 62.943 s | 26 | 351 | `(6.0, -1.25, 0.0)` | 0.19804943 | 0.04193327 |

- Selected action match: `False`
- Selected sensor distance: `111.803 m`
- Bellman/DQN runtime ratio: `15.590x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.01941283`
- Exact-baseline minus DQN-selected exact `J_D`: `0.01195621`
- DQN-selected sensor exact local certificate: `False`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
