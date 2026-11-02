# Phase-7 neighborhood sweep G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `5 deg`, Chebyshev `r=2`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 1379.353 s | 73 | 449 | `(6.75, 0.0, 0.0)` | 0.21877923 | 0.09346211 |
| Frozen DQN | 64.252 s | 46 | 130 | `(4.25, -2.25, 0.0)` | 0.18390380 | 0.01553631 |

- Selected action match: `False`
- Selected sensor distance: `336.341 m`
- Bellman/DQN runtime ratio: `21.468x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.01469778`
- Exact-baseline minus DQN-selected exact `J_D`: `0.08622527`
- DQN-selected sensor exact local certificate: `False`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
