# Phase-7 neighborhood sweep G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `10 deg`, Chebyshev `r=4`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 1029.280 s | 131 | 169 | `(4.5, -0.25, 0.0)` | 0.18007672 | 0.02438421 |
| Frozen DQN | 137.673 s | 147 | 449 | `(6.75, 0.0, 0.0)` | 0.23426248 | 0.10531882 |

- Selected action match: `False`
- Selected sensor distance: `226.385 m`
- Bellman/DQN runtime ratio: `7.476x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.00950275`
- Exact-baseline minus DQN-selected exact `J_D`: `-0.06825275`
- DQN-selected sensor exact local certificate: `True`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
