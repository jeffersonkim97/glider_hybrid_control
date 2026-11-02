# Phase-6 G2 Local-SSE Pilot

Terrain `001`, dx `100 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 50.889 s | 14 | 27 | `(6.5, -0.5, 0.0)` | 0.22017439 | 0.09131489 |
| Frozen DQN | 0.339 s | 14 | 27 | `(6.5, -0.5, 0.0)` | 0.22017439 | 0.09131489 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN runtime ratio: `150.219x`
- DQN-selected sensor exact `J_A` gap: `0.00000000`
- Exact-baseline minus DQN-selected exact `J_D`: `0.00000000`
- DQN-selected sensor exact local certificate: `True`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
