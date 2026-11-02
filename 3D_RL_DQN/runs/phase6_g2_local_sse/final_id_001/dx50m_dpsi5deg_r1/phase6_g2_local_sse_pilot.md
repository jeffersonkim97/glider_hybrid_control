# Phase-6 G2 Local-SSE Pilot

Terrain `001`, dx `50 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 131.597 s | 24 | 120 | `(6.75, 0.25, 0.0)` | 0.21227144 | 0.07911929 |
| Frozen DQN | 0.974 s | 12 | 54 | `(4.75, -0.75, 0.0)` | 0.18795690 | 0.02116356 |

- Selected action match: `False`
- Selected sensor distance: `223.607 m`
- Bellman/DQN runtime ratio: `135.058x`
- DQN-selected sensor exact `J_A` gap: `0.01183269`
- Exact-baseline minus DQN-selected exact `J_D`: `0.06305866`
- DQN-selected sensor exact local certificate: `False`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
