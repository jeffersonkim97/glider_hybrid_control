# Phase-6 G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 961.422 s | 44 | 449 | `(6.75, 0.0, 0.0)` | 0.21877923 | 0.09346211 |
| Frozen DQN | 19.529 s | 12 | 164 | `(4.5, -1.5, 0.0)` | 0.18380122 | 0.01191783 |

- Selected action match: `False`
- Selected sensor distance: `270.416 m`
- Bellman/DQN runtime ratio: `49.230x`
- DQN-selected sensor exact `J_A` gap: `0.01397560`
- Exact-baseline minus DQN-selected exact `J_D`: `0.08451190`
- DQN-selected sensor exact local certificate: `False`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
