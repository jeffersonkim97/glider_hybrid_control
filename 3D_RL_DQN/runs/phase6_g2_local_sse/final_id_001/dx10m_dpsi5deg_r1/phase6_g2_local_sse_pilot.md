# Phase-6 G2 Local-SSE Pilot

Terrain `001`, dx `10 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 5999.106 s | 36 | 1622 | `(5.35, -1.15, 0.0)` | 0.17491735 | 0.01345620 |
| Frozen DQN | 3787.477 s | 19 | 1013 | `(4.55, -1.25, 0.0)` | 0.17952511 | 0.01058335 |

- Selected action match: `False`
- Selected sensor distance: `80.623 m`
- Bellman/DQN runtime ratio: `1.584x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.00776490`
- Exact-baseline minus DQN-selected exact `J_D`: `0.00570313`
- DQN-selected sensor exact local certificate: `False`

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings. Because the implemented game is non-zero-sum, the single-point J_A gap is diagnostic only; the exact selected-plus-neighbor audit supplies the local Stackelberg certificate.
