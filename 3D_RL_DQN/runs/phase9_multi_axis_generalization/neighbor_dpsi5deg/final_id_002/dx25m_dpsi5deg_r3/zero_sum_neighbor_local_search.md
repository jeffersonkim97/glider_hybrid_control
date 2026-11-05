# Zero-sum neighborhood main effect G2 Local-SSE Pilot

Terrain `002`, dx `25 m`, heading `5 deg`, Chebyshev `r=3`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 970.755 s | n/a | 46 | 449 | `(6.75, 0.0, 0.0)` | 0.48199857 | 0.45969527 |
| Frozen DQN | 315.304 s | 180.926 s | 46 | 449 | `(6.75, 0.0, 0.0)` | 0.55380491 | 0.49969924 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `3.079x`
- Bellman/DQN repeated-query ratio: `5.365x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.07180634`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.07180634`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
