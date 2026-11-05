# Zero-sum G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 835.657 s | n/a | 42 | 449 | `(6.75, 0.0, 0.0)` | 0.21877923 | 0.09346211 |
| Frozen DQN | 210.765 s | 65.276 s | 33 | 232 | `(5.0, 0.0, 0.0)` | 0.19770409 | 0.00825768 |

- Selected action match: `False`
- Selected sensor distance: `175.000 m`
- Bellman/DQN first-query ratio: `3.965x`
- Bellman/DQN repeated-query ratio: `12.802x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.01891320`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.01891320`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
