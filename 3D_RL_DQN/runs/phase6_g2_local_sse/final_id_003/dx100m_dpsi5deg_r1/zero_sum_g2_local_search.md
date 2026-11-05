# Zero-sum G2 Local-SSE Pilot

Terrain `003`, dx `100 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 31.559 s | n/a | 8 | 27 | `(6.5, -0.5, 0.0)` | 0.58686241 | 0.56105539 |
| Frozen DQN | 41.055 s | 0.414 s | 8 | 27 | `(6.5, -0.5, 0.0)` | 0.59868399 | 0.56338936 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `0.769x`
- Bellman/DQN repeated-query ratio: `76.165x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `6`
- DQN-selected sensor exact `J_A` gap: `0.01182157`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.01182157`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
