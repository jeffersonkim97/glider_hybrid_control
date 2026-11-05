# Zero-sum G2 Local-SSE Pilot

Terrain `003`, dx `50 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 165.802 s | n/a | 11 | 119 | `(6.75, -0.25, 0.0)` | 0.44297647 | 0.42151190 |
| Frozen DQN | 74.080 s | 3.373 s | 11 | 119 | `(6.75, -0.25, 0.0)` | 0.50285110 | 0.43411232 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `2.238x`
- Bellman/DQN repeated-query ratio: `49.151x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `11`
- DQN-selected sensor exact `J_A` gap: `0.05987463`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.05987463`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
