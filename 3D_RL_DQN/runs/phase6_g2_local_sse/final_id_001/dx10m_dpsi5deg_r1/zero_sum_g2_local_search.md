# Zero-sum G2 Local-SSE Pilot

Terrain `001`, dx `10 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 14503.425 s | n/a | 94 | 2697 | `(6.75, -0.04999999999999982, 0.0)` | 0.21888489 | 0.08690560 |
| Frozen DQN | 1689.922 s | 2832.838 s | 17 | 1089 | `(4.65, -1.25, 0.0)` | 0.18135940 | 0.00933023 |

- Selected action match: `False`
- Selected sensor distance: `241.868 m`
- Bellman/DQN first-query ratio: `8.582x`
- Bellman/DQN repeated-query ratio: `5.120x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `51`
- DQN-selected sensor exact `J_A` gap: `0.00875586`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.00875586`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
