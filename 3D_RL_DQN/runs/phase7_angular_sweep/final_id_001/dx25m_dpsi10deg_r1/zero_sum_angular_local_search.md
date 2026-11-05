# Zero-sum angular main effect G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `10 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 277.384 s | n/a | 42 | 449 | `(6.75, 0.0, 0.0)` | 0.22475973 | 0.09263696 |
| Frozen DQN | 204.120 s | 18.358 s | 42 | 449 | `(6.75, 0.0, 0.0)` | 0.23426248 | 0.10531882 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `1.359x`
- Bellman/DQN repeated-query ratio: `15.110x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `15`
- DQN-selected sensor exact `J_A` gap: `0.00950275`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.00950275`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
