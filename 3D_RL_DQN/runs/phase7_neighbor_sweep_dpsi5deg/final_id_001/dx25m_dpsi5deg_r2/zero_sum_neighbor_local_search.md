# Zero-sum neighborhood main effect G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `5 deg`, Chebyshev `r=2`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 1377.393 s | n/a | 73 | 449 | `(6.75, 0.0, 0.0)` | 0.21877923 | 0.09346211 |
| Frozen DQN | 628.121 s | 846.524 s | 88 | 449 | `(6.75, 0.0, 0.0)` | 0.24047693 | 0.09319379 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `2.193x`
- Bellman/DQN repeated-query ratio: `1.627x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `73`
- DQN-selected sensor exact `J_A` gap: `0.02169770`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.02169770`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
