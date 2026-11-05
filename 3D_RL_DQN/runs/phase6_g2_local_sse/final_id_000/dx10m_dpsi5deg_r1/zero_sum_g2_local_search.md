# Zero-sum G2 Local-SSE Pilot

Terrain `000`, dx `10 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 7752.825 s | n/a | 50 | 2688 | `(6.75, -0.9499999999999997, 0.0)` | 0.22150761 | 0.09031448 |
| Frozen DQN | 112.018 s | 12.081 s | 12 | 1859 | `(5.65, -0.25, 0.0)` | 0.24577851 | 0.04996946 |

- Selected action match: `False`
- Selected sensor distance: `130.384 m`
- Bellman/DQN first-query ratio: `69.211x`
- Bellman/DQN repeated-query ratio: `641.759x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `16`
- DQN-selected sensor exact `J_A` gap: `0.05398754`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.05398754`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
