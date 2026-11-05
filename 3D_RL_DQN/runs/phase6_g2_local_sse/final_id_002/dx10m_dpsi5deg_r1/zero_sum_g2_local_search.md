# Zero-sum G2 Local-SSE Pilot

Terrain `002`, dx `10 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 3378.477 s | n/a | 23 | 2698 | `(6.75, 0.050000000000000266, 0.0)` | 0.45186861 | 0.42843329 |
| Frozen DQN | 2352.207 s | 4733.295 s | 28 | 2697 | `(6.75, -0.04999999999999982, 0.0)` | 0.56467611 | 0.52744697 |

- Selected action match: `False`
- Selected sensor distance: `10.000 m`
- Bellman/DQN first-query ratio: `1.436x`
- Bellman/DQN repeated-query ratio: `0.714x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `23`
- DQN-selected sensor exact `J_A` gap: `0.11415028`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.11415028`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
