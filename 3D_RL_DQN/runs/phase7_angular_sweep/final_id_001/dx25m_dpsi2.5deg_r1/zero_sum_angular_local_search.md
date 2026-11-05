# Zero-sum angular main effect G2 Local-SSE Pilot

Terrain `001`, dx `25 m`, heading `2.5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 1865.045 s | n/a | 42 | 449 | `(6.75, 0.0, 0.0)` | 0.23247829 | 0.11073121 |
| Frozen DQN | 128.477 s | 36.628 s | 17 | 228 | `(5.0, -1.0, 0.0)` | 0.21186745 | 0.01749631 |

- Selected action match: `False`
- Selected sensor distance: `201.556 m`
- Bellman/DQN first-query ratio: `14.517x`
- Bellman/DQN repeated-query ratio: `50.919x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `18`
- DQN-selected sensor exact `J_A` gap: `0.03565038`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.03565038`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.
