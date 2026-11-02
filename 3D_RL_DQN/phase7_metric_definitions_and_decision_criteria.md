# Phase 7 Metric Definitions and Decision Criteria

Recorded on 2026-11-01 for the Bellman-versus-generalized-DQN Local-SSE study.

## Controlled Local-SSE experiment

The defender is the leader and selects a sensor-grid action. For each evaluated
sensor position, Bellman or the frozen DQN supplies the attacker best response
(BR). The same deterministic outer Local-SSE search, initial sensor, defender
grid, objective evaluator, feasibility rules, and Chebyshev neighborhood are
used. Only the attacker-BR backend changes.

## Table-column definitions

### `r`

`r` is the integer Chebyshev radius on the defender sensor grid. At defender
grid spacing `dx`, radius `r` covers index offsets satisfying

```text
0 < max(abs(delta_i), abs(delta_j)) <= r.
```

For `dx=25 m`, the maximum x/y displacement is `25*r` metres.

### Maximum neighbors

At an interior grid point, the maximum number of neighboring defender actions
is

```text
N_r = (2*r + 1)^2 - 1.
```

Map boundaries and invalid sensor positions can reduce the actual count.

### Bellman evaluations

The number of unique defender sensor positions for which the main exact
Local-SSE search computed an exact Bellman attacker BR. It does not count
post-hoc DQN-neighbor certification calls. The count can exceed `N_r` because
the search may move to a new sensor and evaluate another neighborhood.

### DQN evaluations

The number of unique defender sensor positions for which the approximate
Local-SSE search generated and authoritatively rescored a frozen-DQN attacker
BR. Bellman and DQN counts can differ because approximate payoffs can produce
different search paths and stopping points.

### Bellman first-solve time

Wall time for the main Local-SSE solve using exact Bellman attacker responses.
It includes all BR calls made by the main search. It excludes shared scene and
graph construction and excludes the post-hoc exact audit of the DQN-selected
sensor. For a resumable run, the equivalent first-solve time is reconstructed
from the persisted per-response runtimes.

### DQN first-solve time

Wall time for the first DQN Local-SSE solve with empty scenario, observation,
and structural action-row caches after scene construction and checkpoint load.
It includes observation construction, feasible-action construction, network
inference, trajectory rollout, authoritative scoring, and the outer search. It
excludes offline training, checkpoint loading, scene construction, CUDA-context
initialization, and the post-hoc exact Bellman audit.

### DQN repeated-query time

A secondary latency diagnostic for an identical Local-SSE query with retained
problem caches. It is not the primary deployment timing and must not replace
the first-solve time in the Bellman-versus-DQN break-even comparison.

### Same-sensor attacker-objective error

At the final sensor selected by DQN, `d_DQN`, the reported follower error is

```text
e_JA = abs(J_A_DQN(d_DQN) - J_A_Bellman(d_DQN))
       / abs(J_A_Bellman(d_DQN)).
```

This measures attacker-BR approximation at the same sensor. It does not certify
the defender decision. The current per-case diagnostic threshold is 10%.

### Signed exact defender-objective gap

The signed comparison between the Bellman-search outcome and the exact response
at the DQN-selected sensor is

```text
g_JD = (J_D_Bellman-final - J_D_exact-at-DQN-sensor)
       / abs(J_D_Bellman-final).
```

A positive value means that the exact defender payoff at the DQN-selected
sensor is lower. Zero means equal payoff. A negative value means that the
DQN-selected sensor has a higher exact defender payoff than the Bellman local
search outcome. A negative value does not establish global optimality because
both points can be different local optima.

### Exact Local-SSE certificate

The DQN-selected sensor is recomputed with exact Bellman responses, as are all
required radius-`r` neighbors. The certificate passes when the selected sensor
is feasible and

```text
J_D_exact(d_DQN) >= max(J_D_exact(d) for d in N_r(d_DQN)) - 1e-12.
```

This is a discrete radius-`r` local-optimality certificate. It is not a global
or continuous-space optimality certificate. The audit is a scientific
validation step and its Bellman cost is excluded from DQN online timing.

### Selected-action match

`yes` means that Bellman and DQN Local-SSE runs terminate at the same defender
sensor-grid action. `no` with a passing exact certificate is possible when the
finite game contains multiple exact local optima. An action match is stronger
evidence than local certification alone, but is not required for an alternative
exact local solution to be valid.

## Model-level quality reference

The following held-out validation thresholds are reported as a common quality
reference for frozen DQN models:

```text
success rate on feasible missions = 100%
median Bellman-relative J_A error <= 10%
P90 Bellman-relative J_A error, method=higher <= 30%
maximum error = report only
```

## Per-condition reported comparison flags

The experiment reports the following facts independently:

```text
held-out model-quality statistics
same-sensor J_A error and its 10% reference threshold
exact Local-SSE certificate result
DQN first-solve time < Bellman first-solve time
selected defender action match
```

These fields are evidence for the user's method-selection decision. The report
does not automatically classify a condition as requiring Bellman or DQN and
does not replace the user's judgment with a hard adoption label.

## Current controlled sweeps

| Sweep | Varied quantity | Fixed quantities | Model use |
|:---|:---|:---|:---|
| Spatial | `dx=100/50/25/10 m` | `dpsi=5 deg`, `r=1` | condition-specific models |
| Angular | `dpsi=10/5/2.5 deg` | `dx=25 m`, `r=1` | condition-specific models |
| Neighborhood A | `r=1..5` | `dx=25 m`, `dpsi=10 deg` | same frozen 10-degree model |
| Neighborhood B | `r=1..5` | `dx=25 m`, `dpsi=5 deg` | same frozen 5-degree model |

The 10-degree neighborhood sweep provides one controlled data slice. The
5-degree neighborhood sweep supplies direct comparability with the existing
spatial sweep. Both retain their validation statistics and raw results so the
user can decide how each slice should be interpreted in the paper.
