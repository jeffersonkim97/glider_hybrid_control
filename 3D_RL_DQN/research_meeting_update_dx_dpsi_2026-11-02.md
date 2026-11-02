# Research Meeting Update: Spatial, Angular, and Neighborhood Discretization

Date: 2026-11-02
Project: Bellman Dynamic Programming vs. Generalized DQN Approximation for Hybrid Attacker–Defender Local SSE

## Purpose of This Update

- Summarize the completed spatial-discretization (`dx`) experiment.
- Summarize the completed angular-discretization (`dpsi`) experiment.
- Summarize the completed Chebyshev-neighborhood-radius (`r`) experiment at `dx=25 m`, `dpsi=5 deg`.
- Separate computational cost, attacker-BR approximation, and defender-decision effects.
- Provide presentation-ready tables and graph-ready numerical data.
- Identify useful visualizations for the research meeting.
- Report measurements and uncertainty without making the final Bellman/DQN method-selection decision.

---

# Recommended Slide Structure

## Slide 1 — Research Goal and Current Position

### Final research question

- Determine how spatial discretization, angular discretization, and defender-neighborhood radius affect:
  - Bellman computation time,
  - DQN online computation time,
  - offline DQN training time,
  - attacker-objective approximation,
  - defender-objective propagation,
  - and the final Local-SSE sensor decision.
- Use these measurements to characterize computational conditions under which Bellman or an RL approximation may be considered.
- The method-selection decision is made by the researcher; the experiment reports the relevant evidence independently.

### Current experimental position

- Completed spatial sweep:

  ```text
  dx = 100, 50, 25, 10 m
  dpsi = 5 deg
  r = 1
  ```

- Completed angular sweep:

  ```text
  dx = 25 m
  dpsi = 10, 5, 2.5 deg
  r = 1
  ```

- Completed neighborhood-radius sweep:

  ```text
  dx = 25 m
  dpsi = 5 deg
  r = 1, 2, 3, 4, 5
  ```

### Recommended visual

- Use the experiment-matrix diagram in the “Combined Experiment Matrix” section.

---

## Slide 2 — Common Experimental Contract

- Held-out unseen random-box Terrain 001.
- Terrain was not used for DQN optimization or checkpoint selection.
- Same terrain, mission, initial defender sensor, objective functions, feasibility rules, and powered-occlusion rules for Bellman and DQN.
- Same deterministic outer Local-SSE search and defender-grid topology.
- Only the attacker best-response backend changes:
  - exact Bellman attacker BR,
  - frozen-DQN approximate attacker BR.
- No test-terrain fine-tuning or retraining.
- Each discretization condition uses a compatible condition-specific checkpoint.

### Timing contract

- `Bellman first solve`
  - Main Local-SSE wall time using exact attacker BRs.
- `DQN first solve`
  - First DQN Local-SSE solve with empty problem caches after scene/checkpoint setup.
- `DQN repeated query`
  - Secondary latency measurement for an identical query with retained caches.
- Excluded from the Bellman-vs-DQN online comparison:
  - offline DQN training,
  - scene/graph construction,
  - checkpoint loading,
  - CUDA initialization,
  - post-hoc exact audit of the DQN-selected sensor.

### Speaker note

- Training time is still reported, but it is not silently added to a single online Local-SSE query.

---

## Slide 3 — Metric Definitions

### Same-sensor attacker-objective error

At the final DQN-selected defender sensor, `d_DQN`:

```text
e_JA = abs(J_A_DQN(d_DQN) - J_A_Bellman(d_DQN))
       / abs(J_A_Bellman(d_DQN)).
```

- Measures attacker-BR approximation at the same sensor.
- Does not by itself measure whether the defender selected the same sensor.
- A 10% line is shown as the current diagnostic reference.

### Signed exact defender-objective gap

```text
g_JD = (J_D_Bellman-final - J_D_exact-at-DQN-sensor)
       / abs(J_D_Bellman-final).
```

- Positive: the exact defender payoff at the DQN-selected sensor is lower.
- Zero: equal exact defender payoff.
- Negative: the DQN-selected sensor has a higher exact defender payoff than the Bellman local-search outcome.
- A negative value does not imply global optimality; it can indicate different local optima.

### Exact Local-SSE certificate

- Recompute the DQN-selected sensor and all required radius-`r` neighbors using exact Bellman attacker responses.
- PASS when no required neighbor has a higher exact defender objective:

  ```text
  J_D_exact(d_DQN)
  >= max(J_D_exact(d) for d in N_r(d_DQN)) - 1e-12.
  ```

- This is a discrete local certificate for the selected radius.

### Selected-action match

- `Yes`: Bellman and DQN terminate at the same defender-grid sensor action.
- `No`: the two searches terminate at different sensors.
- A certificate can pass without an action match if multiple exact local solutions exist.

---

## Slide 4 — Spatial-Discretization Experiment

### Varied condition

```text
dx = dy = dh = 100, 50, 25, 10 m
```

### Fixed conditions

```text
dpsi = 5 deg
r = 1
terrain = final-ID Terrain 001
```

### Results

| dx | Bellman first | DQN first | DQN repeated | Bellman/DQN first | J_A error | signed J_D gap | Exact certificate | Action match |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 100 m | 50.889 s | 65.014 s | 0.339 s | 0.783x | 0.00% | 0.00% | PASS | Yes |
| 50 m | 131.597 s | 51.062 s | 0.974 s | 2.577x | 6.72% | 79.70% | FAIL | No |
| 25 m | 961.422 s | 116.917 s | 19.529 s | 8.223x | 8.23% | 90.42% | FAIL | No |
| 10 m | 5999.106 s | 2022.959 s | 3787.477 s | 2.966x | 4.52% | 42.38% | FAIL | No |

### Main observations

- Bellman Local-SSE time rises sharply as the spatial grid becomes finer.
- DQN online time also rises, but with a different scaling pattern.
- DQN first-solve time is lower than Bellman at 50, 25, and 10 m in this pilot.
- Same-sensor `J_A` error remains below 10% at every tested spatial resolution.
- The defender action and exact local certificate do not follow directly from the same-sensor `J_A` error.
- At 100 m, Bellman and DQN select the same defender action and produce equal reported objectives; Bellman has the lower first-solve time in that condition.

### Recommended figures

- Spatial runtime, log scale.
- Spatial runtime, linear scale.
- Bellman/DQN first-solve ratio with a horizontal `1.0x` line.
- Same-sensor `J_A` error with a 10% reference line.
- Signed exact `J_D` gap.
- Certificate and action-match marker matrix.

---

## Slide 5 — Angular-Discretization Experiment

### Varied condition

```text
dpsi = 10, 5, 2.5 deg
```

### Fixed conditions

```text
dx = 25 m
r = 1
terrain = final-ID Terrain 001
```

### Action-space size

| dpsi | DQN actions |
|---:|---:|
| 10 deg | 36 |
| 5 deg | 72 |
| 2.5 deg | 144 |

### Training and validation measurements

| dpsi | Training time | Validation median J_A error | Validation P90 J_A error | Validation maximum J_A error |
|---:|---:|---:|---:|---:|
| 10 deg | 32.20 min | 3.88% | 8.58% | 10.71% |
| 5 deg | 48.67 min | 6.75% | 35.52% | 94.95% |
| 2.5 deg | 47.60 min | 5.22% | 60.28% | 65.94% |

### Unseen-terrain Local-SSE measurements

| dpsi | Bellman first | DQN first | DQN repeated | Bellman/DQN first | J_A error | signed J_D gap | Exact certificate | Action match |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 10 deg | 177.604 s | 52.302 s | 2.721 s | 3.396x | 3.88% | 58.29% | FAIL | No |
| 5 deg | 961.422 s | 116.917 s | 19.529 s | 8.223x | 8.23% | 90.42% | FAIL | No |
| 2.5 deg | 981.267 s | 214.095 s | 62.943 s | 4.583x | 10.87% | 45.16% | FAIL | No |

### Main observations

- Angular refinement increases the action count from 36 to 72 to 144.
- DQN first-solve time increases from 52.3 to 116.9 to 214.1 seconds.
- Average Bellman BR time per evaluated sensor also rises substantially:
  - approximately 6.1 s at 10 degrees,
  - approximately 21.9 s at 5 degrees,
  - approximately 49.1 s at 2.5 degrees.
- The observed increase in computational cost is not accompanied by a monotonic improvement in the reported `J_A`, `J_D`, certificate, or action-match measurements.
- Total Local-SSE time depends on both per-BR cost and the number of defender sensors visited by the outer search.

### Recommended figures

- Action count versus `dpsi`.
- Training time versus `dpsi`.
- Validation median/P90 versus `dpsi`.
- Bellman and DQN first-solve time versus `dpsi`, log scale.
- Average Bellman BR time per evaluated sensor versus `dpsi`.
- Same-sensor `J_A` error and signed exact `J_D` gap in separate panels.

---

## Slide 6 — Computational Decomposition

### Why total Local-SSE time is not determined by discretization alone

```text
Total Local-SSE time
approximately equals
number of unique defender positions evaluated
times
attacker-BR time per defender position.
```

### Angular-sweep Bellman decomposition

| dpsi | Bellman Local-SSE time | Bellman defender evaluations | Approx. Bellman time/evaluation |
|---:|---:|---:|---:|
| 10 deg | 177.604 s | 29 | 6.12 s |
| 5 deg | 961.422 s | 44 | 21.85 s |
| 2.5 deg | 981.267 s | 20 | 49.06 s |

### Interpretation

- Finer angular discretization makes an individual Bellman BR more expensive.
- The number of defender positions visited is path-dependent.
- At 2.5 degrees, the individual BR cost is highest, but the Bellman search visits fewer defender positions than at 5 degrees.
- Both quantities should be plotted to explain the total runtime correctly.

### Recommended two-panel visualization

- Left panel: unique defender positions evaluated.
- Right panel: average time per evaluated defender position.
- Use matching colors for Bellman and DQN.

---

## Slide 7 — Combined Interpretation

- Spatial refinement strongly increases Bellman computation time.
- Angular refinement strongly increases per-BR computation time and DQN action-space size.
- DQN online computation also depends on discretization, candidate count, observation generation, and the outer search path.
- A low same-sensor attacker-objective error does not automatically imply:
  - the same defender sensor,
  - a small defender-objective difference,
  - or an exact local certificate.
- First-solve and repeated-query DQN timings answer different deployment questions and should remain separate.
- Offline training and online Local-SSE computation should remain separate cost categories.
- The current measurements provide the inputs for the researcher’s final Bellman/DQN selection analysis; no automatic selection is made in this report.

### Recommended visual

- Summary heatmap with rows as computational conditions and columns as:
  - Bellman time,
  - DQN first time,
  - DQN repeated time,
  - `J_A` error,
  - signed `J_D` gap,
  - certificate,
  - action match.

---

## Slide 8 — Chebyshev-Neighborhood-Radius Experiment

### Varied condition

```text
dx = 25 m
dpsi = 5 deg
r = 1, 2, 3, 4, 5
maximum interior neighbors = 8, 24, 48, 80, 120
```

### Results

| r | Max. neighbors | Bellman/DQN evaluations | Bellman first | DQN first | DQN repeated | Bellman/DQN first ratio | J_A error | Signed J_D gap | Certificate | Action match |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | 8 | 44 / 12 | 961.422 s | 116.917 s | 19.529 s | 8.223 | 8.23% | 90.42% | FAIL | no |
| 2 | 24 | 73 / 46 | 1379.353 s | 246.830 s | 64.252 s | 5.588 | 8.69% | 92.26% | FAIL | no |
| 3 | 48 | 102 / 114 | 1637.406 s | 675.706 s | 516.787 s | 2.423 | 9.92% | -314.99% | PASS | no |
| 4 | 80 | 147 / 226 | 2566.386 s | 1441.131 s | 1307.503 s | 1.781 | 9.92% | 0.00% | PASS | yes |
| 5 | 120 | 190 / 190 | 3368.593 s | 1156.165 s | 1076.014 s | 2.913 | 9.92% | 0.00% | PASS | yes |

### Main observations

- Bellman first-solve time increases from 16.02 minutes at `r=1` to 56.14 minutes at `r=5`.
- Runtime is not determined by the maximum neighborhood size alone; the realized outer-search path changes the number of evaluated defender actions.
- This path dependence explains why the measured DQN time at `r=5` is lower than at `r=4`: 190 versus 226 DQN defender evaluations.
- At `r=3`, the selected actions differ, while the DQN-selected sensor independently passes the exact local-neighborhood audit.
- At `r=4` and `r=5`, Bellman and DQN select the same defender action (`449`).
- The negative signed `J_D` gap at `r=3` means that the DQN-selected sensor has a higher exact defender payoff than the Bellman local-search outcome; it is not an absolute-error percentage.

### Recommended figures

- Runtime versus `r`, with Bellman first, DQN first, and DQN repeated shown separately.
- Realized defender evaluations versus `r`, overlaid with the theoretical maximum interior-neighbor count.
- `J_A` error and signed `J_D` gap versus `r`, with certificate and action-match markers.

---

## Slide 9 — Next Measurements

- Repeat representative computational conditions on additional held-out terrains.
- Report distributions rather than only one terrain/search realization.
- Quantify median and P90 runtime and objective differences across the held-out terrain bank.
- Report exact-certificate and action-match frequency across cases.
- Preserve the distinction among discretization cost, realized outer-search path, and approximation error.

---

# Combined Experiment Matrix

The completed spatial and angular sweeps form a one-factor-at-a-time cross, not a full factorial grid.

| Spatial resolution | dpsi=10 deg | dpsi=5 deg | dpsi=2.5 deg |
|---:|:---:|:---:|:---:|
| 100 m | — | Measured | — |
| 50 m | — | Measured | — |
| 25 m | Measured | Measured | Measured |
| 10 m | — | Measured | — |

### Presentation note

- Use this matrix to show exactly which computational conditions have been measured.
- Do not present the existing data as a full `dx × dpsi` factorial experiment.

---

# Graph-Ready Data

## Spatial sweep CSV

```csv
dx_m,dpsi_deg,r_neighbor,bellman_first_sec,dqn_first_sec,dqn_repeated_sec,bellman_over_dqn_first_ratio,JA_error_percent,signed_JD_gap_percent,exact_certificate,action_match
100,5,1,50.889,65.014,0.339,0.783,0.00,0.00,true,true
50,5,1,131.597,51.062,0.974,2.577,6.72,79.70,false,false
25,5,1,961.422,116.917,19.529,8.223,8.23,90.42,false,false
10,5,1,5999.106,2022.959,3787.477,2.966,4.52,42.38,false,false
```

## Angular training/validation CSV

```csv
dx_m,dpsi_deg,r_neighbor,action_count,training_min,validation_median_error_percent,validation_p90_error_percent,validation_max_error_percent
25,10,1,36,32.20,3.88,8.58,10.71
25,5,1,72,48.67,6.75,35.52,94.95
25,2.5,1,144,47.60,5.22,60.28,65.94
```

## Angular Local-SSE CSV

```csv
dx_m,dpsi_deg,r_neighbor,bellman_evaluations,dqn_evaluations,bellman_first_sec,dqn_first_sec,dqn_repeated_sec,bellman_over_dqn_first_ratio,JA_error_percent,signed_JD_gap_percent,exact_certificate,action_match
25,10,1,29,9,177.604,52.302,2.721,3.396,3.88,58.29,false,false
25,5,1,44,12,961.422,116.917,19.529,8.223,8.23,90.42,false,false
25,2.5,1,20,26,981.267,214.095,62.943,4.583,10.87,45.16,false,false
```

## Angular Bellman per-evaluation CSV

```csv
dpsi_deg,bellman_first_sec,bellman_evaluations,approx_bellman_sec_per_evaluation
10,177.604,29,6.124
5,961.422,44,21.850
2.5,981.267,20,49.063
```

## Neighborhood-radius Local-SSE CSV

```csv
dx_m,dpsi_deg,r_neighbor,maximum_interior_neighbors,bellman_evaluations,dqn_evaluations,bellman_first_sec,dqn_first_sec,dqn_repeated_sec,bellman_over_dqn_first_ratio,JA_error_percent,signed_JD_gap_percent,exact_certificate,action_match
25,5,1,8,44,12,961.422,116.917,19.529,8.223,8.23,90.42,false,false
25,5,2,24,73,46,1379.353,246.830,64.252,5.588,8.69,92.26,false,false
25,5,3,48,102,114,1637.406,675.706,516.787,2.423,9.92,-314.99,true,false
25,5,4,80,147,226,2566.386,1441.131,1307.503,1.781,9.92,0.00,true,true
25,5,5,120,190,190,3368.593,1156.165,1076.014,2.913,9.92,0.00,true,true
```

---

# Recommended Additional Visualizations

## 1. Experiment-condition matrix

- Show measured and unmeasured `dx × dpsi` cells.
- Clearly distinguishes completed evidence from future extensions.

## 2. Runtime decomposition

- Panel A: unique defender sensor evaluations.
- Panel B: time per sensor evaluation.
- Panel C: total Local-SSE time.
- Helps separate graph/BR complexity from outer-search path length.

## 3. Defender-search trace

- Plot every evaluated sensor position.
- Overlay Bellman and DQN accepted search paths.
- Mark final Bellman and DQN sensor positions.
- Draw the exact audit neighborhood around the DQN-selected sensor.
- Annotate certificate and action-match results.

## 4. Representative 3D trajectory comparison

- Terrain geometry.
- Start and goal.
- Bellman-selected and DQN-selected sensors.
- Sensor-specific LOS tangent surfaces.
- Powered segments.
- Glide trajectories.
- Switching points.
- Recommended representative cases:
  - `dx=100 m, dpsi=5 deg, r=1`: equal reported result and action match.
  - `dx=25 m, dpsi=5 deg, r=1`: low same-sensor `J_A` error but different defender action.
  - `dx=25 m, dpsi=10 deg, r=1`: angular-sweep reference case.

## 5. Objective propagation figure

- Horizontal axis: same-sensor `J_A` error.
- Vertical axis: signed exact `J_D` gap.
- Marker shape: certificate PASS/FAIL.
- Marker outline: action match yes/no.
- Color: spatial or angular condition.
- Directly visualizes whether follower approximation error predicts leader-level effects.

## 6. Summary evidence heatmap

- Rows: tested computational conditions.
- Columns:
  - Bellman first time,
  - DQN first time,
  - DQN repeated time,
  - training time,
  - `J_A` error,
  - signed `J_D` gap,
  - certificate,
  - action match.
- Use normalized colors but print the numerical value inside every cell.

---

# Existing Figure and Artifact Locations

## Spatial closeout

- Runtime figure:

  `3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_runtime_first_vs_repeated.png`

- Quality figure:

  `3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_quality_error_vs_regret.png`

- Defender search traces:

  `3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_defender_search_traces.png`

- Aggregate report:

  `3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_paper_closeout.md`

## Angular closeout

- Runtime figure:

  `3D_RL_DQN/runs/phase7_angular_sweep/final_id_001/closeout_v1/phase7_angular_runtime.png`

- Quality figure:

  `3D_RL_DQN/runs/phase7_angular_sweep/final_id_001/closeout_v1/phase7_angular_quality.png`

- Training-time figure:

  `3D_RL_DQN/runs/phase7_angular_sweep/final_id_001/closeout_v1/phase7_angular_training_time.png`

- Aggregate report:

  `3D_RL_DQN/runs/phase7_angular_sweep/final_id_001/closeout_v1/phase7_angular_sweep.md`

## Neighborhood-radius closeout at dpsi=5 deg

- Human-readable summary:

  `3D_RL_DQN/runs/phase7_neighbor_sweep_dpsi5deg/final_id_001/closeout_v1/phase7_neighbor_sweep.md`

- Machine-readable aggregate:

  `3D_RL_DQN/runs/phase7_neighbor_sweep_dpsi5deg/final_id_001/closeout_v1/phase7_neighbor_sweep.json`

  `3D_RL_DQN/runs/phase7_neighbor_sweep_dpsi5deg/final_id_001/closeout_v1/phase7_neighbor_sweep.csv`

- Figures:

  `3D_RL_DQN/runs/phase7_neighbor_sweep_dpsi5deg/final_id_001/closeout_v1/phase7_neighbor_runtime.png`

  `3D_RL_DQN/runs/phase7_neighbor_sweep_dpsi5deg/final_id_001/closeout_v1/phase7_neighbor_evaluations.png`

  `3D_RL_DQN/runs/phase7_neighbor_sweep_dpsi5deg/final_id_001/closeout_v1/phase7_neighbor_quality.png`

- Radius-specific raw JSON, exact-response caches, and execution logs are preserved under:

  `3D_RL_DQN/runs/phase7_neighbor_sweep_dpsi5deg/`

## Metric definitions

- Full column definitions and comparison conventions:

  `3D_RL_DQN/phase7_metric_definitions_and_decision_criteria.md`

---

# Brief Current Scope Statement

- The current end-to-end Local-SSE comparison uses one held-out terrain as the controlled pilot case.
- Spatial and angular sweeps are one-factor-at-a-time studies rather than a complete factorial design.
- The exact audit certifies discrete local behavior for the configured neighborhood.
- These scope statements should be presented briefly; the main meeting focus should remain on the measured computational and objective trends.

---

# Raw-Data Provenance to Preserve for the Paper

- Condition:
  - `dx`,
  - `dpsi`,
  - `r`,
  - action count.
- Terrain and dataset:
  - terrain ID/hash,
  - training dataset manifest hash,
  - final-ID split.
- Model:
  - checkpoint path,
  - checkpoint SHA-256,
  - training seed,
  - episode count.
- Computation:
  - Bellman/DQN unique defender evaluations,
  - per-response runtime,
  - first-solve runtime,
  - repeated-query runtime,
  - training runtime.
- Objectives and decisions:
  - same-sensor exact and DQN `J_A`,
  - Bellman-final exact `J_D`,
  - exact `J_D` at the DQN-selected sensor,
  - signed `J_D` gap,
  - exact certificate,
  - action IDs and sensor positions,
  - action match.
- Search provenance:
  - evaluated defender actions,
  - accepted search path,
  - neighbor counts per iteration,
  - termination status.
