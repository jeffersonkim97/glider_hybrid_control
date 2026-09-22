# R2 - Discretization sweeps against computation time

Two figures, one per discretization axis, each plotting three measured series on
log-log and again on linear axes. No fitted curve is drawn.

The two axes are **not measured on the same quantity**, because the data on disk
does not support it:

| axis | quantity | why |
|---|---|---|
| Cartesian `dx` | the **whole Defender search** (full local SSE) | six points exist, from the R1 sweep |
| heading `dpsi` | **one best response** at a fixed Defender position | the only heading sweep run; the full SSE heading data is two points at one `dx`, not a sweep |

This matters for reading the numbers: at `dx` = 5 m one best response costs
445.8 s, but the search runs 48 of them and costs 21,221.3 s. The Cartesian
figure shows the second number.

## Conditions

Common to both axes, unchanged from R1:

| | |
|---|---|
| terrain | `centered_cube_half_height` - one box, footprint x,y in [-2, 2], height 2.0, base 0 |
| lattice bounds | x [-8, 8], y [-4, 4], altitude [0, 5] map units (1 map unit = 100 m) |
| start / goal | (-8, 0, 0) / (8, 0, 0) |
| attacker objective | `J_A = 0.5*(H/1.0) + 0.5*(T/221.23893805309734 s)`, minimised |
| defender objective | `J_D = 1 - exp(-H)`, maximised |
| hazard | radar form, rate proportional to 1/R^4, integrated along each edge |
| Q-learning | tabular, cost form, `Q <- Q + a[c + g min Q' - Q]`, gamma = 1 |
| | alpha 0.5, epsilon 1.0 -> 0.05 decaying 0.999 per episode, max_steps 400, seed 0 |
| episodes | 20,000 |
| repetitions | 1 per point, single seed. No variance estimate. |

Per axis:

| | Cartesian | heading |
|---|---|---|
| independent variable | `dx` in {100, 50, 25, 12.5, 10, 5} m | `dpsi` in {10, 5, 2.5, 1.25} deg |
| the other one held at | `dpsi` = 5 deg | `dx` = 12.5 m |
| Defender | Chebyshev search, r = 1, seeded at the lattice point nearest (5, 0) | pinned at (5, 0, 0), no search |
| what is timed | the whole search | one best response |

`dx` = 20 m appears in the fixed-Defender data but was never run as a full SSE,
so it is absent from the Cartesian figure. `dpsi` = 20 deg and 15 deg were
attempted and are infeasible at d = (5, 0, 0): no admissible switching state
survives lattice snapping.

## Series plotted

```
Exact, total           timing.T_total_s        all four exact stages plus search control
Q-learning, training   timing.T_train_s
Q-learning, query      timing.T_search_s  (Cartesian) / rl_s - rl_train_s  (heading)
```

The learned side is split because the two halves behave differently: training
carries nearly all of its growth while query stays comparatively flat. The exact
side is its **total**, not the Bellman sweep alone - the learned side's query
already carries the same LOS and candidate geometry, so charging that fixed cost
to one side and not the other would understate the exact solver.

## Result 1 - Cartesian axis, full local SSE, r = 1

| dx | exact total | LOS | filter | Bellman | powered | ev | it | RL total | train | query | ev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 m | 19.8 | 7.7 | 10.1 | 1.8 | 0.2 | 9 | 1 | 38.1 | 18.5 | 19.6 | 9 |
| 50 m | 28.9 | 7.8 | 10.9 | 8.9 | 1.2 | 9 | 1 | 236.6 | 200.5 | 36.1 | 14 |
| 25 m | 74.8 | 9.9 | 13.9 | 48.5 | 2.5 | 12 | 2 | 581.6 | 513.5 | 68.1 | 12 |
| 12.5 m | 379.9 | 9.5 | 14.5 | 353.4 | 2.3 | 12 | 2 | 1637.5 | 1290.5 | 347.1 | 14 |
| 10 m | 1041.9 | 15.0 | 22.9 | 999.3 | 3.9 | 18 | 4 | 1850.3 | 1394.5 | 455.7 | 14 |
| 5 m | 21221.3 | 44.2 | 69.9 | 21080.0 | 8.6 | 48 | 11 | 5040.6 | 3523.0 | 1517.6 | 14 |

Measured readings:

- The two solvers cross between 10 m and 5 m. At 10 m exact is 0.56x RL; at 5 m it is 4.21x.
- Exact's Bellman share of its own total goes 9.1% -> 99.3%. LOS, filter and powered together
  are 122.8 s at 5 m against 21,080.0 s of Bellman.
- Exact's evaluation count rises with refinement (9, 9, 12, 12, 18, 48); RL's does not
  (9, 14, 12, 14, 14, 14). Exact's iteration count rises likewise (1, 1, 2, 2, 4, 11).
- The 5 m exact point is 21,080.0 s of Bellman across 48 evaluations, which is 439.2 s each;
  the fixed-Defender measurement of one response at 5 m is 442.7 s. The search multiplies the
  single-solve cost by its evaluation count and by nothing else.

### Which run each number comes from

The exact rows are the **backfill** re-solve, not the original R1 run; the RL
rows are the original. The backfill re-ran only the exact side, to recover the
per-position evaluations, stage timings and termination status that the original
`run_comparison` never wrote down. The exact solver is deterministic and each
backfill reproduced its stored objective before anything was written, so the two
runs describe the same computation, but they are different executions on the same
machine and their wall times differ:

| dx | original exact_seconds | backfill T_total_s | difference |
|---|---|---|---|
| 100 m | 21.2 | 19.8 | -6.6% |
| 50 m | 27.8 | 28.9 | +4.0% |
| 25 m | 75.1 | 74.8 | -0.4% |
| 12.5 m | 374.5 | 379.9 | +1.4% |
| 10 m | 985.4 | 1041.9 | +5.7% |
| 5 m | 22984.7 | 21221.3 | -7.7% |

The figure uses the backfill column throughout, so all six exact points come from
one execution. The R1 README's Result 1 table quotes the original column.

## Result 2 - heading axis, one best response at d = (5, 0, 0), dx = 12.5 m

| dpsi | A | N_S | N_S*A | cand | build s | RSS MiB | exact | LOS | filter | Bellman | powered | RL | train | query | acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 deg | 36 | 12,376,260 | 445,545,360 | 2,646 | 1.6 | 451 | 7.11 | 0.85 | 1.24 | 4.89 | 0.12 | 62.35 | 59.81 | 2.53 | 0.9503 |
| 5 deg | 72 | 24,752,520 | 1,782,181,440 | 3,208 | 6.4 | 2075 | 34.00 | 0.84 | 1.25 | 31.55 | 0.33 | 177.73 | 174.76 | 2.97 | 0.9351 |
| 2.5 deg | 144 | 49,505,040 | 7,128,725,760 | 3,492 | 35.5 | 3477 | 99.35 | 0.86 | 1.28 | 96.62 | 0.54 | 226.48 | 223.39 | 3.09 | 0.9431 |
| 1.25 deg | 288 | 99,010,080 | 28,514,903,040 | 4,078 | 137.0 | 5066 | 291.42 | 0.85 | 1.28 | 287.30 | 1.88 | 289.80 | 286.52 | 3.28 | 0.9618 |

`acc` is `attacker_accuracy` = exact `J_A` at (5, 0, 0) divided by the learned
`J_A` at the same position - a fixed-opponent comparison, never across different
Defender positions.

Measured readings:

- Halving `dpsi` doubles `A` and doubles `N_S`, so `N_S*A` quadruples; exact's Bellman time
  rises 4.89 -> 31.55 -> 96.62 -> 287.30, factors of 6.45, 3.06, 2.97.
- LOS and filter are flat (0.84-0.86 s, 1.24-1.28 s) across a 64-fold change in `N_S*A`.
- RL query is flat (2.53-3.28 s); RL training carries all of its growth, but over the same
  range grows by a factor of 4.79 (59.8 -> 286.5) against exact's 58.8 (4.89 -> 287.30).
- The two totals meet at 1.25 deg: exact 291.42 s, RL 289.80 s.
- `attacker_accuracy` stays in [0.9351, 0.9618] and clears `RL_threshold = 0.9` at every step.
  It does not vary monotonically with `dpsi`.
- Peak RSS at 1.25 deg is 5,066 MiB; scene build alone is 137.0 s, which is not counted in
  either solver's time.

## Files

```
data/spatial_sweep_full_sse.json          6-point Cartesian series, extracted from the R1
                                          per-resolution records (see the run-provenance note)
data/heading_sweep_fixed_defender.json    4-point heading series, one process, sequential
data/heading_sweep_run.log                that run's console output
data/spatial_sweep_fixed_defender.json    7-point fixed-Defender Cartesian series, including
                                          20 m; kept for reference, not plotted here
figure/1_cartesian_discretization_vs_time.png         log-log
figure/1_cartesian_discretization_vs_time_linear.png  linear, axes from the origin
figure/2_heading_discretization_vs_time.png           log-log
figure/2_heading_discretization_vs_time_linear.png    linear, axes from the origin
```

Figures are produced by `P1b_sweep_figures.py`. On the linear pair the x ticks are
evenly spaced rather than placed at the measured steps, because 10 and 12.5 collide
on a linear axis; the markers show where the measurements are.

## Not measured

- Variance over seeds. One run per point on both axes.
- The heading axis as a full local SSE. Only two points exist (`dx` = 10 m at 5 deg and
  2.5 deg), which is not a sweep.
- `dx` = 20 m as a full local SSE, so the Cartesian figure has six points and the
  fixed-Defender data has seven.
- `dpsi` = 20 deg and 15 deg, infeasible at d = (5, 0, 0).
- The Defender neighbourhood radius `r` as an axis. Partial data exists (r = 1 at several
  `dx`, r = 5 at 100, 50 and 25 m) but no series at one fixed `dx`, so no figure is drawn.
- Scene build time is excluded from every plotted series; it is recorded on the heading axis
  only (1.6 - 137.0 s).
