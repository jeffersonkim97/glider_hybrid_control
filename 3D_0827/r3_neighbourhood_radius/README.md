# R3 - Defender neighbourhood radius against computation time

The third axis. Unlike the spatial and heading steps, `r` is a property of the
Defender search and cannot be measured at a fixed Defender position - there is no
neighbourhood without a search. Every point here is therefore a **complete local
SSE**, both solvers, and the y axis is the whole search rather than one solve.

## Why dx = 50 m

The Defender grid is finite, so `r` saturates:

| dx | N_D | r that reaches the grid edge |
|---|---|---|
| 100 m | 49 | ~4 |
| 50 m | 178 | ~7 |
| 25 m | 676 | ~14 |
| 12.5 m | 2,632 | ~26 |

At 100 m, `r` = 5 already evaluates all 49 positions, so larger `r` repeats the
same answer and the axis is dead. At 25 m the axis has headroom but the learned
side costs 77.8 s per Defender evaluation, putting `r` = 8 alone above six hours.
50 m is the lattice where the whole usable range of `r` fits in one sitting.

`r` = 1 and `r` = 5 at this lattice existed from earlier runs but were re-measured
here. Those two came from different processes on a contended machine, and the
point of this figure is one comparable series; all seven points below are from a
single sequential process with nothing else running.

## Conditions

| | |
|---|---|
| independent variable | `r` in {1, 2, 3, 4, 5, 6, 7}, Chebyshev |
| spatial step | 50 m, fixed |
| heading spacing | 5 deg (72 bins), fixed |
| terrain | `centered_cube_half_height` |
| Defender seed | nearest lattice point to (5, 0) |
| Defender grid | N_D = 178 |
| lattice sizes | N_S_cart 444,312 · N_S_active 132,452 · N_E 4,133,164 · A 72 |
| episodes | 20,000 at the first Defender position, 4,000 warm-started at each later one |
| Q-learning | alpha 0.5, epsilon 1.0 -> 0.05 decaying 0.999, max_steps 400, seed 0 |
| repetitions | 1 per point, single seed. No variance estimate. |
| wall clock | 3 h 52 min for all seven points |

## Result

| r | nbhd | exact total | ev | it | exact d* | RL total | train | query | ev | it | RL d* | RL/exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 26.7 | 9 | 1 | (5.0, 0.0) | 229.2 | 194.6 | 34.6 | 14 | 2 | (5.5, -0.5) | 8.58 |
| 2 | 24 | 100.8 | 35 | 3 | (7.0, 0.0) | 629.3 | 532.0 | 97.3 | 35 | 3 | (7.0, 0.0) | 6.24 |
| 3 | 48 | 160.5 | 56 | 3 | (7.0, 0.0) | 1090.9 | 929.4 | 161.5 | 56 | 3 | (7.0, 0.0) | 6.80 |
| 4 | 80 | 238.8 | 81 | 2 | (7.0, 0.0) | 1619.8 | 1386.3 | 233.6 | 81 | 2 | (7.0, 0.0) | 6.78 |
| 5 | 120 | 323.1 | 110 | 2 | (7.0, 0.0) | 2301.2 | 1978.8 | 322.4 | 110 | 2 | (7.0, 0.0) | 7.12 |
| 6 | 168 | 393.0 | 134 | 2 | (7.0, 0.0) | 2844.0 | 2446.8 | 397.2 | 134 | 2 | (7.0, 0.0) | 7.24 |
| 7 | 224 | 459.0 | 156 | 2 | (7.0, 0.0) | 3356.6 | 2889.9 | 466.8 | 156 | 2 | (7.0, 0.0) | 7.31 |

`nbhd` is (2r+1)^2 - 1, the neighbourhood the search would evaluate on its first
iteration if the grid were unbounded.

Measured readings:

- **r = 1 is the only radius at which the two solvers disagree.** At r >= 2 both
  land on (7.0, 0.0) and their evaluation counts and iteration counts are
  identical at every radius. At r = 1 exact stops at (5.0, 0.0) and the learner at
  (5.5, -0.5).
- The r = 1 -> r = 2 step is where the climb leaves the neighbourhood it was
  trapped in. Measured separately at this same lattice, d = (7, 0) scores
  J_D = 0.216615 against 0.025019 at (5, 0), a factor of 8.7.
- **Exact total and the learned query are the same curve.** From r = 3 on they
  differ by 1-2%: 160.5 vs 161.5, 238.8 vs 233.6, 323.1 vs 322.4, 393.0 vs 397.2,
  459.0 vs 466.8. Both run once per Defender position, and at this lattice one
  table lookup rollout costs what one Bellman sweep costs.
- All of the learned side's excess is training: 2,889.9 s at r = 7, which is 6.3x
  the exact total. Each new Defender position adds 4,000 warm-started episodes,
  so widening r multiplies the training bill directly.
- Evaluation counts fall below the neighbourhood size as the search reaches the
  grid edge: 110 of 120 at r = 5, 134 of 168 at r = 6, 156 of 224 at r = 7. At
  r = 7 the search has touched 156 of the 178 grid positions, 87.6%.
- Iterations fall from 3 to 2 at r = 4 and stay there. A wider neighbourhood
  travels further per iteration, so the climb terminates in fewer of them.
- **There is no crossing on this axis.** RL/exact is 6.24-7.31 for every r >= 2.
  The two discretization axes both cross (at dx = 5 m and at dpsi = 1.25 deg);
  widening r never lets the learner catch up, because r multiplies both sides by
  the same evaluation count.

Fitted exponents, printed to the console and **not drawn on the figure**:

| series | exponent | R^2 |
|---|---|---|
| Exact, total | 1.438 | 0.9874 |
| Q-learning, training | 1.401 | 0.9991 |
| Q-learning, query | 1.338 | 0.9981 |

All three sit near 1.4, well below the 2 that an unbounded (2r+1)^2 neighbourhood
would give. The grid edge truncates the neighbourhood and the iteration count
drops from 3 to 2 over the same range; both push the exponent down.

## Files

```
data/r_sweep_50m.json       7 points: per-position Defender evaluations for both
                            solvers, stage timings, termination status, d*, the
                            learned side's cumulative training record, lattice sizes
data/r_sweep_50m_run.log    the run's console output, including every Defender
                            position the learned side visited
figure/3_neighbourhood_radius_vs_time.png         log-log
figure/3_neighbourhood_radius_vs_time_linear.png  linear, axes from the origin
```

Produced by `P1b_sweep_figures.py`, which draws all three axes. On this axis the
legend sits upper left, because unlike the discretization axes the series rise to
the right; and the linear panel labels all seven radii, since every value `r` can
take in range is measured.

## Not measured

- Variance over seeds. One run per point.
- `r` > 7 at this lattice. The grid edge makes it meaningless: r = 7 already
  reaches 87.6% of the 178 positions.
- This axis at any other `dx`. Partial points exist at 100 m and 25 m (r = 1 and
  r = 5 only) and are not a series.
- Whether the r = 1 -> r = 2 escape to (7, 0) survives a finer lattice. At 12.5 m
  with r = 1 both solvers stayed near x = 5.
- A seed other than the lattice point nearest (5, 0). The whole axis inherits that
  starting point.
