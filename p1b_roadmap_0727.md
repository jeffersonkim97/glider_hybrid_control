# P1b Roadmap (2026-07-27): ACC/L-CSS Submission Plan

**Supersedes `P1b_roadmap.md`** for planning purposes. That file describes an
earlier architecture (reverse-mode AD through a symbolic CasADi Bellman
recursion, with defender sensor position as a gradient-ascent decision
variable) that was superseded at some point by the architecture actually
implemented in `p1b_4D`/`p1b_3DExtension`: a true leader-follower Stackelberg
solve — DIRECT (asymptotically-global search) for the Defender's
continuous sensor placement, nested around an **exact tabular Bellman DP**
for the Attacker's hybrid powered-glide best response. `P1b_roadmap.md` is
kept as historical record, not deleted. This file reflects reality as of
2026-07-27 and is the one to follow going forward.

Advisor: Dr. Goppert. Prior related work: Kim et al., "A Surveillance
Evasion Game with Continuous Sensor Redeployment via Bilevel Optimization"
(IEEE RA-L, submitted 2026-05-27, arXiv:2605.27917) — see "Relationship to
prior work" below for how this project differs.

---

## Deadlines

- **Joint L-CSS + ACC**: 2026-09-11 (submit via L-CSS site; strict **6-page**
  limit; fast-tracked to ACC presentation if accepted).
- **ACC regular**: 2026-09-25 (**8-page** limit; "finished work" full
  description; no separate short-paper category).
- Both confirmed against the official ACC 2027 CFP/author-info pages.
- Today (2026-07-27) → Sep 11 is 46 days; → Sep 25 is 60 days.

## The actual architecture (current, correct)

- **Stackelberg security game**: Defender commits to sensor position
  (leader); Attacker (hybrid powered→glide glider) computes an exact best
  response (follower) minimizing weighted detection hazard + time.
- **Follower (Attacker)**: exact tabular Bellman DP over a discretized
  state-action grid. `p1b_4D` (2D, side-view `z-h` plane) sweeps z-primary;
  `p1b_3DExtension` (3D, `x-y-h` + heading) sweeps h-primary (h is
  monotonic since `gamma < 0` regardless of heading; z isn't monotonic once
  heading is free). One-time, one-directional powered→glide mode switch;
  the switching point is itself part of what the DP optimizes.
- **Leader (Defender)**: `scipy.optimize.direct` (DIRECT), asymptotically
  global for Lipschitz-ish objectives as evaluation budget → ∞, but a
  **finite budget does not certify global optimality** — this is a known,
  currently-mislabeled gap (`"certified_global": True` is hardcoded
  regardless of whether DIRECT converged via tolerance or exhausted its
  evaluation budget). See Plan item 7.
- **Detection physics**: radar (RCS + aspect angle), Doppler (radial
  velocity), acoustic (powered phase only), LOS/occlusion gating from
  terrain. Multi-modal, physically parameterized (real drag polar: Ostler
  et al. 2009 Unicorn flight-test data).
- **2D status**: mature. Three terrain scenarios already validated
  (1-hill, 2-hill, 2-hill-with-valley) — the framework runs correctly on
  all three. 60/60 unit tests passing. Energy-term experiment (added and
  fully reverted this session — not part of the codebase).
- **3D status**: structurally complete (config→geometry→detection→
  stage_cost→bellman→stackelberg_solver all implemented and individually
  validated), but the Defender's outer loop (`solve_stackelberg_game`) had
  **never been run** until an overnight exact-baseline run was started
  today (2026-07-27, ~19:00) — expect completion ~13 hours later. No test
  suite exists for `p1b_3DExtension` (0 test files, vs. 60 in `p1b_4D`).
  Measured cost: one Defender-position evaluation ≈ 304s, ~4.75GB peak
  Python-tracked memory; a full DIRECT search (`maxfun=150`) extrapolates
  to **~12.65 hours**. This is the concrete, measured reason RL is being
  pursued — not a hypothesis.
- **Known open issue (deferred, not root-caused)**: at a shared degenerate
  (y≈0, heading≈0) scenario built to cross-validate 2D vs. 3D, the two
  packages' Bellman searches select different local-optimal switching
  points, and the gap does **not** shrink with finer resolution (it grew:
  mission_pod delta 0.075→0.27 going from coarse to finer grids). Formula-
  level detection-rate consistency between the packages IS confirmed exact
  (`test_cross_dimensional_consistency.py`, 4/4 passing). Whether the
  full-pipeline gap is (a) 2D and 3D exploring genuinely different feasible
  trajectory sets (3D can express lateral escape, 2D structurally cannot)
  or (b) a real bug in one sweep architecture is **unresolved** — user
  explicitly deferred root-causing this in favor of the ACC/RL work.

## Relationship to prior work (RA-L, arXiv:2605.27917)

Same author, same general Stackelberg/bilevel sensor-placement-vs-evasion
problem family, but a **different projection of the 3D problem** and a
different solution method — not overlapping, complementary:

| | RA-L (own prior work) | This project |
|---|---|---|
| View | Top-down (horizontal `x-y` plane) | Side-view (vertical `z-h` plane), extended to full 3D in `p1b_3DExtension` |
| Occlusion geometry | Buildings (convex polygons) | Terrain elevation profile (ridge silhouette) |
| Attacker dynamics | Generic UAS (kinematic) | Hybrid powered→glide, real drag polar, one-time mode switch |
| Defender | Multiple sensors sliding along building perimeters (continuous redeployment) | Single sensor, terrain-mounted |
| Solve method | Gradient-based bilevel alternation (log-sum-exp smoothing) → **local** Nash equilibrium | DIRECT (asymptotically-global) nested around **exact** discretized-optimal Attacker DP |
| Validation | 500-trial Monte Carlo, 4x vs. random placement | Residual/telescoping-sum checks; formal discrete-optimality proposition (planned, item 6) |

Must be cited explicitly in related work with this table's distinctions
made clear — self-citation is fine and expected, but the paper must not
read as a re-solve of the RA-L problem.

## The actual contribution (precise statement)

No new theorem/algorithm at the "new math" level — DAG-structured tabular
Bellman DP optimality and DIRECT's asymptotic convergence are both
established results. The contribution is the **formulation**: jointly
optimizing (a) *where* to switch from powered to glide flight and (b) the
subsequent glide trajectory, as a single exact discretized DP problem,
nested as the follower inside a continuous Defender sensor-placement
Stackelberg game with physically-grounded multi-modal detection. No prior
work found combining all three of {hybrid mode-switch trajectory
optimization} × {detection-avoidance} × {Stackelberg sensor placement} —
each pairwise/individual piece has precedent (boost-glide mode-switch
literature; terrain-masking radar-avoidance path planning, e.g. Pelosi et
al. 2012; bilevel sensor-deployment literature), the triple combination
does not.

This sentence (or a refined version of it) should open the paper's
contributions paragraph, stated *before* any implementation/pipeline
description — the risk GPT flagged (ACC cares about control-theoretic
content, not "a system that runs") is real and this is the fix.

---

## Final experimental/writing plan

Order matters: convergence checks (1-2) come before the expensive
multi-terrain sweep (4), so an unstable resolution doesn't force redoing
everything downstream.

**1. Follower-level resolution convergence**
Representative sensor positions (2-3: default/domain-midpoint,
coverage-max/near-ridge, and any position where a coarse outer sweep shows
Attacker topology changing) × coarse/medium/fine grids. Track `J_A`, PoD,
mission time, switching point, path topology, terrain/goal feasibility.
*Expected output*: per-position table/plot of these quantities vs.
resolution, judged against a pre-declared numeric tolerance (not "looks
stable").

**2. Defender-level resolution convergence**
Same sensor-position set, each resolution level: compare `J_D(z_s)`
landscape shape, local-maxima locations, winning-basin identity, sensor
ranking. A coarse discretized sweep suffices — no need to rerun full DIRECT
at every resolution.
*Expected output*: overlaid `J_D` vs. sensor-position curves per
resolution, confirming the optimal basin doesn't move.

**3. Production resolution, locked**
Pick the coarsest resolution that satisfies both (1) and (2)'s tolerance —
coarsest because everything downstream (4) re-runs this many times.
*Expected output*: one `(z_count, h_count, v_count, gamma_count)` tuple,
documented with the tolerance it satisfied.

**DECIDED (2026-07-28)**: 2D production resolution = **medium tier**,
`z_count=201, h_count=131, v_count=4, gamma_count=14`. At 3 representative
sensor positions (2900/3625/4400), `defender_objective` ranking was
identical across coarse/medium/fine (4400 > 3625 > 2900), and each
position's value varied <2% between medium and fine — tolerance: *ranking
unchanged + <2% relative value change vs. fine*. Medium was chosen over
fine for cost (~5s/eval vs. ~45-50s/eval, ~9x cheaper) since it already
meets the tolerance. See `results/resolution_convergence_2d/` for the raw
sweep data (3x3 grid + a 250m dense scan at medium resolution across the
full sensor bound range, used to confirm the underlying PoD-vs-position
curve is smooth, not just spot-checked at 3 points).

**CORRECTION (2026-07-28, post-review)**: the note above understated a
real problem. An external review (independently recomputed from the raw
JSON) showed follower-level relative differences between medium and fine
are large (e.g. PoD at z=4400: 60.5% relative difference) -- not
acceptable noise, and the "medium, locked" decision above was premature:
it rested only on the aggregated `defender_objective` (coverage-dominated,
insensitive to the actual instability) rather than the follower metrics
that actually matter.

**Root cause, isolated via a 2x2 factorial (spatial grid x action grid,
holding one fixed while varying the other) at z=4400**:

| spatial | action | mission_cost | mission_time |
|---|---|---|---|
| coarse (121,81) | coarse (3,8) | 0.3041 | 131.90s |
| coarse (121,81) | fine (5,20) | 0.3035 | 131.90s |
| fine (321,201) | coarse (3,8) | 0.3812 | 164.71s |
| fine (321,201) | fine (5,20) | 0.3803 | 163.86s |

The **spatial (z,h) grid is the entire driver** -- action (v,gamma) grid
resolution changes cost by <0.2%; spatial grid resolution changes it by
~25%. Mechanism: `construct_coarse_transitions` maps each kinematic
step's continuous displacement to the next grid index via `ceil`, which
always advances a *full* grid cell regardless of how small the actual
sub-cell displacement was. At coarse spacing this systematically
**overstates** distance-covered-per-step (e.g. a true ~14m step ceils to
a full ~45.8m coarse cell vs. a ~17.2m fine cell) -- fewer discrete steps
are recorded than physically occurred, so mission_time and accumulated
hazard are **directionally undercounted**, more severely as resolution
coarsens. This is not random noise that averages out; it is the same
grid-snapping mechanism flagged during the (reverted) energy-term
investigation earlier this session, now confirmed to also drive this
instability.

**Decision**: production resolution is the existing default
(`z_count=321, h_count=201, v_count=5, gamma_count=20`, i.e. what this
document previously called "fine") -- **no change to
`p1b_4D/configuration.py` is needed**, since that was never actually
changed to "medium" (a real gap the external review also caught: this
document claimed the resolution was "locked" to medium while the code
remained at the current defaults the whole time). Coarser tiers are kept
only as cheap diagnostic/exploratory resolutions, explicitly documented as
carrying a characterized, directional undercounting bias -- not used for
any reported result. This is a case for the paper's limitations section:
discretization error here is not simply "smaller grid = more noise," it
has a known sign and mechanism.

**DONE (2026-07-28)**: ran all 12 (3 terrain x 4 baseline) evaluations,
each baseline position re-evaluated by the same authoritative
`evaluate_defender_position`. Full results in
`results/multiterrain_strategic_baselines/multiterrain_baseline_results.json`.

| Terrain | fixed J_D | coverage_only J_D | nominal_path J_D | stackelberg J_D |
|---|---|---|---|---|
| single_hill | 0.3111 | 0.3278 | 0.3029 | **0.3278** |
| two_hill | 0.3707 | 0.4753 | 0.2122 | **0.4780** |
| goal_in_valley | 0.3447 | 0.3498 | 0.3449 | **0.3531** |

Key finding, more nuanced (and more useful for the paper) than a flat
"Stackelberg always wins big" story:
- **single_hill**: `stackelberg` picks the *exact same* sensor position as
  `coverage_only` (4498.8) and gets an identical result — strategic
  reasoning adds nothing here; the single-obstacle geometry is simple
  enough that maximizing raw LOS coverage already finds the optimum.
- **two_hill**: `stackelberg` narrowly beats `coverage_only` (+0.6%), but
  `nominal_path` (assume a non-adaptive attacker) badly *backfires*
  (0.2122, worse than even `fixed`) — assuming the attacker won't adapt is
  actively dangerous once there's more than one obstacle to route around.
- **goal_in_valley**: `stackelberg` clearly beats all three naive
  baselines (+0.9% over the best of them) and — the most interesting part
  — picks a *qualitatively different* sensor region (1705.8) while the
  other three all cluster around 1950-2199. This is the strongest
  "strategic reasoning finds something naive placement structurally
  cannot" result of the three terrains.

Narrative for the paper: strategic value is terrain-complexity-dependent,
not uniform — worth stating explicitly rather than only reporting the
best-case (goal_in_valley) result.

**4. Multi-terrain × strategic-baseline sweep**
3 terrains (1-hill, 2-hill, 2-hill-with-valley; already built/validated) ×
4 sensor-placement rules, each an *independent selection process*: (a)
fixed/default, (b) coverage-only-optimal (geometry alone, no attacker
knowledge), (c) nominal-path-optimal (assumes a fixed non-adaptive
Attacker), (d) Stackelberg-optimal (DIRECT + exact Attacker). Every
candidate position is then re-evaluated with the *same*
`evaluate_defender_position` call — same model/objective/solver/grid, but a
fresh Attacker best-response solve per position. Never reuse a trajectory
across baselines.
*Expected output*: 3×4 table of (mission_pod, mission_cost,
defender_objective) — the paper's headline empirical result: does
accounting for the Attacker's strategic response actually beat naive
placement, and does it hold across terrains (not a one-off artifact).

**5. Scalability writeup**
Compile runtime/memory data collected as a byproduct of (1)-(2) alongside
the convergence results.
*Expected output*: one figure/table pairing compute cost against solution
stability ("what does exactness cost, and is it worth it").

**DONE (2026-07-28)**: `results/scalability_figure.png` (runtime and
memory vs. state-action cell count, log-log, 2D and 3D both plotted).

| | Cell count | Runtime | Peak memory |
|---|---|---|---|
| 2D coarse | 235,224 | 1.0s | 42.0MB |
| 2D medium (production) | 1,474,536 | 5.3s | 255.5MB |
| 2D fine | 6,452,100 | 46.0s | 1,104MB |
| 3D coarse | 8,718,192 | 27.4s | not measured |
| 3D medium (default) | 33,767,928 | 304.0s | 4,750MB |
| 3D fine | 175,750,272 | 705.7s | not measured |

Outer-loop (DIRECT) totals, 2D vs. 3D:

| | per-eval | outer evals | total |
|---|---|---|---|
| 2D single_hill | ~15s | 60 | 882s (14.7 min) |
| 2D two_hill | ~18s | 60 | 1088s (18.1 min) |
| 2D goal_in_valley | ~25s | 60 | 1518s (25.3 min) |
| 3D (default) | ~304s | 190 | 19,526s (5.42 h) |

**Key finding**: log-log regression of runtime vs. cell count across all 6
points (2D and 3D together, ~3 orders of magnitude in cell count) gives a
**scaling exponent of 1.043 — essentially linear, O(N)**. This is not just
a nice empirical fact: it's independent supporting evidence for item 6's
discrete-optimality proposition, specifically the claim that
`solve_coarse_bellman` is a *single* backward sweep over a topologically-
sorted DAG doing O(1) work per cell (confirmed by reading the code:
`"sweep_count": 1` in its own diagnostics, not an iterate-to-convergence
loop) — theory and measurement agree, worth stating together in the paper
rather than as two disconnected sections.

Note: tracemalloc instrumentation itself adds ~10x runtime overhead
(measured directly) -- runtime and memory numbers above come from
separate, purpose-specific runs, never the same run for both.

**6. Discrete-optimality proposition + small-instance cross-check**
Formal proposition, with proof: (i) the transition graph is a finite DAG
(monotonic z or h sweep); (ii) all feasible `(v, gamma[, heading])` actions
are enumerated; (iii) switching-seed completeness is over the
**discretized admissible switching set the code actually defines** (not
the continuous LOS boundary — precise phrasing matters); (iv) local cost
is additive along any path; (v) terminal cost matches the true objective;
(vi)+(vii) backward induction therefore yields the exact discrete Attacker
optimum. Existing residual/telescoping-sum tests are *implementation
validation of this proposition's hypotheses*, not a substitute for the
proof — keep these framed as distinct in the writeup. Add one new small
toy-grid cross-check against an independent brute-force/shortest-path
solver.
*Expected output*: proposition + proof (paper section draft) + one passing
cross-check test.

**DONE (2026-07-28)**: `p1b_4D/discrete_optimality_proposition.md` —
4 propositions (finite DAG forced by vehicle dynamics; exhaustive
action/switching-seed enumeration; backward induction = exact DAG
shortest-path; combined = exact global discrete Attacker optimum), each
proved directly from the code's control flow, plus an explicit "scope and
what this does not claim" section (discretization-only, attacker-only,
scaling-exponent-is-corroboration-not-proof). Cross-check:
`p1b_4D/test_discrete_optimality_crosscheck.py` — independent networkx
Dijkstra reimplementation over the real `construct_coarse_transitions`
output, matches `solve_coarse_bellman` exactly (rtol=0, atol=1e-9) across
211 reachable toy-grid cells. Full `p1b_4D` suite re-run afterward to
confirm no regression.

**7. `certified_global`/`converged` terminology sweep**
Full sweep — not just the `direct_global_optimizer` metadata dict found
this session, but tests, plot titles, notebook markdown, exports, and any
docs — replacing overclaims with precise language:
- Attacker: "global optimum of the discretized attacker problem" (this one
  IS a certified claim, given the proposition in item 6).
- Defender: "best-found continuous defender solution" (DIRECT,
  asymptotically-global algorithm, finite-budget result — not certified).
- Overall: "nested Stackelberg solution with an exact discretized follower
  response."
*Expected output*: grep-clean codebase, consistent terminology everywhere.

**DONE (2026-07-28)**. Changes, both `p1b_4D` and `p1b_3DExtension`:
- `direct_global_optimizer`'s `"converged"` now reflects `result.success`
  (DIRECT's real length-tolerance-vs-budget-exhaustion signal) instead of
  a hardcoded `True`.
- `"certified_global": True` removed from metadata entirely (no boolean
  value of it was ever a rigorous claim). Replaced with
  `"algorithm_class": "direct_asymptotically_global"` (describes the
  algorithm's theoretical property) and `"terminated_via": "length_
  tolerance" | "evaluation_budget"` (describes this run's actual outcome).
- **Real bug found and fixed while wiring this through**:
  `validate_stackelberg_solution`'s `checks` dict included
  `"outer_optimizer_convergence"` as a *hard* pass/fail gate, contradicting
  the code's own stated philosophy that budget-exhaustion termination is
  "normal and expected, not a failure" for an expensive per-evaluation
  objective. Previously masked because `converged` was hardcoded `True`
  (so this check silently never fired); fixing `converged` to be honest
  would have made `solve_stackelberg_game` start reporting `status.
  success = False` on every ordinary run that exhausts budget instead of
  hitting `len_tol` -- which is the common case for an expensive
  objective. Removed from `checks`, kept as the (already-existing, more
  appropriate) warning: *"Outer optimizer terminated via evaluation-budget
  exhaustion... a valid best-found result, not a certified global optimum
  for this run."* Also surfaced in `metrics.outer_optimizer_converged` for
  visibility without gating pass/fail.
- Docstrings/comments in both `stackelberg_solver.py` files and
  `p1b_3DExtension/configuration.py` reworded ("certified-global" ->
  "asymptotically-global") to stop asserting a stronger guarantee than
  DIRECT actually gives.
- Both notebooks (`p1b_4D/stackelberg_security_problem.ipynb`,
  `p1b_3DExtension/p1b_3DExtension.ipynb`) swept -- markdown cells, code
  cells (print/title strings), and stored text outputs all updated to the
  same wording.
- `p1b_4D/test_stackelberg_solver.py`'s `certified_global` assertion
  replaced with `algorithm_class`/`terminated_via` assertions; found (via
  actually running the test, not assumption) that even its synthetic
  smooth-objective case terminates via budget exhaustion at maxfun=40 --
  assertion corrected to match measured reality rather than what seemed
  intuitive.
- Full `p1b_4D` suite re-run: 61/61 passing. `p1b_3DExtension` has no test
  suite (known, already-tracked gap) -- changes there verified by direct
  code inspection mirroring the `p1b_4D` fix exactly, plus an import/
  syntax smoke test.

**(Parallel, no-regret, starts now)**: 2D ACC manuscript draft, filled in
as (1)-(7) complete. Independent of the go/no-go outcome below.

## Go/no-go checkpoint: ~mid-August

Decide whether to pursue the ambitious 2D+3D+RL story (Sep 11, joint
L-CSS+ACC) or fall back to a polished 2D-only story (Sep 25, ACC regular).
**"Well-finished 2D ACC regular beats a half-finished 2D+3D+RL L-CSS
submission"** — the fallback is not a failure state, it's the safe default.

Objective criteria for proceeding to the L-CSS track (need most/all, not
"some progress"):
- 3D nested pipeline runs end-to-end (reduced or full setting).
- RL and the exact Bellman DP share the same objective and a common
  evaluator (if RL uses continuous state/control rather than the same
  discretized grid/MDP, teacher-student framing weakens and the common
  evaluator becomes mandatory, not optional).
- High feasibility rate on held-out sensor positions/terrains (generalizes,
  not just memorizes the training scenario).
- Bellman-vs-RL objective gap within a pre-stated bound.
- Real, measured runtime speedup over the exact 3D pipeline.
- Stable across multiple training seeds.
- **Defender regret**: `J_D(z_D^exact) - J_D(z_D^RL)` — i.e. does plugging
  the RL-approximated Attacker response into the *outer* Defender search
  actually pick a meaningfully worse sensor position, even if the
  follower-level approximation error looks small? This is the metric that
  actually matters, not raw value-function MSE.
- The whole 2D+3D+RL story is explainable as **one central claim** that
  fits in 6 pages (e.g., framed as: "the exact Bellman solver is a teacher
  for the discretized 3D follower problem; RL approximates it to make the
  outer Defender loop tractable").

If several of these are missing by mid-August: commit fully to the 2D-only
ACC-regular path (Sep 25) and treat 3D/RL as future work in the
discussion/conclusion section.

## RL groundwork status (as of 2026-07-27)

- `torch==2.13.0` (CPU), `gymnasium==1.3.0`, `stable-baselines3==2.9.0`
  installed into `.venv_p1b`, no conflicts with pinned casadi/numpy/scipy.
  `p1b/requirements.txt` and `SETUP.md` updated and re-verified so a fresh
  machine reproduces this via the existing single `uv pip install -r`
  step.
- Algorithm direction (approximate/fitted value iteration on the known
  transition model vs. genuine model-free RL) is **not yet decided** —
  advisor has mandated model-free RL specifically (not just a Bellman/RL
  reframing), to be confirmed at the Thursday meeting (2026-07-30).
- Actual environment-wrapper code (Gymnasium-style `step()`/`reset()`
  around the existing validated `p1b_4D` physics) has **not been built
  yet** — paused pending the Thursday direction confirmation. When built:
  new `p1b_rl/` directory, wraps but does not modify the validated
  `p1b_4D`/`p1b_3DExtension` pipeline code.
- Overnight exact-3D-baseline run (`results/overnight_3d_baseline/`,
  detached process, started 2026-07-27 ~19:00) exists specifically to give
  the eventual RL/approximate solution a ground truth to validate against.

## Explicit scoping decisions

- **`n_sensor = 1`, kept for this submission.** Not a placeholder — a
  deliberate foundational-case scope. Multi-sensor is a categorically
  different problem, not a config bump: (a) Defender decision dimension
  grows to `N_s` (2D) or `2N_s` (3D), plus sensor-permutation symmetry,
  minimum-separation, and combinatorial local-maxima issues — directly
  worsens the already-severe outer-loop compute bottleneck; (b) LOS
  geometry changes structurally — visible-to-any-sensor is a *union*,
  hidden-from-all-sensors (the powered-flight-feasible region) is an
  *intersection* of per-sensor occlusion regions, and the switching
  boundary is no longer a single LOS tangent; (c) a detection-fusion rule
  must be chosen (summed hazard vs. independent-detection product vs.
  strongest-sensor vs. correlated) and the choice changes the game itself.
  Justification for the paper: *"this work characterizes the foundational
  Stackelberg interaction between a single terrain-mounted sensor's
  continuous placement and a hybrid Attacker's best response; multi-sensor
  extension is separate future work requiring new defender-dimensionality,
  viewshed-union geometry, and detection-fusion treatment."* Architecture
  is *not* a dead end for this — the existing additive-hazard telescoping
  pattern generalizes naturally to a sum over sensors later.

## If RL succeeds later (separate track, not part of the Sep deadlines)

JGCD (Journal of Guidance, Control, and Dynamics) is a strong long-term
target — arguably a more natural subject-matter fit than ACC given the
real aerodynamics/vehicle-model content, and JGCD actively publishes
RL-for-guidance/control work. Should be planned as a **later, separate,
non-deadline-driven extension** once RL is genuinely mature (more
scenarios, deeper validation than any conference paper needs) — most
naturally as an extended journal version of a successful L-CSS letter, the
standard letter-then-journal-extension pattern. Do not let this pull focus
from the Sep 11/25 planning above.
