# P1b Roadmap: CasADi Symbolic Bellman → Minimax → Reinforcement Learning → 3D

**Revision note (v3)**: The previous draft incorrectly scoped the symbolic
reconstruction target to `p1b_main.ipynb` Steps 1-6 (the 2D point-mass
Dijkstra baseline, constant `v = V_LAUNCH`), and stated that Step 9's 4D
fixed-wing backward DP was out of scope. **This was a misunderstanding of
the advisor's actual direction.** Per direct confirmation: the intended
baseline is the **4D fixed-wing backward DP, projected to 2D** — i.e. Step
9's approach (`p1b_4d_dp.py`), not the point-mass Dijkstra. The point-mass
model is retained only as a *pedagogical stepping stone* (toy-scale pattern
validation before tackling the full 4D problem), not as the target model
itself.

**Context:** Counter-UAS Stackelberg Security Game (SSG). Attacker:
fixed-wing UAV (`glider_dynamics` — RK4 integration in Cartesian velocity
components, lift/drag/gravity, closed-loop `required_CL` attitude autopilot;
see `p1b_main.ipynb` Step 4 / `p1b_4d_dp.py`). Defender: sensor placement
(acoustic, Doppler, RCS models with Poisson fusion). Current baseline: a 2D
**projected** cost-to-go map `J_2D(z,h) = min_{v,γ} J_4D(z,h,v,γ)`, where
`J_4D` is the exact backward Bellman solution over the full 4D state
`(z,h,v,γ)` — unlike a point-mass model, `v` is a genuine physical state
here (it evolves under real dynamics, not reset every stage), which is
exactly why the 4D solve is needed before projecting down to 2D for
visualization/use. Numerically, this is what `p1b_4d_dp.py` already
computes (`J_2D_4d = np.nanmin(J_4D, axis=(v,γ))` in the notebook's Step 9).

The powered phase (launch → switching point on the LOS boundary) remains a
simple constant-speed straight-line kinematic dash (`powered_trajectory`,
no thrust/drag model) — that simplification is orthogonal to this roadmap
and is not being revisited here. Only the **glide phase** cost-to-go
(`J_4D` / its 2D projection) is being reconstructed symbolically.

Advisor (Dr. Goppert) confirmed the 4D-then-project approach as the basis
for this work and defined a 3-stage plan. Stage 3 is the journal submission
target (L-CSS + ACC 2027, deadline Sep 11, 2026).

**Development approach**: build and validate the symbolic pipeline
incrementally, at toy scale, before committing to the full-resolution grid
(which is expensive to iterate on):
1. **Point-mass toy grid** (2D state, ~20x20) — validates the CasADi
   mechanics in isolation (smooth occlusion, `z_tangent` via
   `casadi.rootfinder`, value iteration as a repeated `Function`
   composition, AD vs. finite-difference) without also debugging 4D
   dynamics at the same time. Complete — see "Phase 1 findings" below.
2. **Fixed-wing toy grid** (4D state, small per-dimension counts) — same
   validation, now with real `glider_dynamics` and the `min_{v,γ}`
   projection step. Next.
3. **Fixed-wing full grid** (matching `p1b_4d_dp.py`'s resolution) — scale
   up once the toy version is validated.

### Phase 0/1 findings (point-mass toy grid, already validated)

These implementation choices were worked out empirically at toy scale and
carry forward to the fixed-wing toy/full grids:

- `casadi.interpolant` with runtime/symbolic coefficients did not behave as
  documented for this use case (wrong values away from grid boundaries,
  zero gradient w.r.t. the data vector). **Workaround**: since transition
  dynamics never depend on `Z_SENSOR` (only the stage cost does), the
  landing-position interpolation weights can be precomputed once in plain
  numpy and applied as a constant weight matrix (`W @ J`) — trivially
  differentiable, sidesteps the issue entirely. For the fixed-wing case
  this generalizes from bilinear (2D, 4-corner) to quad-linear (4D,
  16-corner) interpolation weights.
- `z_tangent` (LOS grazing point) was a discrete, non-differentiable search
  in the numeric code; reimplemented via `casadi.rootfinder` on the
  tangency condition. Matches the numeric value to within the numeric
  method's own discretization error.
- Occlusion (`z < z_tangent and h < h_los(z)`) was a hard boolean in the
  numeric code, used for two distinct purposes that must be smoothed
  separately: (1) zeroing detection cost inside the pocket, and (2)
  excluding the pocket from the glide-phase DP's domain (it belongs to the
  powered phase). A hard cutoff gives ~zero gradient w.r.t. `Z_SENSOR`
  almost everywhere, defeating the point of the symbolic reconstruction.
  **Resolved**: sigmoid-smoothed indicator `w(z,h) ∈ [0,1]`, applied to (1)
  the stage cost and (2) as a steep smooth penalty (not literal exclusion —
  CasADi's fixed graph topology can't structurally remove
  `Z_SENSOR`-dependent cells anyway). Validated: AD gradient matches
  finite-difference to ~1e-4, non-zero for 285/286 reachable toy cells.
- Value iteration (fixed-count Bellman backup sweeps) replaces Dijkstra,
  since point-mass backward flight makes the reachability graph non-DAG and
  CasADi needs a static graph topology regardless. The goal cell's `J=0`
  must be explicitly re-pinned every sweep (a bug found during
  implementation — without it, the goal's own "backup" overwrites its
  anchor value and nothing propagates).
- Toy-scale symbolic `J_2D` matches the numeric reference reasonably well
  in shape/magnitude; the largest discrepancies are near the
  highest-detection-risk region, likely from the toy's coarse grid and its
  simplified (non-substepped) stage cost evaluation — flagged for the
  fixed-wing toy/full-grid passes.

---

## Stage 1-1: Symbolic CasADi Reconstruction of the 4D Fixed-Wing Cost-to-Go + Energy Constraint

**Objective:** Reconstruct the 4D fixed-wing backward Bellman recursion
(`J_4D(z,h,v,γ)`, exact — not a point-mass approximation) as a symbolic
CasADi computation, project to 2D (`J_2D = min_{v,γ} J_4D`), and add an
energy constraint. Enables reverse-mode AD through the full pipeline,
required for Stage 1-2.

### Step-by-step tasks for Claude Code

1. **Fixed-wing toy grid (current step)**
   - Small per-dimension grid, e.g. `NZ~10-15, NH~10-15, NV~3-4, NG~7-9`.
   - Replace point-mass kinematics with symbolic `glider_dynamics` (RK4 in
     Cartesian velocity components; the `required_CL` autopilot's `CL_MAX`
     clip is a smooth `fmin`/`fmax`, not a hard branch, so this should port
     cleanly).
   - Reuse: rootfinder for `z_tangent`, sigmoid occlusion (applied to both
     stage cost and glide-phase domain scoping), goal re-pinning.
   - New: generalize the weight-matrix trick to quad-linear (4D)
     interpolation; port the LOS-zone-only domain restriction, the
     landing-altitude goal condition, and the climb-angle-extended `γ`
     state grid from `p1b_4d_dp.py`'s fixes (see prior session — occlusion
     leaking into the glide-phase domain, and the goal condition crediting
     any altitude once `z >= Z_GOAL`, were both real bugs caught and fixed
     there).
   - New: the `J_2D = min_{v,γ} J_4D` projection is itself a new operation
     in the AD graph and must be validated separately — check
     `d(J_2D)/d(Z_SENSOR)` via AD vs. finite-difference specifically
     through the projection step, not just through the 4D backup.
   - Validate against `p1b_4d_dp_results.npz`'s numeric `J_2D_4d`.

2. **Energy constraint**
   - Unlike the point-mass model, `v` is a real evolving state here, so
     specific energy `e = (KE + PE)/m = g·h + v²/2` is physically
     meaningful (not degenerate) — this resolves the earlier confusion
     about what "energy" even means for a constant-speed vehicle.
   - Prototype comparison already done (see `p1b_casadi_symbolic.ipynb`,
     "Stage 1-1 Task 3" section): two candidate interpretations — (A) cap
     on powered-phase range/distance, (B) cap on altitude gained
     (`g·h_sw`) — rank the same switching candidates in exactly opposite
     order. Advisor input pending on which (or both, applied to different
     phases — powered vs. glide) is intended, and whether exceeding the
     budget should be a hard exclusion or a cost penalty.
   - Once confirmed, incorporate using the real `v²/2` term (now
     non-degenerate) for the glide phase.

3. **Full-grid fixed-wing reconstruction**
   - Scale to match `p1b_4d_dp.py`'s resolution (`NZ=100, NH=100, NV=4,
     NG=37` after the climb-angle grid extension — ~1.48M `(z,h,v,γ)`
     combinations).
   - **Sparse weight matrices are mandatory here** (not optional, unlike
     the point-mass toy where dense 400x400 was fine) — a dense per-action
     weight matrix at this scale would be ~1.48M x 1.48M, infeasible.
   - Re-verify sweep count for convergence at full resolution (the toy
     grid's ~30 sweeps was resolution-dependent, not a general constant).

4. **Validation**
   - Plot symbolic vs. numeric `J_2D` heatmaps side by side (reuse
     `p1b_4d_dp.py`'s plotting conventions: terrain mask, occlusion mask,
     LOS line, sensor marker).
   - Confirm the occlusion domain-scoping does not leak cost-to-go into the
     powered-phase pocket (the exact bug fixed in `p1b_4d_dp.py` last
     session — re-verify it doesn't reappear in smoothed form).
   - Full-grid `d(J_2D)/d(Z_SENSOR)` AD vs. finite-difference check — the
     actual deliverable enabling Stage 1-2.

**Deliverables:** CasADi symbolic 4D→2D Bellman pipeline producing `J_2D`
matching the `p1b_4d_dp.py` numeric baseline; energy-constrained cost-to-go
(form confirmed with advisor); documented occlusion-differentiability
approach (done — see Phase 0/1 findings above).

**Open questions to confirm with advisor:** (1) energy constraint form —
Option A (range cap) vs. Option B (altitude/energy cap) vs. both applied to
different phases; (2) hard constraint vs. penalty for the energy budget.

---

## Stage 1-2: Simultaneous Attacker-Defender Minimax via Reverse-Mode AD

**Objective:** Replace the fixed defender position assumption from Stage
1-1 with a simultaneous minimax optimization: the defender's sensor
position becomes a decision variable optimized jointly with the attacker's
trajectory, using reverse-mode AD through the symbolic 4D→2D Bellman
pipeline built in Stage 1-1.

**Framing note (from advisor's meeting analogy):** the defender position is
a single decision variable ("where to move the camera left or right"),
differentiated through the full symbolic cost-to-go via reverse-mode AD,
rather than solved via grid search or bilevel optimization as in prior work
(RA-L submission). This is a new outer-loop optimization — `Z_SENSOR_MIN` /
`Z_SENSOR_MAX` / `N_SENSOR_SWEEP` exist as declared parameters in
`p1b_main.ipynb` but are not currently used by any implemented sweep, so
there is no existing brute-force loop being "replaced," only new
optimization being added.

### Step-by-step tasks for Claude Code

1. **Parametrize defender position symbolically**
   - Introduce sensor position as a CasADi decision variable rather than a
     fixed parameter in the stage cost function from Stage 1-1.
   - Confirm dimensionality: start with 1D (matches "left or right"
     framing) before generalizing to 2D if needed.
   - Note: `z_tangent` (via `rootfinder`) and the LOS slope/intercept are
     already expressed as symbolic functions of `Z_SENSOR` in Stage 1-1
     (not baked in as constants), so this promotion should not require
     re-deriving the occlusion geometry.

2. **Formulate the minimax problem**
   - Attacker minimizes cost-to-go (`α1·P_D + α2·T`) as before.
   - Defender maximizes (`β1·P_D + β2·Coverage`) by choosing sensor
     position.
   - Set up as a nested/simultaneous optimization: either (a) a true
     minimax NLP solved via CasADi's `nlpsol` with appropriate sign
     conventions, or (b) reverse-mode AD to get `∂J/∂(defender position)`
     and use gradient ascent on defender position wrapped around the Stage
     1-1 Bellman solve.

3. **Reverse-mode AD implementation**
   - Use CasADi's automatic differentiation to compute the gradient of
     total cost-to-go with respect to defender position.
   - Implement gradient ascent (or a proper minimax solver, e.g.,
     alternating gradient steps or a single joint NLP) to find the
     simultaneous optimum.
   - Verify gradient correctness via finite-difference check on a small
     grid before scaling up (already the established pattern from Stage
     1-1's toy-grid validation).

4. **Compare against Stage 1-1 fixed-defender baseline**
   - Quantify the gap between fixed defender placement and optimized
     simultaneous defender placement (headline metric, analogous to the
     ΔJ_D gap reported in the GameSec ridge scenario).

5. **Literature check (per advisor's note)**
   - Search whether reverse-mode AD applied directly to a Bellman/dynamic-
     programming cost-to-go for adversarial/pursuit-evasion games has
     precedent, to establish novelty framing for Stage 3's journal paper.
     Use Google Scholar only, exclude MDPI results.

**Deliverables:** Simultaneous minimax solver (attacker trajectory +
defender position) via CasADi reverse-mode AD; quantified improvement over
fixed-defender baseline; literature summary on AD-through-Bellman
precedent.

---

## Stage 2: Reinforcement Learning Reformulation

**Objective:** Solve the same problem (attacker cost-to-go, with the Stage
1-2 simultaneous minimax defender optimization) using reinforcement
learning instead of symbolically computing the exact cost-to-go, to compare
a learned approximation against the exact symbolic solution.

**Scope confirmed:** RL is applied to the Stage 1-2 formulation
(simultaneous minimax), not the Stage 1-1 fixed-defender version.

### Step-by-step tasks for Claude Code

1. **Literature review**
   - Survey RL literature for learning cost-to-go functions in place of
     symbolic/exact DP solutions, particularly in adversarial or
     pursuit-evasion/security game settings. Use Google Scholar only,
     exclude MDPI.
   - Identify closest prior work to determine what is genuinely novel about
     combining Stage 1-2's simultaneous AD-based minimax with RL, versus
     what is standard.

2. **Define RL problem formulation**
   - State space: `(z, h, v, γ)` (the real 4D fixed-wing state — not the 2D
     projection, since the projection is a visualization/summary of the
     exact solution, not the state the vehicle actually occupies) plus the
     energy state from Stage 1-1.
   - Action space: `γ_cmd` for the attacker; defender position as either a
     separate learning agent or an outer-loop optimization (decide based on
     whether a two-agent/multi-agent RL setup or a single-agent RL with
     defender optimized in an outer loop is more tractable).
   - Reward: negative of attacker's stage cost (`α1·P_D + α2·T`), consistent
     with Stage 1-1/1-2 cost structure.

3. **Select and implement RL algorithm**
   - Given continuous state/action spaces, candidates: DDPG, SAC, or PPO.
     Note: prior sessions discussed SAC vs. Bellman optimality and
     advisor's stated preference for deterministic worst-case (minimax)
     formulations, so document how the RL formulation reconciles with or
     departs from that preference.
   - Implement using an existing RL library (e.g., stable-baselines3)
     rather than from scratch, unless a specific reason requires custom
     implementation.

4. **Train and validate**
   - Train RL policy on the same state/action/cost structure as Stage 1-2.
   - Compare learned cost-to-go / policy against the symbolic Stage 1-2
     solution: trajectory shapes, defender placement, and total
     cost-to-go values.

5. **Comparative analysis**
   - Quantify gap between symbolic exact solution (Stage 1-2) and
     RL-learned approximation (Stage 2).
   - Discuss trade-offs: computational cost, scalability to higher
     dimensions (motivates Stage 3), exactness vs. generalization.

**Deliverables:** Trained RL policy matching Stage 1-2 problem structure;
quantitative comparison against symbolic minimax solution; literature
positioning for novelty claim.

---

## Stage 3: Extension to Realistic 3D Environment (Journal Target)

**Objective:** Extend Stage 1/Stage 2 methods, currently confined to the 2D
z-h plane (with the underlying 4D `(z,h,v,γ)` fixed-wing state), to a
realistic 3D environment. This is the primary contribution targeted for the
L-CSS + ACC 2027 joint submission (deadline Sep 11, 2026).

### Step-by-step tasks for Claude Code

1. **Define 3D state space**
   - Extend `(z,h,v,γ)` to full 3D position + attitude (e.g., downrange,
     crossrange, altitude, flight-path angle, heading/bank angle).

2. **Extend sensor models to 3D**
   - Acoustic, Doppler, and RCS detection probability models currently
     depend on range `r` and aspect angle `γ`; extend range and
     aspect-angle computations to full 3D geometry.

3. **Extend defender decision variable**
   - Generalize defender position from 1D (Stage 1-2) to 2D or 3D sensor
     placement.

4. **Re-derive/extend symbolic Bellman (from Stage 1) and/or RL (from
   Stage 2) to 3D**
   - Decide whether Stage 3 uses the symbolic CasADi approach, the RL
     approach, or both, for the 3D case, based on which scaled better in
     the Stage 1 vs. Stage 2 comparison.
   - Address curse-of-dimensionality concerns explicitly: Stage 1 is
     already a 4D (not 2D) exact solve projected for display, so this is a
     continuation of an already-accepted higher-dimensional approach, not
     a new departure from a "2D-only" baseline. Stage 3 must justify why a
     5-6D solve (3D position/attitude + energy) is tractable, e.g. via
     sparse methods validated in Stage 1, or the RL approach.

5. **Terrain and realistic environment modeling**
   - Incorporate realistic terrain (occlusion, ridge/dose-response effects
     as explored in the GameSec submission) into the 3D environment.

6. **Validation and journal write-up**
   - Validate 3D results against 2D baseline behavior as a sanity check
     (e.g., 3D results should reduce to 2D results under a degenerate/flat
     configuration).
   - Draft L-CSS + ACC 2027 paper: emphasize the progression from symbolic
     Bellman (Stage 1) through AD-based minimax (Stage 1-2) and RL
     comparison (Stage 2) to the realistic 3D extension (Stage 3) as the
     core contribution.

**Deliverables:** 3D counter-UAS SSG framework; journal paper draft for
L-CSS + ACC 2027 (deadline Sep 11, 2026).

---

## Cross-Stage Notes

- **Energy constraint** is introduced in Stage 1-1 and should propagate
  through Stages 1-2, 2, and 3 unless found to be intractable, in which
  case document why it was dropped or simplified at a later stage. Its
  form depends on resolving Option A vs. B (see Stage 1-1 open questions).
- **Literature checks** (AD-on-Bellman precedent for Stage 1-2; RL-for-
  cost-to-go precedent for Stage 2) should be done early in each respective
  stage, since they affect novelty framing for the eventual journal paper.
- **Scope**: the fixed-wing 4D model (`p1b_main.ipynb` Step 4 dynamics,
  Step 9 / `p1b_4d_dp.py`'s backward DP) is the target model for all of
  Stages 1-3. The 2D point-mass model (Steps 1-6, `glider_dynamics_pointmass`)
  is used only as a toy-scale validation step for the CasADi/AD machinery,
  not as a standalone deliverable.
- **"Stage 1-1" / "Stage 1-2" labeling** is an internal working breakdown
  for tracking purposes, not terminology the advisor necessarily uses —
  describe the actual technical work in plain terms in advisor-facing
  communication.
- Use standard academic terminology in all code comments, docstrings, and
  eventual paper text; avoid internal jargon.
