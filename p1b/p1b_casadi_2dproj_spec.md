# P1b: `p1b_casadi_2dproj.ipynb` Implementation Spec (for Claude Code)

## Mission Statement

Build a new Jupyter notebook `p1b_casadi_2dproj.ipynb` implementing the
**advisor-confirmed architecture**:

```
CasADi symbolic 4D stage-cost map (z, h, v, gamma; action = gamma_cmd)
  -> 2D projected stage-cost map: softmin over (v, gamma) at each (z, h)
  -> 2D projected cost-to-go map: J(z_goal, 0) = 0, Bellman propagation
     with fixed-wing template displacements
```

The entire pipeline must be differentiable w.r.t. the sensor position
`z_sensor` (scalar CasADi symbol) via reverse/forward-mode AD. This is the
foundation for Stage 1-2 (gradient-based Stackelberg solve, "Goal 3", not in
this file's scope).

**Do NOT modify** `p1b_main.ipynb` or `p1b_casadi_symbolic.ipynb`. The old
`p1b_casadi_symbolic.ipynb` is archived/frozen — reuse code FROM it (copy),
never import or edit it.

## Goals

- **Goal 1** (Phases 0-4): fixed defender at `Z_SENSOR = 2000`. Output =
  attacker's optimal trajectory/strategy + two named plots (see Phase 4).
- **Goal 2** (Phase 5): defender sweep `z_sensor in [1450, 2250]`
  (Stackelberg enumeration version). Static plots at 3 positions, timing
  table, ground-truth payoff curves, and an mp4 animation.

## Hard Conventions (apply to every phase)

1. **COMPUTE / PLOT cell split.** Every visualization lives in its own PLOT
   cell, separate from the COMPUTE cell that produces its data. COMPUTE
   cells save to `data/*.json` or `data/*.npz`; PLOT cells load from disk
   (or reuse in-memory variables if the save/load round trip is
   prohibitively large — but the cell split is mandatory regardless).
   PLOT cells never compute anything beyond trivial reshaping/masking.
2. **USER VISUALIZATION GATE at the end of every phase.** After each
   phase's final PLOT cell, insert a markdown cell:

   ```
   ---
   ## [GATE] Phase N complete — STOP. User visualization check required.
   Do not proceed to Phase N+1 until the user has reviewed the figures
   above and explicitly approved. List of figures to review:
   - <figure filenames>
   Questions for the user (if any): <...>
   ---
   ```

   When executing this spec interactively, Claude Code must actually stop
   at each gate, report results + figure paths, and wait for user approval
   before writing/running the next phase's cells.
3. **AD discipline:** gradients w.r.t. the scalar `zsens_sym` use
   `ca.jtimes(expr, zsens_sym, ca.DM(1.0), False)` (forward mode, one
   pass). **Never call `ca.jacobian` on a large output vector** — this
   caused a 920s+ blowup in the archived notebook. Add this as a code
   comment wherever a gradient is computed.
4. **Stage cost must accumulate hazard over RK4 substeps** (see cell 2.2).
   Single-point stage-cost evaluation is forbidden — it was the leading
   suspect for the systematic z-dependent residual in the archived
   notebook.
5. **No closures over global symbols.** `bellman_backup(J, zsens)` and all
   symbolic constructors take symbols as explicit arguments (the archived
   notebook needed a `ca.substitute` hack because stage-cost expressions
   closed over a global `zsens_sym`; do not repeat this).
6. **Sparse everywhere:** all interpolation/shift matrices are built as
   `scipy.sparse.csc_matrix` -> `ca.DM(sparse)`. Never a dense
   intermediate. Print an estimated memory footprint before construction.
7. Matplotlib only; every figure saved to `data/*.png` at dpi=120 with
   `bbox_inches='tight'`, then `plt.show()`.
8. Reuse the parameter file `data/params.json` produced by
   `p1b_main.ipynb`. Assert (not silently override) any value this
   notebook must pin: `Z_GOAL == 2500`, `Z_SENSOR == 2000`.

## Global Constants (define once in cell 0.1)

| Name | Value | Notes |
|---|---|---|
| `Z_MAX_GRID` | 3000.0 | grid extends past Z_GOAL by 500m (> max stage displacement ~183m) — prevents the edge-cliff artifact without moving the goal |
| `Z_GOAL` | 2500.0 | from params.json, assert equality |
| `Z_SENSOR_DEFAULT` | 2000.0 | Goal-1 defender position |
| `N_Z, N_H` | 101, 100 | dz ~ 30m, dh ~ 3m |
| `v_grid` | [14, 30, 55, 100] | matches archived Phase 2 |
| `g_state_grid` | linspace(-90, +90, 37) | state gamma, deg |
| `a_grid` | linspace(-90, 0, 19) | action gamma_cmd, deg |
| `SOFTMIN_TAU` | placeholder 0.05, finalized in cell 3.4 | logsumexp temperature |
| `OCC_WIDTH` | single scalar, placeholder 5.0, tuned in cell 3.4b | ONE width for both detection zeroing and domain exclusion (the archived notebook had 10 vs 0.1 — do not repeat) |
| `LARGE` | 50.0 | unreachable sentinel; add `assert LARGE > 3 * (expected max J)` once reference J is known |
| `ZS_SWEEP` | arange(1450, 2250 + 1, 50) | Goal-2 sweep, 17 points |
| `ZS_STATIC` | [1450, 1850, 2250] | Goal-2 static plots |

`h_sensor` is **not** a constant: `h_sensor_sym = h_terrain_sym(zsens_sym)`
(terrain-following defender). At `zsens=2000` this is ~0 (Gaussian tail),
matching the old setup; at `zsens=1450` it is ~60m. All sensor-relative
geometry must use the symbolic `h_sensor_sym`.

---

## Phase 0 — Setup & Numeric Reference

### [0.1] COMPUTE: imports, params, grids
- Load `data/params.json`; assert `Z_GOAL==2500`, `Z_SENSOR==2000`.
- Define all Global Constants above. Print a one-line summary of grid
  sizes and total 4D combo count `N_Z*N_H*len(v_grid)*37` and transition
  combo count `len(v_grid)*37*19 = 2812`.
- **Acceptance:** cell runs, prints grid summary, no reference to the
  archived notebook's variables.

### [0.2] COMPUTE: numeric reference (new architecture, new grid)
- Do **not** load `p1b_4d_dp_results.npz` (old grid, old architecture).
- Implement the same pipeline in **plain numpy with hard min** as the
  ground truth: template transitions (same as cell 2.1), 4D stage cost
  with substep hazard accumulation (same math as 2.2, numpy), hard-min
  projection over (v, gamma) per (z, h, action), then 2D value iteration
  (Jacobi is fine; iterate until `max|dJ| < 1e-9` or 500 sweeps).
- Save `data/ref_2dproj.npz`: `J_2D_ref`, grids, terrain/occlusion masks,
  sweep count used, `max|dJ|` history.
- **Acceptance:** converged (`max|dJ|` below tol), finite-cell count > 0,
  J range printed.

### [0.3] PLOT: reference sanity
- Heatmap of `J_2D_ref` + terrain fill + goal star + sensor marker.
- Save `casadi2d_phase0_reference.png`.

### [GATE] Phase 0 — user check
Figures: `casadi2d_phase0_reference.png`. Question for user: does the
reference cost-to-go field look qualitatively right (low near goal,
growing toward launch, occlusion region excluded)?

---

## Phase 1 — Symbolic Building Blocks

### [1.1] COMPUTE: terrain & LOS symbolic
- `h_terrain_sym(z)` (Gaussian, same params).
- `zsens_sym = ca.MX.sym('zsens')`; `h_sensor_sym = h_terrain_sym(zsens_sym)`.
- `z_tangent` via `ca.rootfinder('newton')` on the tangency residual
  (port from archived Phase 1, but with symbolic `h_sensor_sym` replacing
  the old constant `h_sensor_c`). LOS slope/intercept as functions of
  `zsens_sym`.
- `los_validity(zs) -> bool` numeric helper: tangent point exists,
  `0 < z_tangent < zs`, slope finite and negative-side sane. Used later to
  guard every frame of the Goal-2 sweep.
- Validate `z_tangent(2000)` against the numeric tangent from the
  reference computation (diff < 1m).
- **Acceptance:** tangent match at zs=2000; `los_validity` returns True for
  all `ZS_SWEEP` values (print the table). If any sweep point fails, STOP
  and report — do not silently skip.

### [1.2] COMPUTE: sensor lambda symbolic
- Port `stage_lambda_sym` from the archived notebook including the
  **sqrt-floor-before-sqrt fix** (floor `r2` before `sqrt`, never
  `fmax(sqrt(r2), R)`). Doppler + RCS, glide phase.
- `dh = h_sensor_sym - h` (symbolic now — double-check every place the old
  code used the constant).
- Occlusion weight `occlusion_weight_sym` with the single `OCC_WIDTH`.
- Numeric spot-check vs the reference lambda at 3 sample points (rel diff
  < 1e-9 with occlusion weight ~0 or ~1 sample locations).

### [1.3] COMPUTE: building-block AD unit tests
- For `z_tangent(zsens)` and `lambda(z0,h0,v0,g0; zsens)` at fixed sample
  points: `jtimes` vs central FD (eps=1.0 and 0.1). Report max rel error;
  require < 1e-4.
- **Acceptance:** both blocks pass AD-vs-FD before any assembly.

*(No PLOT cells in Phase 1 — the gate reviews printed diagnostics.)*

### [GATE] Phase 1 — user check
No figures; paste the tangent-validation table, the sweep-validity table,
and AD unit-test errors. Wait for approval.

---

## Phase 2 — Template Transitions & Symbolic 4D Stage Cost

### [2.1] COMPUTE: template transitions (numpy, zsens-independent)
- Port the archived Phase 2 template rollout: for each of the 2812
  `(v, gamma_state, action)` combos, RK4 substep rollout with
  `glider_dynamics` (closed-loop `required_CL`), storing:
  - `dz_cs[iv,ig,ia,k], dh_cs[iv,ig,ia,k]`: cumulative substep displacements
  - `gamma_sub[iv,ig,ia,k]`: **actual** gamma at each substep (deg) — needed
    by 2.2 so the sensor model sees the true attitude, not the command
  - `v_sub[iv,ig,ia,k]`: actual speed at each substep — same reason
  - `v_next, g_next`, stall/out-of-range validity
- Save to `data/templates_2dproj.npz`.
- **Acceptance:** print valid-combo fraction; ~consistent with archived
  notebook's numbers; wall time printed.

### [2.2] COMPUTE: symbolic 4D stage cost — substep hazard accumulation
- For each `(z, h)` grid cell and each valid `(v, g, a)` combo, the stage
  hazard is
  `H = sum_k lambda(z + dz_cs[k], h + dh_cs[k], v_sub[k], gamma_sub[k]; zsens_sym) * DT_GLIDE`
  and the stage cost is `C = 1 - exp(-H) + W_TIME * DT_DP`.
- **Forbidden:** evaluating lambda once at the cell's start point.
- Vectorize over the flattened `(z,h)` grid per combo (the archived
  notebook's per-action vectorization pattern generalizes). Watch graph
  size: 2812 combos x N_SUB substeps of lambda evaluations — print an
  estimated node count before building; if it is clearly infeasible,
  report options to the user (e.g., reduce N_SUB in cost only, or
  midpoint-rule 2-sample hazard) **at the gate, do not decide alone**.
- Terrain-violating substeps: combo invalid at that cell (mask, consistent
  with the reference in 0.2).

### [2.3] COMPUTE: 2D projected stage-cost map (for the record + Fig)
- `c_2D(z,h) = softmin over (v, g, a)` of the stage cost (numeric eval at
  `Z_SENSOR_DEFAULT`), plus the hard-min version for comparison.
- Save `data/costmap_2dproj.npz`.

### [2.4] PLOT: 2D projected stage-cost map
- Two-panel: hard-min vs softmin stage-cost heatmap at zs=2000, terrain
  fill, LOS line. Save `casadi2d_phase2_costmap.png`.

### [GATE] Phase 2 — user check
Figures: `casadi2d_phase2_costmap.png`. Include the graph-size estimate
and any feasibility concerns from 2.2 as explicit questions.

---

## Phase 3 — Softmin Projection + 2D Bellman + Validation

### [3.1] COMPUTE: shift operators W
- Displacements are translation-invariant, so each valid `(v,g,a)` combo
  has a constant 4-corner bilinear **shift matrix** `W_{vga}` on the 2D
  grid (n_cells2D x n_cells2D, ~4 nnz/row). Build as sparse -> `ca.DM`.
- Cells whose shifted target leaves the grid or whose substep path hits
  terrain: excluded via a validity mask per combo (consistent with 0.2).
- Print total memory estimate BEFORE building; if > ~2GB, prune invalid
  combos first and report the reduction.

### [3.2] COMPUTE: Bellman backup definition
- Signature: `bellman_backup(J, zsens)` — pure function of its arguments.
- Per action `a`: `Q_a(z,h) = softmin_{v,g} [ C(z,h,v,g,a) + (W_{vga} @ J)(z,h) ]`
  using logsumexp with `SOFTMIN_TAU`, computed stably
  (`-tau * log(sum(exp(-(x - x_min_detached)/tau))) + x_min_shift` — use the
  standard shifted-logsumexp; the shift can be a `ca.fmin` chain, which is
  fine inside logsumexp for stability).
- Across actions: **hard `ca.fmin` chain by default**; keep a boolean
  `SOFTMIN_OVER_ACTIONS` toggle (default False).
- Then, in order: terrain override (multiplicative mask to LARGE),
  occlusion domain-exclusion (smooth multiplicative with
  `occlusion_weight_sym`, `OCC_WIDTH`), goal pin (`J[goal]=0`).
- Invalid-combo handling inside softmin: add `LARGE * invalid_mask` to the
  argument BEFORE logsumexp so invalid entries contribute ~exp(-LARGE/tau)
  ~ 0 (verify this doesn't underflow into NaN; if it does, use masked
  logsumexp instead).

### [3.3] COMPUTE: mapaccum chain + convergence
- Wrap backup as `ca.Function`, chain with `mapaccum`.
- Convergence criterion: `max|J_N - J_{N-10}|` over both-finite cells
  (NOT finite-cell count — that plateaus early and is a false signal).
- Choose `N_SWEEPS` = first checkpoint with `max|dJ| < 1e-8`, plus 10%
  margin. Print the convergence table; save the per-checkpoint deltas.

### [3.3b] PLOT: convergence curves
- Semilog `max|dJ|` vs N, and finite-count vs N (two panels).
  Save `casadi2d_phase3_convergence.png`.

### [3.4] COMPUTE: 3-way validation + tau selection
- (i) hard-min symbolic (SOFTMIN_TAU -> tiny, or an fmin-chain variant)
  vs `J_2D_ref` from 0.2 -> reconstruction accuracy. Target:
  `mean|diff| < 0.05`, `max|diff|` explained (report outlier cells).
- (ii) softmin vs hard-min symbolic -> temperature bias, quantified per
  tau in {0.01, 0.05, 0.1, 0.2}.
- (iii) for each tau: AD gradient `dJ_2D/dzsens` magnitude statistics near
  the occlusion boundary (the trade-off: small tau ~ exact values but
  jumpy gradients; large tau ~ smooth gradients but biased values).
- Output a small table (tau, mean bias, max bias, boundary-gradient
  L2). **Propose a tau; the user confirms at the gate.**

### [3.4b] COMPUTE: OCC_WIDTH trade-off scan
- For OCC_WIDTH in {0.5, 2, 5, 10}: (a) value error vs reference,
  (b) gradient magnitude in a band ±2 grid cells around the LOS boundary.
- Same deal: propose, user confirms at gate. (Archived-notebook lesson:
  width tuned for value accuracy alone killed the gradient pathway.)

### [3.5] COMPUTE: full AD check
- `ca.jtimes` (forward) for `dJ_2D/dzsens` at zs=2000; central FD eps=1.0.
- Report max/mean |AD-FD| over reachable cells, wall time for the AD pass
  (this is the first Goal-3 feasibility datapoint — record it in a
  markdown note).

### [3.6] PLOT: validation panels
- 3 panels: symbolic J_2D | reference J_2D | diff (RdBu, symmetric).
  Save `casadi2d_phase3_validation.png`.
- Separate figure: AD gradient field heatmap `dJ_2D/dzsens`.
  Save `casadi2d_phase3_gradfield.png`.

### [GATE] Phase 3 — user check
Figures: convergence, validation, gradient field. Decisions requested
from the user: final `SOFTMIN_TAU`, final `OCC_WIDTH` (present the two
trade-off tables). Do not proceed to Phase 4 without these confirmed.

---

## Phase 4 — Goal 1 Outputs (defender fixed at zs=2000)

### [4.1] COMPUTE: switching-point selection
- Candidates on the LOS tangent line, occlusion side:
  `z_sw in linspace(z_margin, z_tangent - z_margin, ~19)` with
  `h_sw = slope * z_sw + intercept`; drop candidates with
  `h_sw > H_MAX_GRID` or below terrain.
- Powered phase: straight line origin -> (z_sw, h_sw) at `V_LAUNCH`;
  `t_powered = sqrt(z_sw^2 + h_sw^2) / V_LAUNCH`. Powered detection cost
  = 0 (entirely in occlusion; acoustic zeroed by convention — note this
  in a markdown comment as a model convention pending advisor review).
- Total: `J_total(z_sw) = W_TIME * t_powered + bilinear_interp(J_2D)(z_sw, h_sw)`.
- Select `z_sw* = argmin J_total`. Save the full candidate table
  (`data/goal1_switching.json`).

### [4.2] COMPUTE: policy extraction + forward simulation
- Per-cell best action from the converged 2D map (argmin over actions of
  `C_a + W_a @ J` evaluated numerically at zs=2000).
- Forward-simulate with the real `glider_dynamics` from
  `(z_sw*, h_sw*, V_LAUNCH, gamma0 = atan2(h_sw*, z_sw*)...)` — entry
  gamma is the powered-phase line angle. NOTE (markdown): entry-state vs
  best-case-(v,gamma) mismatch is a **known, accepted approximation** of
  the 2D-projected architecture; if the sim misses the goal, plot the path
  in red and report — do not silently fix.
- Also simulate from every candidate (cheap) to color-code reachability.
- Save `data/goal1_paths.json`.

### [4.3] PLOT: "2D Projected cost-to-go heatmap plot"  (canonical name)
- Content, exactly: `J_2D` heatmap (viridis) + **terrain fill in brown** +
  **LOS tangent line dashed green** + **occlusion zone overlay red,
  alpha=0.25** + **LOS zone overlay green, alpha=0.25**. Goal star,
  sensor marker.
- Implement as a reusable helper
  `plot_costtogo_with_overlays(ax, J_2D_vals, zs_val, ...)` — Phase 5
  reuses it verbatim.
- Save `casadi2d_goal1_costtogo.png`.

### [4.4] PLOT: "2D Projected attacker strategy plot"  (canonical name)
- Same base as 4.3 via the helper, PLUS: powered straight line
  (origin -> z_sw*, distinct color, e.g. orange, solid), optimal glide
  path (z_sw* -> goal, white; red if goal not reached), switching-point
  marker, non-selected candidates as faint dots.
- Save `casadi2d_goal1_strategy.png`.

### [GATE] Phase 4 — user check  (== Goal 1 sign-off)
Figures: `casadi2d_goal1_costtogo.png`, `casadi2d_goal1_strategy.png`.
This gate is the Goal-1 acceptance review. Do not start Phase 5 without
explicit approval.

---

## Phase 5 — Goal 2: Defender Sweep (Stackelberg enumeration)

### [5.1] COMPUTE: `solve_for_zsensor(zs)` wrapper
- Build the symbolic graph ONCE (already built in Phase 3); this wrapper
  only **evaluates** compiled `ca.Function`s at a new `zs`.
- Returns dict: `zs, h_sensor, los_valid, J_2D (ndarray), z_sw*, h_sw*,
  J_total*, path (z,h arrays), reached_goal, P_D (path hazard integral ->
  1-exp(-H_path)), t_wall_eval`.
- `P_D` computation: along the simulated optimal path (powered + glide),
  accumulate `lambda * DT_GLIDE` per substep (powered contributes 0 by
  convention), then `P_D = 1 - exp(-cum_hazard)`. **Record J_total and
  P_D separately** — defender payoff is NOT -J (non-zero-sum; J contains
  the W_TIME term).
- Guard with `los_validity(zs)`; on failure, mark the frame invalid and
  report (should not happen for zs >= 1450 per Phase 1 check).

### [5.2] COMPUTE: sweep execution
- Run `solve_for_zsensor` over `ZS_SWEEP` (17 points). Save everything to
  `data/goal2_sweep.npz` (**required — Goal-3 input**): per-zs `J_total*`,
  `P_D`, `z_sw*`, timings, plus the three static-case full results.

### [5.3] PLOT: static strategy plots x3
- For zs in `ZS_STATIC = [1450, 1850, 2250]`: one figure each, the
  "2D Projected attacker strategy plot" via the Phase-4 helper.
- Save `casadi2d_goal2_strategy_zs{1450,1850,2250}.png`.

### [5.4] COMPUTE+markdown: timing table
- Markdown table: one row per static case (eval wall time, path-sim time),
  plus a separate row for the one-time graph-build cost (from Phase 3).
  Also save `data/goal2_timings.json`.

### [5.5] PLOT: ground-truth payoff curves
- Two-panel (shared x = zs): `J_total*(zs)` and `P_D(zs)`, markers at the
  3 static positions. Annotate the argmax of `P_D` (enumeration-Stackelberg
  defender optimum).
- Save `casadi2d_goal2_groundtruth.png`.

### [5.6] COMPUTE: animation frames + encode
- 17 frames (zs = 1450 -> 2250 step 50). Each frame: the strategy plot
  via the helper + current-zs marker on a small `P_D(zs)` progress inset
  (bottom-right).
- Encode at **10 fps, mp4** via `matplotlib.animation.FFMpegWriter`; if
  ffmpeg is absent, fall back to `imageio` + `imageio-ffmpeg`
  (`pip install imageio imageio-ffmpeg` allowed). Save
  `data/goal2_sweep_animation.mp4`. Also keep the individual frame PNGs
  in `data/frames/` (cheap insurance if encoding fails).
- Consistent color scale across frames: fix vmin/vmax from the sweep-wide
  J range so the heatmap doesn't rescale per frame.

### [GATE] Phase 5 — user check  (== Goal 2 sign-off)
Deliverables: 3 static PNGs, timing table, ground-truth PNG, mp4.
End of this file's scope.

---

## Final markdown cell — Goal 3 prep notes (no code)

- Record: is `P_D(zs)` non-convex over the sweep? Where are the local
  maxima? (These become multi-start brackets for the gradient-based
  Stackelberg solve.)
- Record the Phase-3.5 AD wall time as the feasibility datapoint.
- Sketch (text only): differentiable `P_D` will need a parallel
  hazard-to-go recursion under the soft-optimal policy — first task of
  Goal 3.

---

## Reuse Map (from `p1b_casadi_symbolic.ipynb`, copy-not-import)

| Reuse as-is | Port with changes | Do NOT port |
|---|---|---|
| rootfinder z_tangent pattern | `stage_lambda_sym` (h_sensor now symbolic) | 4D Bellman machinery (Phase 1.5/2) |
| sqrt-floor fix | stage cost -> substep hazard accumulation | reach-weighted conditional-mean backup |
| sparse `ca.DM` + mapaccum pattern | occlusion weight -> single OCC_WIDTH | dual-width occlusion (10 / 0.1) |
| jtimes forward-mode AD pattern | backup -> pure function, no closures | `ca.jacobian` on large outputs |
| Phase-0 loader/validation harness | template rollout (add gamma_sub, v_sub) | finite-cell-count convergence criterion |

## Known Pitfalls Checklist (from the archived notebook — verify each)

- [ ] No `fmax(sqrt(r2), R)` — floor r2 first.
- [ ] No single-point stage lambda — substep accumulation.
- [ ] No `ca.jacobian` on vector outputs — jtimes only.
- [ ] No closures over global symbols in backup/stage-cost builders.
- [ ] Convergence judged by max|dJ|, not finite-cell count.
- [ ] One OCC_WIDTH, tuned with BOTH value error and gradient magnitude.
- [ ] LARGE sentinel margin asserted against actual J range.
- [ ] Sparse matrices only; print memory estimate before building.
- [ ] Grid/goal consistency: Z_GOAL=2500 everywhere, asserted from params.json.
- [ ] Every plot in its own PLOT cell; every phase ends with a user gate.
