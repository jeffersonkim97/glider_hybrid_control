# B0 Nested-Discretization Protocol — Frozen 2026-07-29

## Status and scope

This document is the normative protocol for Direction B. It freezes the
finite-problem family before B1 implementation begins. It does not claim a
continuous optimum or an analytic continuous-time feasibility certificate.

The primary result is a nested-discretization consistency study evaluated by
one common high-fidelity replay. A relaxed lower bound is optional and may be
promoted only after the time-boxed validity/tightness gate in this document.

## 1. Fixed physical problem data

Within one terrain and sensor candidate, all resolution levels use identical:

- terrain parameters and continuous terrain-height function;
- sensor position and continuous LOS-boundary function;
- launch and goal positions;
- vehicle model, cost weights, normalization constants, and detection model;
- `goal_radius = 10 m`;
- `terrain_tolerance = 1e-6 m`;
- physical admissibility rules and deterministic tie-breaking.

No outer defender optimization is repeated in Direction B. The primary pilot
uses the final P2 two-hill candidates:

- coverage-only: `z_sensor = 1966.4609053497943 m`;
- Stackelberg: `z_sensor = 1982.9218106995881 m`.

The later terrain extension uses the final P2 candidates already recorded for
single hill and goal in valley.

## 2. Exactly nested spatial levels

For every terrain,

\[
N_{\ell+1}=2N_\ell-1,
\qquad
\Delta z_{\ell+1}=\Delta z_\ell/2,
\qquad
\Delta h_{\ell+1}=\Delta h_\ell/2.
\]

The three frozen levels are:

| Terrain | Level | Grid `(Nz,Nh)` | `dz` (m) | `dh` (m) |
|---|---|---:|---:|---:|
| single hill | L0 | `161x101` | `34.375` | `4.0` |
| single hill | L1 | `321x201` | `17.1875` | `2.0` |
| single hill | L2 | `641x401` | `8.59375` | `1.0` |
| two hill | L0 | `81x51` | `34.375` | `4.0` |
| two hill | L1 | `161x101` | `17.1875` | `2.0` |
| two hill | L2 | `321x201` | `8.59375` | `1.0` |
| goal in valley | L0 | `117x51` | `34.48275862068966` | `4.0` |
| goal in valley | L1 | `233x101` | `17.24137931034483` | `2.0` |
| goal in valley | L2 | `465x201` | `8.620689655172415` | `1.0` |

The valley family deliberately replaces the prior `467`-point refined z-grid.
Its 466 intervals cannot form an exact three-level 2:1 nesting. The new
464-interval grid changes refined `dz` by less than 0.5% while making every L0
node an L1 node and every L1 node an L2 node.

No prior valley P2/P3 objective is reused as a B result.

## 3. Frozen physical successor envelope

Let

\[
F_\ell=2^\ell,
\qquad
D_\ell=2^{\ell+1}.
\]

Thus the maximum cell offsets are:

| Level | `F_l` forward cells | `D_l` descent cells |
|---|---:|---:|
| L0 | 1 | 2 |
| L1 | 2 | 4 |
| L2 | 4 | 8 |

Because spatial spacing halves at every level, the maximum physical envelope
is invariant:

| Terrain | Maximum forward reach | Maximum descent reach |
|---|---:|---:|
| single hill | `34.375 m` | `8.0 m` |
| two hill | `34.375 m` | `8.0 m` |
| goal in valley | `34.48275862068966 m` | `8.0 m` |

This envelope is close to the existing refined P2 `3x8` envelope while using
offset counts divisible by two across all three levels.

## 4. Regular-grid action families

For grid level \(\ell\), one physical displacement is

\[
\delta x_\ell(p,q)
=
(p\Delta z_\ell,-q\Delta h_\ell).
\]

Speed is selected independently, while flight-path angle and duration are
derived from the physical endpoints:

\[
\gamma_\ell(p,q)
=
\operatorname{atan2}(-q\Delta h_\ell,p\Delta z_\ell),
\qquad
T=\|\delta x_\ell(p,q)\|_2/v.
\]

### 4.1 Enriched family — primary B family

\[
\mathcal O_\ell^{E}
=
\{(p,q):1\le p\le F_\ell,\ 1\le q\le D_\ell\}.
\]

Before vehicle feasibility filtering, the spatial-offset counts are `2`, `8`,
and `32` for L0, L1, and L2. The family is physically nested because

\[
(p,q)_\ell\mapsto(2p,2q)_{\ell+1}
\]

preserves displacement, angle, length, and duration at the same speed.

This enrichment adds intermediate reachable endpoints, directions, and edge
lengths inside a fixed physical envelope. It must be called `action-lattice
enrichment`, not pure angular refinement.

The B0 arithmetic/control audit confirms that every frozen offset is feasible
at every V5 speed under the current vehicle filter. Expected regular-graph
action counts are therefore `10`, `40`, and `160` at L0/L1/L2 for V5, and
`18`, `72`, and `288` for V9. These counts are regression expectations before
terrain/LOS state-dependent edge filtering.

### 4.2 Transported family — structural ablation

The L0 base offsets are `(1,1)` and `(1,2)`. At level \(\ell\),

\[
\mathcal O_\ell^{T}
=
\{(2^\ell,2^\ell),(2^\ell,2^{\ell+1})\}.
\]

This family preserves exactly two physical displacement vectors at every
level. It is used to isolate state/switching/goal-grid effects and to test
exact physical embedding. It is an ablation and is not required to produce a
feasible mission on every terrain.

With V5 its expected regular-graph action count is `10` at every level.

All actions remain subject to the existing speed, gamma, lift-coefficient,
drag, terrain, LOS, domain, and goal-terminal checks.

## 5. Virtual-switch successor set

Switching seeds use every z-grid node satisfying

\[
z_{start}\le z_i<z_{sensor}
\]

whose continuous LOS-boundary altitude lies inside the airspace:

\[
\sigma_i=(z_i,h_{LOS}(z_i;s)).
\]

The seed z-sets are exactly nested. No switching altitude is snapped.

The current base-index cell-window definition for virtual edges is superseded
for B. A virtual target is any grid node \(x'\) satisfying the physical box

\[
0<z'-z_\sigma\le L_z^{max},
\qquad
0<h_\sigma-h'\le L_h^{max},
\]

plus the same vehicle, terrain, LOS, domain, and finite-cost checks as a
regular physical edge. This physical-box definition ensures that every coarse
virtual target remains available at refined levels even though the switching
altitude is off-grid.

The transported virtual-target set contains mapped coarse targets only; the
enriched virtual-target set contains every grid node in the physical box.

## 6. Nested speed grids

The main B study uses

\[
\mathcal V_5
=
\{10.0,13.15,16.3,19.45,22.6\}\ \text{m/s}.
\]

The sensitivity tier is

\[
\mathcal V_9
=
\{10.0,11.575,13.15,14.725,16.3,17.875,19.45,21.025,22.6\}
\ \text{m/s},
\]

so `V9[::2] == V5` exactly. V9 is evaluated only at L1 and L2 after the main
V5 family succeeds.

`gamma_count` is not an experimental factor for the successor solver because
gamma is induced by physical endpoint offsets.

## 7. Solver quadrature and common evaluator

### 7.1 Planning-time quadrature

- Main L0/L1/L2 solves: 9-point trapezoidal edge quadrature.
- Quadrature sensitivity: repeat L2 enriched V5 with 17 points.
- Quadrature count is not changed simultaneously with a spatial level in the
  primary comparison.

### 7.2 Independent high-fidelity evaluation

Every selected powered-plus-glide policy is reevaluated with one common
high-fidelity evaluator using:

- exact physical duration profiles and no grid snapping;
- the same continuous terrain, LOS, detection, cost, and goal definitions;
- 129 equally spaced samples including both endpoints on every powered and
  glide edge;
- trapezoidal hazard integration on those 129 samples;
- terrain and LOS checks at the same 129 samples;
- fixed `goal_radius = 10 m` and `terrain_tolerance = 1e-6 m`;
- no LOS-clearance relaxation for a glide edge: before the sensor, sampled
  altitude must be at or above the continuous LOS boundary;
- reporting of minimum terrain/LOS margins, endpoint residuals, and first
  invalid samples.

This evaluator replaces the current replay's left-Riemann hazard accumulation
for B comparisons. The current replay remains a regression tool.

The 129-point evaluator must be qualified on all B2 selected policies against
a 257-point repeat. It passes qualification only if:

- feasibility and goal-reach classifications are identical;
- absolute attacker-objective difference is at most `1e-6`; and
- absolute mission-PoD difference is at most `1e-6`.

If any path fails, 257 points becomes the common evaluator and is qualified
against 513 points using the same thresholds. This is a numerical
qualification, not an analytic all-time feasibility proof.

## 8. Deterministic selection and tie breaking

The finite follower selection order is frozen as:

1. minimum planning-time mission cost;
2. only exact-equal minimum costs are treated as ties, and those ties choose
   the smallest switching-z seed, followed by seed index;
3. within a Bellman state, equal-cost actions use lexicographic construction
   order `(forward_cells, descent_cells, speed_index)`;
4. terminal goal intersections use the first positive segment-circle
   intersection.

The implementation and proposition must state the same rule. Candidate IDs
may encode the rule but may not be the normative definition.

## 9. Primary B experiment matrix

For each of the two fixed two-hill sensor candidates:

1. enriched V5, quadrature 9 at L0, L1, and L2;
2. transported V5, quadrature 9 at L0, L1, and L2 as an ablation;
3. enriched V9, quadrature 9 at L1 and L2;
4. enriched V5, quadrature 17 at L2;
5. common high-fidelity evaluation of every selected feasible policy.

The required primary consistency table is the enriched-V5 three-level family.
Transported-family infeasibility is an admissible ablation outcome and must be
reported rather than repaired by changing the frozen envelope mid-run.

After B2 passes, the same protocol is extended to the fixed final P2 candidates
for single hill and goal in valley. No result from a different grid family is
silently reused.

## 10. Reported error indicators

For candidate \(c\),

\[
e_{\ell,c}^{J}
=
|J_{\ell+1,c}^{HF}-J_{\ell,c}^{HF}|,
\]

\[
e_{\ell,c}^{\sigma}
=
\|\sigma_{\ell+1,c}-\sigma_{\ell,c}\|_2.
\]

Path comparisons use common-z altitude RMSE, maximum altitude difference,
symmetric Hausdorff distance, path-node count, and a categorical topology
change flag. Feasibility reporting includes goal reach, goal miss, first
violation, minimum terrain margin, minimum LOS margin, and maximum physical
edge-endpoint residual.

For the coverage/Stackelberg pair,

\[
M_\ell
=
J_{D,\ell}^{HF}(s_{stack})-J_{D,\ell}^{HF}(s_{coverage}).
\]

The diagnostic resolution shift is

\[
R_{1\to2}
=
\max_c|J_{D,2,c}^{HF}-J_{D,1,c}^{HF}|.
\]

Ranking is called `diagnostically resolved` only if its sign is stable and

\[
|M_2|>2R_{1\to2}.
\]

This two-times rule is an explicitly heuristic reporting rule, not an
optimality certificate. Otherwise the ordering is reported as unresolved.

## 11. Production-lattice decision for C-lite

L2 enriched is the spatial/action baseline for C-lite. Speed and planning
quadrature are selected using the B sensitivities.

Let

\[
\tau_B=\max(10^{-6},0.1R_{1\to2}).
\]

- Use V5 if the maximum L1/L2 V9-minus-V5 high-fidelity objective change is at
  most \(\tau_B\); otherwise use V9.
- Use 9-point planning quadrature if the L2 17-minus-9 high-fidelity objective
  change is at most \(\tau_B\); otherwise use 17 points.

These choices control numerical sensitivity within the finite C-lite model;
they do not certify the continuous problem.

## 12. Optional lower-bound time box

At most three working days are allocated after B1 to define and test a relaxed
follower lower bound. It may enter the main result only if:

1. a written proof establishes `J_relaxed <= J_continuous` under stated
   assumptions;
2. automated toy cases verify `J_relaxed <= J_feasible`;
3. no NaN, monotonicity, or feasibility exception is hidden; and
4. the two-hill pilot relative gap `(U-L)/max(abs(U),1e-12)` is at most 5%.

If these gates fail, the lower bound is dropped from the ACC core and B remains
a nested numerical-consistency study.

## 13. B0 acceptance gates

B0 is complete when:

- every level count, spacing, physical envelope, action set, speed set,
  switching set, quadrature, evaluator, metric, and tie-break rule is fixed;
- no primary comparison changes more than one named factor at a time;
- valley nesting is corrected explicitly and stale values are excluded;
- the virtual-switch target set is defined in physical coordinates;
- common-evaluator qualification and fallback are specified;
- production-lattice selection for C-lite is deterministic; and
- this protocol and the roadmap agree.

B1 must implement regression tests for node subset relations, physical-edge
embedding, action/speed/switching nestedness, endpoint residual, and evaluator
qualification before B2 production experiments begin.

## 14. Post-freeze B1 audit note — 2026-07-29

B1 implemented and passed all structural entry gates. The implementation also
identified a feasibility obstruction that was not visible in the B0
arithmetic audit: for the fixed two-hill Stackelberg sensor candidate, the
frozen enriched L0 and L1 families produce no switching response that reaches
the goal; L2 is feasible. The minimum available descent angles are `6.6373`,
`3.3298`, and `1.6663` degrees at L0/L1/L2, respectively. The physical virtual
target box of every L0/L1 LOS switching seed is disjoint from the regular
DAG's finite goal-reachable states.

This note records a failed B2 precondition; it does not silently alter the
frozen protocol. B2 production runs are blocked until a revised exactly
nested family is selected and documented.

The L2 evaluator pilot also triggered the prescribed sampling fallback:
129/257 gave `|delta J_A| = 3.84864e-6`, while 257/513 passed with
`|delta J_A| = 9.62142e-7` and `|delta PoD| = 5.02816e-9`. Final evaluator
qualification still applies to all B2 selected policies.

## 15. B1.5 revised feasible family — 2026-07-29

The pre-B2 revision preserves the original local enriched rectangle and adds
one exactly transported shallow backbone to the enriched family:

\[
(4,1)_{L0}\mapsto(8,2)_{L1}\mapsto(16,4)_{L2}.
\]

For single- and two-hill domains this is the same physical vector
`(137.5,-4.0) m` at every level, with descent angle `1.6663 deg`. It is the
smallest tested augmentation that both:

1. makes enriched L0 and L1 feasible at both fixed B2 sensor candidates; and
2. gives every level the same shallowest direction already present locally at
   L2 through offset `(4,1)`.

The rejected `(2,1)` L0 backbone remained infeasible at L0/L1. `(3,1)` made
the cases feasible but retained a coarser minimum angle than L2. The `(4,1)`
choice is therefore based on angle matching, not on post-hoc objective tuning.

The revised enriched offset counts are `3`, `9`, and `33`, giving V5 regular
action counts `15`, `45`, and `165`. The original local rectangular envelope
remains `34.375 m x 8 m`; the supplemental backbone extends maximum forward
reach to `137.5 m`. The enriched virtual-switch target box uses that revised
maximum physical reach. Transported-family ablations retain their original
two physical vectors and original virtual-target reach.

All node, action, speed, geometry, switching-target, and endpoint nesting
tests were rerun after this revision. Two additional feasibility regressions
cover enriched L0/L1 at both fixed B2 sensors.

## 16. B2 execution record — 2026-07-29

The complete two-sensor, nine-case-per-sensor matrix was executed with the
revised B1.5 family. Twelve enriched cases were feasible. All six transported
ablation cases were infeasible, which is an admissible outcome under Section
9 and was recorded without changing their action set.

The original 129/257 and 257/513 evaluator pairs did not satisfy the absolute
`1e-6` attacker-objective gate, although every feasibility and goal
classification agreed. The same endpoint-inclusive trapezoidal rule was
therefore continued by sample-count doubling. All twelve feasible policies
passed 1025/2049; the largest objective difference was `8.74272e-7`. The B2
common evaluator is consequently 1025 points per physical edge. This is an
extension of the numerical qualification sequence, not an analytic
continuous-time certificate.

The enriched V5/Q9 common-evaluator objectives were:

| Sensor | L0 | L1 | L2 |
|---|---:|---:|---:|
| coverage | 2.409797807 | 2.398538133 | 2.254588383 |
| Stackelberg | 3.469798211 | 3.171011091 | 3.001366624 |

Successive absolute shifts were:

- coverage: `0.011259673` and `0.143949750`;
- Stackelberg: `0.298787120` and `0.169644467`.

Thus the three-level objective shifts are not monotonically decreasing for
the coverage candidate. This numerical observation is retained explicitly;
no additional level is inferred or fabricated.

The common L2-reference defender margins were positive at every level:
`0.023949998`, `0.018420070`, and `0.019997874`. The L1-to-L2 defender
resolution shift was `0.005443445`; the frozen two-times diagnostic classified
the ordering as resolved. Maximum V9-minus-V5 sensitivity was `0.001869241`,
larger than `tau_B = 0.000544344`, so the production speed family is V9.
Q17 and Q9 selected policies had identical common-evaluator objectives for
both sensors, so production planning quadrature remains Q9.

The machine-readable result is
`results/direction_b/b2_two_hill_nested_consistency.json`.

## 17. B3 multi-terrain execution record — 2026-07-29

The frozen B2 matrix was extended to the final P2 fixed candidates for single
hill and goal in valley. The outer defender search was not repeated. The full
36-case run completed in `1034.48 s`: 30 cases were feasible. Single hill had
12 feasible enriched cases and 6 infeasible transported ablations; goal in
valley had 18 feasible cases, including all transported ablations.

All feasible policies passed the Direction-B common 1025/2049 evaluator
qualification pair. The maximum attacker-objective difference was
`5.18120e-8`, the maximum PoD difference was `2.86160e-8`, and the maximum
physical endpoint residual was `4.54748e-13 m`.

The common-evaluator enriched-V5/Q9 objectives were:

| Terrain | Sensor | L0 | L1 | L2 |
|---|---|---:|---:|---:|
| single hill | coverage | 0.674388957 | 0.544707020 | 0.535539594 |
| single hill | Stackelberg | 0.674465576 | 0.544718126 | 0.535540156 |
| goal in valley | coverage | 0.913991916 | 0.369200570 | 0.323638614 |
| goal in valley | Stackelberg | 0.903246377 | 0.367479675 | 0.323403740 |

Single-hill defender margins were positive but unresolved:
`5.07757e-5`, `1.67218e-5`, and `7.36917e-6`, against an L1-to-L2 resolution
shift of `0.011579167`. Goal-in-valley margins were negative but also
unresolved: `-0.002088802`, `-0.001230536`, and `-0.000224653`, against a
resolution shift of `0.034359843`. The valley result records sensitivity of
the fixed P2 candidates to the B grid family; it is not a resolved continuous
ranking reversal.

For both added terrains, V5/V9 and Q9/Q17 selected-policy objectives were
identical under the common evaluator. The B4 global choice must still combine
these terrain-specific sensitivities with B2, which selected V9/Q9.

The machine-readable result is
`results/direction_b/b3_multiterrain_nested_consistency.json`. The associated
trajectory, consistency, and evaluator figures are stored under
`results/direction_b/figures/b3_*.png`.

## 18. B4 production-lattice freeze — 2026-07-29

B4 freezes one follower configuration for every finite C-lite leader
candidate. The machine-readable configuration ID is
`direction_b_l2_enriched_v9_q9_e1025`:

- L2 position grid for each terrain;
- enriched 33-direction physical movement set;
- V9 speed set, giving 297 regular direction/speed choices per state;
- Q9 planning-time edge quadrature;
- 1025-sample common continuous replay;
- `successor_grid_physical_edge` with no endpoint snapping.

V9 is retained globally because the two-hill speed sensitivity exceeded its
terrain tolerance by a factor of about `3.43`; single hill and goal in valley
did not require the extra speeds individually. Q9 is retained because Q17 did
not alter any selected policy or common-evaluator objective. The common replay
count remains 1025 because it is the maximum requirement across B2 and B3.

The production factory is
`build_direction_b_production_configuration`. It marks the configuration as
frozen, records the intended finite-C-lite use, and explicitly sets both
continuous-optimum claims to false.

C-lite must re-enumerate its complete stated finite sensor set. The P2 sensor
positions used for B2/B3 diagnostics are not reused as C-lite optima.

The freeze manifest is
`results/direction_b/b4_production_lattice_freeze.json`; its SHA-256 is
`62b573c6a029067fb68d52dde471bd93cf2c990e5f539665ec797f5e3685da44`.
The decision visualization is
`results/direction_b/figures/b4_production_lattice_freeze.png`.
