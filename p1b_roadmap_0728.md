# P1B ACC Paper Roadmap — 2026-07-28

## Purpose

This roadmap organizes the remaining work required to support the paper's central claims, numerical reliability, mathematical consistency, novelty assessment, and reproducibility. Work should proceed in Tier order. Tier 1 is blocking: later experimental conclusions must not be treated as paper evidence until the continuous evaluator and production response are validated.

## Tier 1 — Core-claim and physical-validity blockers

### 1. Diagnose `los_violation_step_11` — RESOLVED (2026-07-28)

#### Resolution implemented

The failure was classified as **accumulated grid-snapping drift**, not a
difference in the LOS formula and not merely sparse segment sampling.  Under
the legacy transition, each Bellman edge propagated the commanded action for a
fixed time and then snapped the physical endpoint to a nearby grid node.  The
next edge started from that snapped node, whereas an unsnapped replay started
from the preceding physical endpoint.  These per-edge discrepancies
accumulated until the replay crossed the LOS boundary.

A second, selectable transition model,
`successor_grid_physical_edge`, was added while preserving the legacy
`snapped_fixed_time_step` model.  Its transition construction is:

- choose the successor spatial grid node first;
- derive the flight-path angle and edge duration so the constant-speed
  kinematics terminate exactly at that successor node;
- represent the powered-to-glide switching point as an unsnapped virtual
  switching state;
- admit and evaluate terminal edges into the goal region; and
- apply terrain and LOS checks along each physical edge.

This makes the Bellman state sequence and continuous execution share the same
edge endpoints instead of resetting an approximate physical endpoint to the
grid after every step.  The solver remains a finite-grid Bellman solver; this
change establishes physical consistency of its reported edges, not continuous
optimality over all possible trajectories.

#### Verification evidence

- Production single-hill case at `z_sensor = 4000 m`: continuous replay is
  feasible and maximum edge-endpoint residual is `0 m`.
- Native-resolution two-hill case: continuous replay is feasible and maximum
  edge-endpoint residual is approximately `2.22e-16 m`.
- The legacy-mode regression still exposes snapping drift, while the
  successor-grid regression requires coincident planned/replayed endpoints.
- The full automated suite passes (`75/75`, including the Tier 1 item 2,
  visualization-timing, and result-provenance regressions added below).
- Diagnostic figure:
  `results/figures/successor_grid_transition_comparison.png`.

The notebook exposes the transition model through
`ATTACKER_TRANSITION_MODEL`, so legacy and successor-grid results can be
selected without deleting the previous implementation.

#### Original issue record

The production-resolution Bellman response is feasible under the discrete transition checks but violates the LOS constraint at step 11 when replayed without grid snapping. Two claims must be evaluated separately:

- Bellman is an exact follower oracle for the implementation-defined finite-grid problem.
- The returned path is physically feasible under continuous kinematics.

Discrete exactness does not by itself establish continuous feasibility.

#### Required diagnostics

At and before the violating step, record:

- Bellman grid state \((z_k^g,h_k^g)\);
- unsnapped kinematic endpoint \((\tilde z_{k+1},\tilde h_{k+1})\);
- accumulated continuous-replay state \((z_{k+1}^c,h_{k+1}^c)\);
- sensor position;
- terrain-clearance margin; and
- LOS margin
  \[
  m_{\mathrm{LOS}}=h-h_{\mathrm{LOS}}(z).
  \]

Confirm that `bellman.py` and `continuous_replay_evaluation.py` use the same LOS-boundary function, inequality convention, and tolerance. Compare their segment sample locations and increase `segment_check_count` systematically.

#### Classify the cause

- **Sparse segment sampling:** replace it with sufficiently dense or adaptive validation and state the tolerance.
- **Accumulated snapping drift:** explicitly analyze the difference between
  \[
  x_k^g\rightarrow\tilde x_{k+1}\rightarrow x_{k+1}^g
  \]
  in Bellman and the unsnapped continuous rollout. Increasing `segment_check_count` alone will not resolve this mechanism.
- **Intrinsic discretization artifact:** restrict the theorem to the finite-grid problem, but also recover the physical validity of reported trajectories through conservative transitions, continuous validation/repair, rejection of continuous-infeasible responses, or a common high-fidelity evaluator.

Merely adding “exact on the grid” to the proposition is not sufficient if the paper continues to report the path as a physically feasible attacker response. If continuous feasibility cannot be recovered, the study must be framed explicitly as a lattice model and must not claim continuous flight feasibility.

#### Completion gate

- Every attacker response reported in the paper passes continuous replay; or
- the discrete/continuous discrepancy is quantified and every claim and result is explicitly limited to the discrete model.
- A regression test reproduces and guards the resolved failure mode.

### 2. Separate kinematics integration from mission feasibility — RESOLVED (2026-07-28)

#### Resolution implemented

The low-level `integrate_action_sequence(...)` interface now performs only the
kinematic update and does not interpret failure to reach the goal as a
kinematic error.  Production `replay_glide_continuous(...)` retains the strict
mission rule: a nonempty action sequence that ends outside the goal radius is
infeasible.

The test suite separately verifies:

- analytic straight-line propagation through the kinematics-only integrator;
- acceptance of a synthetic action sequence that reaches the goal; and
- rejection of a kinematically valid but deliberately short sequence with
  `feasible == False`, `reached_goal == False`, and
  `violation == "action_sequence_exhausted_without_reaching_goal"`.

The focused continuous-replay suite passes `7/7`, and the complete suite
passes `75/75`.  The successor-grid solver required no additional change for
this item.

#### Original issue record

`action_sequence_exhausted_without_reaching_goal` occurs because a test intended to validate pure kinematics is coupled to the production rule “goal not reached means infeasible.”

#### Required refactor

- `integrate_action_sequence(...)`: continuous kinematics only.
- `evaluate_continuous_feasibility(...)`: terrain, LOS, airspace, and goal checks.
- `replay_glide_continuous(...)`: production composition of the two.

The production API must continue to require goal arrival by default. If `require_goal_reached=False` is retained, it must be a diagnostic-only option and must not be used to generate production results.

#### Required tests

- analytic straight-line kinematics through the low-level integrator;
- a synthetic sequence that reaches the goal and passes production replay;
- a sequence that does not reach the goal and is correctly rejected; and
- the complete test suite.

#### Completion gate

- Kinematics and business-level feasibility have separate interfaces.
- Production semantics remain strict.
- All tests pass.

## Tier 2 — Stale results and numerical reliability

### 1. Regenerate the single-hill baseline

The existing conclusion that coverage-only and Stackelberg placement are effectively identical was generated before the medium-to-fine grid correction and is stale.

After Tier 1 is complete:

- rerun `experiment_multiterrain_baselines.py` with the authoritative fine configuration;
- evaluate fixed, coverage-only, nominal-path, and Stackelberg strategies with the same configuration and follower solver;
- mark or remove the prior artifact from paper-facing results; and
- retain the conclusion only if the corrected run supports it.

### 2. Perform cross-resolution ranking-stability checks

Do not compare optima or objective values produced for different candidates by different-resolution oracles as though they were the same numerical quantity.

#### Correct protocol

1. Fix physical sensor candidates \(d_1,d_2,\ldots\).
2. Evaluate every identical sensor position at coarse, medium, and fine resolution.
3. Recompute the follower best response at each resolution.
4. When possible, cross-evaluate all discrete responses with the same high-fidelity continuous evaluator.
5. Report a common table:

| Sensor candidate | Coarse follower | Medium follower | Fine follower | Continuous replay |
|---|---:|---:|---:|---:|
| \(d_1\) | \(J_D\) | \(J_D\) | \(J_D\) | \(J_D^c\) |
| \(d_2\) | \(J_D\) | \(J_D\) | \(J_D\) | \(J_D^c\) |

Assess:

- ranking stability;
- the sign and magnitude of the objective margin;
- optimal sensor-position changes;
- switching-topology changes; and
- continuous-feasibility changes.

#### Completion gate

- The ranking of paper-critical candidates is stable across resolution; or
- instability is reported as numerical uncertainty or a tie.
- Any claimed Stackelberg improvement exceeds discretization uncertainty when measured with the same quantity.

### 3. Separate spatial- and action-grid error

Run the spatial/action-resolution factorial experiment to distinguish:

- \((z,h)\) grid refinement;
- \((v,\gamma)\) action refinement;
- their interaction; and
- if needed, sensitivity to `segment_check_count`.

Report effects on objective value, PoD, switching point, path topology, and feasibility.

### 4. Strengthen result provenance — RESOLVED (2026-07-28)

A shared `ResultProvenance` schema is now generated by
`p1b_4D/result_provenance.py` and attached to:

- every standardized notebook result-bundle manifest and the master result
  collection manifest;
- standalone Stackelberg solution exports; and
- baseline, resolution-convergence, multi-terrain convergence, and factorial
  experiment summaries, including failed-run records.

The standardized export validator rejects missing provenance and inconsistent
configuration hashes across the collection.  Regression tests verify required
fields and confirm that a resolution change changes the configuration hash.
The complete suite passes `75/75`.

Recorded fields are:

Every result summary must include:

- source commit or working-tree identifier and dirty flag;
- configuration hash;
- spatial and action resolution;
- `segment_check_count` and feasibility tolerances;
- continuous-validation status;
- script/version identifier;
- generation timestamp; and
- random seed.

## Tier 3 — Mathematical and documentation consistency

### 1. Make the normative contracts Bellman-only

The top of `stackelberg_mathematical_formulation.md` marks NLP refinement as superseded, while later sections and other specifications still describe it as mandatory.

Update the normative content directly rather than adding repeated superseded banners:

- revise the mathematical formulation;
- revise the architecture notebook;
- revise acceptance criteria and module interfaces;
- revise NLP-dependent coding and data policies;
- remove candidate filtering, Top-\(K\), warm-start, and NLP refinement from the authoritative pipeline; and
- retain NLP only as an optional offline discretization-error comparison, if used at all.

Keep migration history in one location. Code, notebooks, interfaces, and acceptance gates must describe one authoritative solver hierarchy.

### 2. Formalize the actual discrete problem in the proposition

Revise `discrete_optimality_proposition.md` as follows.

#### Powered cost

- Remove the “closed form” claim.
- Define the nine-sample trapezoidal quadrature operator as the powered-cost definition of the discrete model, not as an exact continuous integral.

#### Switching set and height mapping

- Define the precise admissible \(z\)-index set.
- State how analytic LOS-boundary height maps to the switching grid state.
- State the exact height rounding/snapping rule.

#### Terminal set and goal handling

- Define the 10 m goal radius as the formal terminal set.
- Define mid-segment goal-entry detection.
- Define the terminal fraction and fractional stage-cost rule.

#### Transition operator

- State the exact `ceil` rule for \(z\) and `round` rule for \(h\).
- Define intermediate feasibility sampling and tolerances.
- State that the snapped transition operator is distinct from unsnapped continuous dynamics.

#### Tie-breaking and unreachable states

- Explain that tie-breaking preserves the optimal value but determines the reproducible selected policy.
- Record the exact tie-breaking order.
- Assign \(+\infty\) to unreachable states and seeds and define their treatment in seed minimization.

#### Claim boundary

Use the following scope rather than the ambiguous phrase “entire discretized decision space”:

> the finite policy space induced by the implementation-defined switching set, state grid, action grid, terminal set, and snapped transition operator

Discrete value optimality and continuous physical feasibility must be separate propositions. The independent Dijkstra cross-check must validate exactly the graph defined by this finite model.

## Tier 4 — Novelty framing and literature verification

### 1. Verify related work from the original papers

Search-result titles and abstracts are insufficient for paper claims. For each relevant paper, verify from the original text:

- leader/follower order;
- defender decision: placement, scheduling, or patrol;
- attacker decision: continuous trajectory or graph path;
- terrain modeling;
- LOS masking;
- multimodal detection;
- hybrid powered–glide switching;
- follower optimality guarantee;
- continuous-feasibility guarantee; and
- leader optimization method.

Cover at least:

- sensor-scheduling/intruder-path games;
- continuous surveillance-evasion planning;
- UAV Stackelberg trajectory and security games;
- terrain-masked trajectory optimization;
- bilevel sensor placement; and
- graph/DAG follower-oracle methods.

Construct a comparison matrix before finalizing novelty claims. Use “first” only if the matrix directly supports it.

### 2. Frame the contribution around the structural combination

Avoid presenting Stackelberg games, Bellman DP, or the four-axis tensor individually as the novelty. The defensible contribution should focus on the combination of:

- terrain-induced sensing geometry;
- a hybrid powered–glide intrusion model;
- continuous sensor placement coupled to an adaptive trajectory response;
- a finite-DAG follower oracle enabled by strictly forward flight; and
- terrain-dependent strategic-placement consequences validated across resolution and continuous replay.

#### Avoid

- “novel Stackelberg game” without qualification;
- “novel Bellman algorithm”;
- “4D dynamic planner”;
- “exact continuous attacker optimum”; and
- “exact Stackelberg equilibrium.”

#### Prefer

- “state–action cost tensor”;
- “2D spatial state with discretized speed and flight-path-angle actions”;
- “\(J(z,h,v,\gamma)\) cost tensor”;
- “exact solution of the implementation-defined discrete follower problem”; and
- “continuous leader optimization with a discrete follower oracle.”

Code-internal 4D names may remain for compatibility, but paper text must not imply a four-dimensional dynamic state or full 4-DOF aircraft model.

## Tier 5 — Repository hygiene and reproducibility

### 1. Adopt a paper-artifact policy

Keep raw and large results ignored:

- NPZ arrays;
- figures;
- logs; and
- intermediate bundles.

Version small, interpretable artifacts used to generate paper tables:

- baseline summaries;
- convergence summaries;
- ranking-stability summaries;
- factorial-resolution summaries; and
- continuous-validation summaries.

Each summary must include configuration and source provenance plus the raw artifact path or checksum.

If no results are versioned, the repository must instead provide a deterministic single-command reproduction workflow, expected runtime, dependency documentation, table-generation script, and at least a smoke-level reproducibility check.

### 2. Review and commit the new validation and experiment code

Use reviewable commit boundaries:

1. **Tier 1 validation fix**
   - continuous-replay diagnostics;
   - kinematics/feasibility API separation;
   - LOS resolution;
   - regression tests; and
   - full passing test suite.
2. **Experiments and mathematical consistency**
   - regenerated baselines;
   - factorial-resolution and ranking-stability experiments;
   - proposition revision;
   - normative-document alignment; and
   - compact paper artifacts.
3. **Paper-facing terminology and related-work matrix**, if useful as a separate change.

Additional rules:

- do not track `__pycache__`;
- mark stale results explicitly;
- prevent failed or continuously unvalidated results from entering paper summaries through an export gate; and
- record a dirty flag or diff identifier for results generated from a dirty working tree.

## Required execution order

1. Diagnose continuous state drift and the LOS violation.
2. Separate kinematics integration from mission-feasibility evaluation.
3. Pass the full test suite and validate all production responses continuously.
4. Regenerate strategic baselines.
5. Run cross-resolution ranking-stability and factorial-resolution studies.
6. Revise the discrete-optimality proposition and align all normative documents.
7. Review original related-work papers and finalize the novelty boundary.
8. Finalize versioned paper artifacts and repository provenance.

## Paper-readiness gate

The work is ready to support an ACC submission only when:

- reported trajectories have an explicit and passing continuous-feasibility status;
- discrete exactness is stated only for the precisely defined finite model;
- strategic rankings are stable or their uncertainty is reported;
- baselines have been regenerated under the authoritative configuration;
- all normative documents describe the same Bellman-only pipeline;
- novelty claims are backed by original-paper comparisons; and
- every paper number is linked to a reproducible, provenance-bearing artifact.
