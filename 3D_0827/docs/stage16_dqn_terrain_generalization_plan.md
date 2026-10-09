# P1b Bellman DP vs RL Approximation — Phased Implementation Plan

## Project Objective

Determine the computational conditions under which the P1b attacker best-response (BR) subproblem should use:

- Bellman DP as the exact discrete baseline,
- Tabular Q-learning as a terrain-specific RL approximation baseline, or
- DQN as a terrain-generalized RL approximation.

The comparison is performed inside the existing Local Strong Stackelberg Equilibrium (Local SSE) framework. Only the attacker BR solver is replaced; the remaining Stackelberg framework is kept unchanged unless explicitly required for interface compatibility.

Primary computational-condition variables:

- Spatial discretization: `dx = dy = dh`
- Angular discretization: `dpsi = dgamma`
- Local SSE defender neighbor-search radius: `r` using Chebyshev distance on the discrete defender plane

Primary performance outputs:

- Attacker-BR computation time
- Total Local SSE computation time
- Final Local SSE attacker objective `J_A`
- Final Local SSE defender objective `J_D`
- Bellman-relative `J_A` / `J_D` approximation error or accuracy

Hardware utilization (CPU, RAM, GPU utilization) is **not** a primary comparison metric in the current study design. GPU may be used for DQN training in later phases, but GPU utilization itself is not part of the required performance comparison unless explicitly added later.

Final evaluation uses unseen, more complex terrain instances and Monte Carlo trials.

---

## Global Repository-Grounding Rule — Applies to Every Phase

For **all phases in this document**, the current local repositories `3D_0827/` and `3D_RL_DQN/` are the authoritative implementation sources. Whenever a phase requires an existing code-related value, implementation detail, path, function, class, configuration, hyperparameter, numerical constant, state/action definition, objective term, dynamics rule, feasibility rule, sensor model, terrain parameter, result convention, or entry point:

1. Inspect the current relevant code first.
2. Reuse the actual current implementation/value when it exists.
3. Do **not** infer or guess it from this planning document, old uploaded snapshots, prior chat descriptions, or typical/default RL values.
4. If the required information is absent or ambiguous in the current repository, mark it as unresolved and report the exact ambiguity instead of silently inventing a value.
5. Any genuinely new design value introduced by a later phase must be explicitly identified as a **new design choice**, separated from values inherited from the existing code, and documented in that phase's manifest/configuration.

This rule overrides any example number or conceptual notation in the planning document whenever the actual repository differs.

---

# Phase Structure

## Phase 0 — Freeze and Validate Existing Baselines

### Objective
Freeze the current Bellman DP, Tabular Q-learning, attacker BR, and Local SSE implementation as the reproducible reference baseline before any DQN work begins.

Phase 0 is a **development / regression-validation phase**, not the final publication performance experiment. The cube case and visualizations in this phase exist to verify that the current code is discovered correctly, can be executed reproducibly, and can later be compared through a common interface. Do not interpret Phase 0 runs as the final Monte Carlo evidence for the paper.

### Repository / Project Layout

- Existing Bellman DP and Tabular Q-learning code: `3D_0827/`
- New DQN development code in later phases: `3D_RL_DQN/`
- The latest local repository must be treated as authoritative. Any old uploaded or copied `3D_0827` snapshot is potentially outdated and must **not** be used to infer exact hyperparameters, entry points, numerical constants, or current file organization.
- `3D_0827/` and `3D_RL_DQN/` must be able to access the required shared solver/environment functionality through importable interfaces. Establish import compatibility without duplicating Bellman or Tabular solver logic. Avoid circular imports. Prefer a minimal package/path/adapter solution after inspecting the actual repository structure.

### Fixed Baseline Test Case

Use the same **cube terrain test case currently used by the existing Tabular Q-learning comparison** as the Phase 0 canonical visualization/smoke-test case.

Codex must discover this case from the current code rather than reconstructing it from this document. Preserve its existing:

- terrain geometry,
- attacker initial condition,
- target / terminal condition,
- defender / sensor configuration,
- objective definitions,
- dynamics and feasibility rules.

Do not redesign or simplify the cube case in Phase 0.

### Known Computational-Condition Domain

The project will later study the following discretization/search conditions:

- Spatial discretization `dx = dy = dh`:
  - `100 m`
  - `50 m`
  - `25 m`
  - `12.5 m`
  - `10 m`
  - `5 m`
- Angular discretization `dpsi = dgamma`:
  - `1.25 deg`
  - `2.5 deg`
  - `5 deg`
  - `10 deg`
- Local SSE Chebyshev neighbor-search radius `r`:
  - `1, 2, 3, 4, 5`

These values define the intended experimental domain. **Phase 0 does not need to run the exhaustive Cartesian sweep.** The Phase 0 baseline runner/configuration must be capable of accepting these parameters, but the canonical smoke test should use the actual discretization values already associated with the current cube Tabular/Bellman test unless an existing baseline script explicitly defines another reference configuration.

### Existing-Code Discovery Requirements

Before modifying implementation code, inspect the **current local repository** and produce a short discovery record containing:

1. Whether a runnable Bellman / Tabular / Local-SSE entry point already exists.
2. The actual file(s), function(s), and command(s) used to execute:
   - Bellman attacker BR,
   - Tabular Q-learning attacker BR,
   - Local SSE using those BR results.
3. The current cube baseline configuration source.
4. The current Tabular Q-learning hyperparameters and training settings, including all values actually used by the current implementation, such as:
   - episode count or stopping rule,
   - learning rate,
   - discount factor,
   - epsilon/exploration schedule,
   - random seed handling,
   - Q-table initialization,
   - any warm start, convergence, or policy-extraction rule.
5. The current result-directory convention and existing output naming convention.
6. The current functions used to compute `J_A` and `J_D`.

Do **not** guess missing values. If a value is not explicitly present in the code/configuration, record it as unresolved rather than inventing one.

### Entry-Point Requirement

If a suitable existing entry point already exists, reuse it with the smallest possible wrapper/adaptation.

If no suitable entry point exists, create a minimal Phase 0 runner that:

1. loads the existing cube case,
2. runs the existing Bellman attacker BR without changing its algorithm,
3. runs the existing Tabular Q-learning attacker BR without changing its algorithm,
4. can invoke the existing Local SSE pipeline for each BR method,
5. records the required metrics and configuration in JSON, and
6. generates the required validation visualizations.

The runner must call the existing solver/objective implementations rather than copy their internal logic.

### Objective-Function Freeze

Use the existing `J_A` and `J_D` definitions exactly as implemented in the current Bellman / Tabular pipeline.

Do not modify:

- objective terms,
- weights,
- normalization,
- sign convention,
- sensor-risk calculation,
- mission-time contribution,
- Stackelberg tie-breaking behavior.

Bellman and RL must be evaluated using the same existing `J_A` and `J_D` computation.

### Runtime Instrumentation

For each solver mode, record exactly two official runtime quantities in Phase 0:

1. `attacker_br_runtime_sec`
2. `local_sse_total_runtime_sec`

Measurement definitions:

- **Bellman attacker-BR runtime:** wall-clock time required by the existing Bellman attacker-BR computation for the evaluated BR call.
- **Tabular Q-learning attacker-BR runtime:** wall-clock time covering the complete case-specific RL attacker-BR procedure, including its required training and final policy/trajectory query needed to return the BR.
- **Local-SSE total runtime:** wall-clock time from the start of the existing Local SSE solve until the final Local SSE result is returned for the selected BR method.

Do not add subsection-level profiling in Phase 0. That may be added later.

Do not use CPU utilization, RAM utilization, or GPU utilization as official Phase 0 comparison metrics.

### Required Result Data

For each Phase 0 run, preserve the existing result-folder convention and write machine-readable JSON containing at least:

```text
phase
solver_method                  # bellman or tabular_q
terrain_case                   # discovered canonical cube-case identifier
spatial_discretization_m       # dx = dy = dh
angular_discretization_deg     # dpsi = dgamma
neighbor_radius_r
random_seed                    # if applicable / available
attacker_br_runtime_sec
local_sse_total_runtime_sec
J_A
J_D
status
```

Also save a baseline/discovery manifest in JSON containing the resolved entry points, current Tabular Q-learning settings, objective-function source locations, and relevant configuration-file locations.

If the existing result convention already contains equivalent fields, preserve that convention instead of creating redundant parallel schemas; add only the missing Phase 0 fields.

### Required Visualization / Output

Use the canonical cube smoke-test case to generate development-validation outputs.

Required visual checks:

1. Bellman attacker trajectory on the cube case.
2. Tabular Q-learning attacker trajectory on the same cube case.
3. Bellman vs Tabular trajectory comparison using the same geometry and sensor/defender setup.
4. A compact Bellman-vs-Tabular result comparison showing:
   - `J_A`,
   - `J_D`,
   - attacker-BR runtime,
   - total Local-SSE runtime,
   - active `dx=dy=dh`, `dpsi=dgamma`, and `r`.

Output format:

- Use `.png` for normal 2D plots, charts, and summary figures.
- Use `.html` only where an interactive 3D visualization is genuinely required.
- Preserve the current result-folder convention.

These visualizations are for code verification and regression checking, not final paper performance figures.

### Implementation Restrictions

During Phase 0:

- Do not implement DQN.
- Do not alter Bellman recurrence / DP logic.
- Do not alter Tabular Q-learning update logic or tuned hyperparameters.
- Do not alter attacker dynamics or feasible transitions.
- Do not alter Local SSE search logic.
- Do not alter `J_A` or `J_D`.
- Do not redesign terrain/sensor models.
- Do not run or present a final Monte Carlo performance study.
- Do not silently replace missing current-code values with values inferred from old files or this planning document.

Only make the minimum structural changes needed for reproducible execution, import compatibility, instrumentation, JSON logging, and visualization.

### Validation Tests

Phase 0 validation must verify:

1. The canonical cube case can be executed with Bellman.
2. The same canonical cube case can be executed with Tabular Q-learning.
3. Both methods return a feasible attacker trajectory under the existing dynamics/constraints.
4. Existing `J_A` / `J_D` evaluation runs successfully for both methods without changing the objective implementation.
5. Attacker-BR runtime is recorded for both methods.
6. Total Local-SSE runtime is recorded for both methods when the Local SSE pipeline is invoked.
7. Required JSON result fields are written.
8. Required PNG / 3D HTML validation outputs are produced.
9. Re-running the same deterministic configuration/seed, where the current implementation supports determinism, reproduces the same or numerically consistent result.
10. Existing Bellman / Tabular behavior is not changed by the Phase 0 wrappers or instrumentation.

### Failure Conditions

Phase 0 is not complete if any of the following occurs:

- Bellman or Tabular logic is rewritten instead of reused.
- The canonical cube configuration is recreated manually despite an existing authoritative configuration in the repository.
- `J_A` / `J_D` definitions differ between methods.
- Tabular training settings are guessed or changed without explicit instruction.
- Timing boundaries are inconsistent between repeated runs of the same method.
- A new result directory convention is created unnecessarily instead of preserving the project convention.
- DQN code is introduced.
- Phase 0 is treated as the final paper-performance experiment.

### Detailed Codex Command

When Phase 0 is executed in Codex, use the following instruction as the implementation contract:

> Work only on **Phase 0 — Freeze and Validate Existing Baselines**. First inspect the current local repository, especially `3D_0827/`, and treat it as authoritative. Do not infer exact implementation values from old uploaded snapshots or from the planning document. Discover the current Bellman attacker-BR implementation, Tabular Q-learning attacker-BR implementation, Local SSE entry path, canonical cube test configuration, Tabular hyperparameters, `J_A`/`J_D` functions, and result-folder convention. Record these discoveries in a JSON baseline manifest. If a suitable runnable entry point already exists, reuse it; otherwise create only a minimal runner/wrapper. Preserve Bellman, Tabular Q-learning, Local SSE, dynamics, feasibility, and objective logic exactly. Establish only the minimum import compatibility needed so later code under `3D_RL_DQN/` can use the required existing functionality without duplicating solver logic or creating circular imports. Instrument exactly two official runtime quantities: attacker-BR runtime and total Local-SSE runtime. For Tabular Q-learning, attacker-BR runtime must include its case-specific training plus the final policy/trajectory query required to return the BR. Use the existing cube case previously used for Bellman-vs-Tabular testing as the canonical Phase 0 smoke test. Save machine-readable results as JSON under the existing result-folder convention. Generate PNG validation figures for ordinary plots and use HTML only when a genuinely interactive 3D plot is required. The Phase 0 outputs are development/regression checks, not final publication performance results. Do not implement DQN, do not run a final Monte Carlo sweep, and do not modify any algorithmic/objective definition. At completion, report the discovered entry points/configuration, files changed, validation results, output paths, and any unresolved values that are genuinely absent from the current code.

### Exit Criteria

Phase 0 is complete only when:

- the current Bellman / Tabular / Local-SSE code paths have been discovered and documented,
- a reusable entry point exists, either pre-existing or minimally added,
- the canonical cube smoke test runs for Bellman and Tabular Q-learning,
- `J_A` and `J_D` are produced from the unchanged existing objective implementation,
- attacker-BR runtime and total Local-SSE runtime are logged,
- JSON results and required validation visualizations are produced under the existing result convention,
- current Tabular hyperparameters/settings are recorded from the actual code rather than guessed,
- import compatibility with the future `3D_RL_DQN/` work area is established at the minimum necessary level, and
- no Bellman, Tabular, Local-SSE, objective, dynamics, or terrain-model behavior has been intentionally changed.

---

## Phase 1 — Unified Attacker BR Environment and Interface

### Objective
Create a **solver-independent attacker best-response problem/interface** that exposes the same existing discrete attacker dynamics, feasibility rules, stage/terminal cost semantics, and terminal conditions to Bellman DP, Tabular Q-learning, and the future DQN implementation.

Phase 1 is an interface/refactoring and equivalence-validation phase. It does **not** introduce a new attacker model, new reward design, terrain-generalized DQN observation, neural network, DQN training, or publication-level performance experiment.

The common interface must make later solver substitution possible without changing the underlying P1b attacker BR problem.

### Architectural Boundary

Keep the following layers conceptually separate:

1. **Attacker BR problem/environment core**
   - scenario / terrain reference,
   - defender/sensor configuration or the existing sensing-hazard source,
   - attacker initial condition,
   - target / terminal specification,
   - spatial and angular discretization used by the attacker problem,
   - existing attacker motion/dynamics rules,
   - feasible transition generation / validation,
   - terminal-condition evaluation,
   - existing attacker cost/objective evaluation.

2. **Solver adapters**
   - Bellman DP adapter,
   - Tabular Q-learning adapter,
   - future DQN adapter.

3. **Local SSE layer**
   - defender local search,
   - defender candidate/neighbor generation,
   - Local SSE Chebyshev search radius `r`,
   - defender-side equilibrium logic.

The Local SSE neighbor-search radius `r` is **not an attacker action-neighborhood parameter** and must not be moved into or conflated with the attacker BR environment. It remains a Local-SSE-layer parameter.

### Mandatory Current-Code Discovery Before Refactoring

Before creating or modifying the common interface, inspect the latest `3D_0827/` code and the Phase 0 discovery manifest/result. Confirm from the actual code:

1. How Bellman represents an attacker state.
2. How Tabular Q-learning represents an attacker state.
3. How each method enumerates or validates feasible transitions/actions.
4. Whether Bellman uses forward successors, backward predecessors, a DAG ordering, or another transition-access pattern.
5. How heading and glide-slope discretization are represented and indexed.
6. How spatial discretization is represented and indexed.
7. How terrain collision / terrain-clearance feasibility is checked.
8. How any kinematic / glide / angular transition constraints are checked.
9. How target/terminal states are identified.
10. How sensing hazard / detection cost is queried.
11. How stage cost, terminal cost, mission-time contribution, and final `J_A` are currently computed.
12. How the existing Tabular Q-learning converts the cost/objective into its reward/update quantity, including sign convention.
13. Whether Bellman and Tabular currently call a shared transition/cost implementation or contain duplicated implementations.
14. Which existing objects/configurations must be importable from `3D_0827/` by future code under `3D_RL_DQN/`.

Do not assume that Bellman and Tabular are already semantically identical merely because they were intended to solve the same problem. Verify equivalence from the code and with parity tests.

### Common BR Problem Contract

Create the smallest common attacker-problem abstraction consistent with the actual repository architecture. The exact file/class/function names should follow the current project convention; do not force a new naming scheme if an equivalent abstraction already exists.

Conceptually, the interface must provide the following capabilities.

#### 1. Problem configuration access
Expose or reference, without duplicating model data:

- current terrain/scenario,
- current defender/sensor configuration or current hazard provider,
- attacker start condition,
- target/terminal condition,
- attacker spatial discretization,
- attacker angular discretization,
- current dynamics/feasibility configuration,
- current objective/cost evaluator.

All values must come from the existing current configuration/code.

#### 2. State conversion / canonicalization
Provide a deterministic way to convert the state representation required by each existing solver into the common problem representation and back when needed.

Do **not** redesign the DQN observation vector here. Phase 1 only needs a canonical environment-level representation sufficient to reproduce existing Bellman/Tabular transitions and costs. Terrain-generalized neural-network observations belong to Phase 2.

If Bellman and Tabular already use the same state representation, reuse it instead of creating redundant conversion layers.

#### 3. Transition access
Provide a single authoritative path for the existing feasible attacker transitions.

Depending on what the actual Bellman implementation requires, expose:

- feasible forward successors, and/or
- feasible predecessors,
- any existing transition metadata required by cost/dynamics evaluation.

The common transition path must preserve the current discretized dynamics exactly.

Do not broaden or shrink the action/transition set for RL convenience.

#### 4. Feasibility evaluation
Reuse the current feasibility rules for:

- terrain collision / terrain clearance,
- spatial bounds,
- heading / glide-slope constraints,
- motion or kinematic constraints,
- any mode-specific constraints actually present in the current attacker BR implementation,
- any other existing transition rejection rule discovered in the code.

Do not invent new feasibility constraints or remove existing ones.

If the current code does not expose reason codes for rejected transitions, do not redesign the physics merely to create them. A visualization may distinguish feasible vs infeasible candidate transitions using existing boolean checks.

#### 5. Terminal evaluation
Expose the existing terminal/goal condition through the common interface.

Bellman, Tabular, and future DQN must use the same terminal semantics. Preserve any terminal tolerance, target set, or terminal-cost behavior found in the current code.

#### 6. Canonical stage/trajectory cost
Expose the existing attacker cost semantics in a solver-neutral form.

The environment core should provide the existing cost quantity used to construct/evaluate `J_A`. If the current Tabular implementation internally uses a reward sign transformation or another equivalent representation, implement that only as a solver adapter around the common cost; do not alter the underlying objective.

Do not choose a new `reward = -cost` convention unless that mapping is confirmed from the current implementation or explicitly introduced later as a documented adapter decision.

#### 7. Common BR solver call contract
Provide a common callable boundary so a caller can request an attacker BR from a selected solver without knowing that solver's internal implementation.

Conceptually:

```text
solve_attacker_br(problem_instance, solver_method, solver_config) -> br_result
```

The exact API/name may differ to match the repository.

`solver_method` must at least support the existing:

- Bellman mode,
- Tabular-Q mode.

DQN mode is **not implemented in Phase 1**, but the interface should be extendable to add it later without changing the Bellman/Tabular problem semantics.

The common BR result must contain or reference enough information for the current downstream code to obtain at least:

- solver status / success,
- returned attacker trajectory/path,
- final state / goal-reached status as currently applicable,
- `J_A` evaluated by the existing common objective implementation,
- existing solver-specific diagnostic data only when useful,
- configuration/method identifier needed for logging.

Do not force `J_D` into the attacker environment if the current architecture evaluates `J_D` at the Local SSE layer. Preserve the existing ownership of `J_D` evaluation.

### Refactoring Rule for Duplicate Existing Logic

If Bellman and Tabular currently duplicate transition, feasibility, terminal, or cost logic:

1. First establish behavior equivalence with tests on the canonical cube case and sampled valid states.
2. Only after equivalence is demonstrated, factor the duplicated behavior into the shared environment/core in the smallest possible change.
3. Keep thin solver adapters for Bellman and Tabular.
4. Re-run the Phase 0 regression outputs after refactoring.

If the two legacy implementations produce different transition sets, costs, or terminal behavior for the same state/configuration:

- do **not** silently choose one implementation,
- do **not** average/merge the behavior,
- record the mismatch with state/configuration details,
- generate a diagnostic visualization where possible,
- stop that specific consolidation and report the discrepancy for review.

### Gymnasium Boundary

Do not require the core Phase 1 environment to inherit from or depend on Gymnasium unless the current repository already uses it and doing so does not distort Bellman compatibility.

The preferred architecture is a solver-independent core problem/transition interface that Bellman can use directly. A Gymnasium-compatible wrapper for DQN can be added in Phase 2/3 around this core if needed.

### Phase 1 Validation Dataset

Use the same canonical cube case identified in Phase 0 for interface/regression validation.

This remains a **development test case**, not a paper performance case.

Use the actual active discretization/configuration from the current cube test for the primary smoke test. The interface must continue accepting the intended project discretization domain established in Phase 0, but Phase 1 does not need an exhaustive sweep over every spatial/angular discretization or Local-SSE radius.

### Required Equivalence / Parity Tests

Implement tests that compare the legacy behavior and the new common interface on the same cube problem.

At minimum test:

1. **Initial-state parity**
   - common interface resolves the same attacker initial condition as the current baseline.

2. **Terminal-condition parity**
   - known terminal and non-terminal states are classified consistently with current code.

3. **Transition-set parity**
   - for selected reachable states, compare feasible successor/predecessor sets required by the solvers.
   - compare actual discrete state identifiers/coordinates/angles, not only the number of neighbors.

4. **Feasibility parity**
   - transitions accepted/rejected by the common interface agree with existing current-code checks.

5. **Immediate-cost parity**
   - for identical feasible transitions, the common interface returns the same existing cost quantity used by the baseline implementation.

6. **Trajectory-objective parity**
   - evaluating the same baseline trajectory through the common evaluation path reproduces the current `J_A` result.

7. **Bellman regression**
   - Bellman through the common interface reproduces the Phase 0 Bellman trajectory/result or a numerically equivalent result under existing tie behavior.

8. **Tabular regression**
   - Tabular Q-learning through the common interface remains consistent with the Phase 0 result under the same supported seed/training configuration.

9. **Downstream compatibility**
   - the existing Local SSE code can still consume Bellman/Tabular BR results without changing the Local SSE algorithm.

For numerical comparisons, reuse tolerance rules already present in the repository/test suite. If no numerical tolerance convention exists, do not silently invent a model tolerance; report absolute/relative discrepancies explicitly and mark acceptance as requiring review.

### Required Visualization / Output

Generate the following Phase 1 development-validation visualizations using the canonical cube case.

#### 1. One-state feasible-transition visualization
Select a representative reachable attacker state and display:

- current state,
- all candidate transitions available from the existing discrete transition mechanism if such candidates are explicitly represented,
- feasible transitions,
- infeasible transitions when they can be obtained without changing the model.

Use labels/legend sufficient to inspect whether the common interface preserves the legacy transition neighborhood.

#### 2. Angular-transition visualization
For the same or another representative state, visualize the existing allowed transition structure in:

- heading `psi`,
- glide-slope angle `gamma`,

using the actual discretization and constraints discovered from the code.

Do not invent angular bounds or step rules.

#### 3. Terrain-feasibility visualization
Show representative feasible and terrain-invalid transitions relative to the cube geometry using the existing collision/clearance logic.

Use PNG if a 2D projection is sufficient; use interactive HTML only when the geometry is genuinely clearer in 3D.

#### 4. Sensor-hazard overlay
Visualize the current sensing-hazard/detection-cost information used by the attacker problem together with the representative attacker state/transition region, using the existing hazard representation.

Do not create a new hazard model for this plot.

#### 5. Feasible rollout
Generate one rollout constructed only from common-interface feasible transitions to demonstrate that repeated transitions remain valid under the current terrain/dynamics rules.

This rollout is a structural test, not an RL result and not an optimal trajectory claim.

#### 6. Transition-parity diagnostic
Produce a compact PNG/table or JSON summary showing the number of sampled states/transitions checked and any mismatch counts between legacy and common-interface behavior.

If mismatches exist, save the relevant state/configuration information for debugging.

### Phase 1 Output / Logging

Preserve the existing result-folder convention established in Phase 0.

Create a Phase 1 JSON manifest containing at least:

```text
phase
canonical_cube_case
common_environment_source
bellman_adapter_source
tabular_adapter_source
state_representation_sources
transition_logic_source
feasibility_logic_source
terminal_logic_source
cost_objective_source
legacy_transition_logic_was_duplicated
parity_test_summary
unresolved_mismatches
files_changed
```

Do not invent paths or names in advance; populate them from the actual repository after implementation.

### Implementation Restrictions

During Phase 1:

- Do not implement DQN.
- Do not design the terrain-generalized DQN observation tensor/vector.
- Do not add CNN/MLP architecture decisions.
- Do not begin multi-terrain training.
- Do not alter `J_A` or `J_D`.
- Do not alter sensing models.
- Do not alter terrain geometry.
- Do not alter attacker dynamics merely to simplify an RL interface.
- Do not alter Local SSE defender-search logic or reinterpret radius `r`.
- Do not change Tabular hyperparameters discovered/frozen in Phase 0.
- Do not replace exact legacy transition logic with an approximate vectorized version unless exact equivalence is first demonstrated.
- Do not duplicate Bellman/Tabular logic into `3D_RL_DQN/` when it can be imported/shared cleanly from the common core.
- Do not run the final discretization/Monte-Carlo performance study.

### Detailed Codex Command

When Phase 1 is executed in Codex, use the following instruction as the implementation contract:

> Work only on **Phase 1 — Unified Attacker BR Environment and Interface**. Apply the document-wide repository-grounding rule: inspect the latest `3D_0827/` and, where relevant, `3D_RL_DQN/`; do not guess any existing code-related value or behavior. Start from the Phase 0 discovery manifest and canonical cube validation case. Inspect how the current Bellman and Tabular Q-learning implementations represent states, enumerate/validate transitions, apply spatial/angular discretization, enforce terrain/dynamic feasibility, identify terminal states, query sensing hazard, calculate stage/trajectory cost and `J_A`, and map cost to the current Tabular reward/update convention. Build the smallest solver-independent attacker-BR problem/environment core that exposes the existing behavior without changing it. Keep the Local SSE layer separate; in particular, Local SSE Chebyshev radius `r` remains a defender-search parameter and must not be reinterpreted as an attacker action-neighborhood setting. Add thin Bellman and Tabular solver adapters behind a common BR-call contract so downstream code can select the attacker solver without duplicating environment semantics. Do not implement DQN or the terrain-generalized DQN observation yet. If Bellman and Tabular contain duplicated transition/feasibility/cost logic, first run parity tests; only consolidate logic after equivalence is demonstrated. If their behavior differs for the same state/configuration, do not silently choose one version—record and visualize the mismatch and report it as unresolved. Preserve current `J_A`, `J_D`, dynamics, sensing, terrain, terminal, and tie-breaking behavior. Re-run Phase 0 regression checks through the new interface. Generate the required Phase 1 transition, angular, terrain-feasibility, hazard-overlay, feasible-rollout, and parity-diagnostic outputs under the existing result convention. Use PNG for normal plots and HTML only when interactive 3D is genuinely required. Write a Phase 1 JSON manifest documenting the actual shared-core/adapters/source locations, parity results, files changed, and unresolved mismatches. At completion, report what was reused, what was minimally refactored, all parity/regression results, output paths, and any ambiguity that cannot be resolved from the current code.

### Exit Criteria

Phase 1 is complete only when all of the following are true:

1. A solver-independent attacker BR problem/environment boundary exists and is documented.
2. Bellman and Tabular Q-learning can both access the same authoritative dynamics/transition/feasibility/terminal/cost semantics through that boundary or proven-equivalent thin adapters.
3. The existing state representations and any required conversion paths are documented from the actual code.
4. Bellman-required successor/predecessor access remains functional.
5. The Local SSE radius `r` remains entirely in the Local SSE layer.
6. Transition-set, feasibility, terminal, immediate-cost, and trajectory-`J_A` parity tests pass, or any genuine legacy mismatch is explicitly isolated and reported rather than hidden.
7. Bellman regression remains consistent with Phase 0.
8. Tabular regression remains consistent with Phase 0 under the same supported configuration/seed.
9. Existing Local SSE downstream code can still consume the Bellman and Tabular BR results without an algorithmic rewrite.
10. Required Phase 1 visual validation outputs and JSON manifest are produced.
11. No DQN network, DQN observation design, multi-terrain training, or final performance sweep has been introduced.
12. Future `3D_RL_DQN/` code can import/use the common attacker-problem semantics without copying the Bellman/Tabular environment logic.

### Failure Conditions

Phase 1 is not complete if any of the following occurs:

- existing transition/dynamics values are guessed instead of read from current code,
- Bellman and Tabular mismatches are silently normalized,
- the common interface changes the feasible action/transition set,
- `J_A` semantics or terminal behavior change,
- Local SSE radius `r` leaks into the attacker action model,
- DQN-specific observation/network design is prematurely embedded into the core environment,
- legacy solver logic is duplicated unnecessarily into `3D_RL_DQN/`,
- Phase 0 regression behavior is broken without an identified pre-existing mismatch,
- validation outputs are treated as publication performance evidence.

---

## Phase 2 — Terrain-Aware DQN State Representation

### Phase 2 Design Log

- **Current scope:** use condition-specific DQN models (A1) for each `(spatial discretization, angular discretization)` condition.
- **Future extension only:** investigate one DQN across multiple discretization conditions (A2) after the current condition-specific pipeline is complete. Do not implement A2 support in the current phases.


### Objective
Define, implement, and validate a **fixed-format, terrain-aware DQN observation representation** that can describe the existing P1b attacker BR state on different terrain geometries without changing the underlying attacker dynamics, feasible transitions, sensing model, or objective.

Phase 2 builds only the observation/state-encoding layer required by the future DQN. It does **not** implement a neural network, replay buffer, DQN update, exploration policy, or DQN training loop.

The observation must be derived from the Phase 1 common attacker-BR environment so that the future DQN observes the same physical/discrete problem that Bellman and Tabular Q-learning solve.

### Primary Design Goal

The Phase 2 observation must separate:

1. **agent / motion state**, which describes the current attacker state required by the existing glide model,
2. **goal-relative information**, which describes where the terminal target lies relative to the attacker,
3. **local terrain geometry**, which allows the policy to react to terrain shape rather than memorize one terrain identifier,
4. **local sensing-hazard information**, which allows the policy to react to the current defender/sensor configuration,
5. **validity metadata / masks** needed to distinguish sampled local cells that are outside the modeled environment or otherwise unavailable.

The representation must not contain a terrain name, terrain file ID, scenario ID, or other direct identifier that would allow geometry memorization by label rather than observation.

### Mandatory Repository Discovery Before Observation Design

Before implementing the observation builder, inspect the current Phase 1 common environment plus the latest authoritative `3D_0827/` and `3D_RL_DQN/` code. Determine from actual code:

1. the canonical attacker state variables required by the current glide-phase BR problem,
2. whether `gamma` / glide-slope is a persistent state variable, an action/transition variable, or derived quantity,
3. whether any motion mode, speed, or additional state variable is actually required by the current attacker model,
4. the physical meaning, units, and valid range of each state component,
5. how terrain is represented internally:
   - height map,
   - mesh,
   - occupancy structure,
   - analytic geometry,
   - or another representation,
6. how terrain elevation/clearance can be queried at arbitrary or discrete `(x,y)` locations,
7. the authoritative sensing-hazard/detection-cost representation used by the attacker BR,
8. every state variable on which that hazard depends, including position, altitude, heading, glide angle, sensor configuration, or other quantities,
9. whether hazard is already precomputed on a grid or evaluated on demand,
10. the environment/spatial bounds used by the current problem,
11. the target representation and terminal set/tolerance,
12. any existing normalization, coordinate transform, local-map extraction, or feature-construction utilities already present in the repository.

Do not infer any of these from old snapshots or prior chat descriptions. Record the discovered sources in the Phase 2 manifest.

### Architectural Requirement

Implement the DQN observation as a layer **on top of** the Phase 1 common attacker-BR problem, not as a replacement for it.

Conceptually:

```text
Phase 1 Common Attacker BR Problem
            │
            ├── dynamics / transitions
            ├── feasibility
            ├── terminal condition
            ├── terrain query
            └── sensing-hazard / cost query
            │
            ▼
Phase 2 Observation Builder
            │
            ├── ego/motion features
            ├── goal-relative features
            ├── local terrain channels
            ├── local hazard channels
            └── validity masks / metadata
            │
            ▼
Future Phase 3 DQN
```

The observation builder must be deterministic: the same problem instance and canonical attacker state must produce the same raw observation.

### Observation Contract

Implement a **structured observation object** first. Do not prematurely bind the representation to an MLP or CNN implementation.

Conceptually, the observation should expose components equivalent to:

```text
observation = {
    ego_features,
    goal_features,
    terrain_channels,
    hazard_channels,
    validity_channels,
    metadata
}
```

The exact Python container/type and field names must follow the repository convention.

Also provide a deterministic tensor-ready conversion function for future DQN use, but do not choose the Phase 3 network architecture here.

The observation schema and tensor dimensions must be written into the Phase 2 JSON manifest.

### 1. Ego / Motion Features

Construct ego features only from state variables confirmed by the current attacker model.

At minimum, inspect whether the current canonical state contains or requires:

- attacker altitude or terrain clearance,
- heading `psi`,
- glide-slope / glide angle `gamma`, if it is truly part of the current state,
- motion mode, if present,
- speed or another kinematic variable, if present.

Do not add a state variable merely because it is common in flight-control RL.

For periodic angles, provide a representation that does not introduce an artificial discontinuity at angle wrap-around. Unless the current code already provides an equivalent encoding, the observation layer should support sine/cosine encoding:

```text
psi -> [sin(psi), cos(psi)]
gamma -> [sin(gamma), cos(gamma)]    # only if gamma is an actual required state variable
```

This encoding changes only the neural observation, not the underlying discrete state or dynamics.

Avoid using raw absolute `(x,y)` world coordinates as the primary terrain-generalization signal unless the current model requires them for information that cannot otherwise be represented. The authoritative discrete/world coordinates remain available internally to the environment, but the DQN observation should preferentially describe geometry relative to the attacker and goal.

### 2. Goal-Relative Features

Construct goal information relative to the current attacker state rather than by terrain/scenario identifier.

The builder must support equivalent information to the physically relevant relative displacement between attacker and goal/terminal set, using the actual target definition found in the code.

Where possible, represent horizontal goal displacement in an attacker-centered frame and include vertical displacement or equivalent altitude relation when relevant.

Do not assume a specific terminal point if the existing code uses a target region/set. Derive the goal-relative quantity from the actual terminal representation.

Any normalization scale used for goal-relative displacement must be explicit, reproducible, and recorded. Do not silently normalize using statistics from final unseen test terrains.

### 3. Local Terrain Representation

The DQN must receive local terrain information sufficient to react to geometry on a terrain not seen during training.

Implement a configurable local terrain observation centered on the current attacker position.

The preferred representation contract is a fixed-size local sampling window producing one or more terrain channels. The exact terrain quantity must be derived from the current terrain model and may include, where supported by the code:

- terrain elevation relative to attacker altitude,
- attacker-to-terrain clearance,
- or an equivalent relative geometric quantity.

Do not substitute a new terrain model.

#### Local reference frame

The observation builder must support an **attacker-centered local frame**. A heading-aligned / egocentric orientation should be available so the same geometric situation can be represented consistently after global translation/rotation. The final choice of whether the primary DQN input is heading-aligned or world-aligned must be explicitly documented as a new design choice; do not allow this to vary silently between runs.

#### Fixed input dimensions

The terrain observation delivered to one trained DQN model must have a fixed tensor shape independent of terrain-map dimensions.

Therefore the local observation must use configurable parameters equivalent to:

```text
local_window_extent_m
local_window_shape_cells = (H, W)
local_window_orientation
terrain_channel_definition
```

Do not hard-code these values from this planning document. They are new DQN design parameters and must be placed in a Phase 2/3 configuration file once selected.

If sampling from a terrain representation at locations not identical to the Bellman grid, use a deterministic interpolation/query method compatible with the existing terrain model. Observation sampling may interpolate terrain for perception, but it must **not** change the Phase 1 discrete transition or collision logic used to determine the actual attacker trajectory.

### 4. Local Sensing-Hazard Representation

Construct local sensing information from the **same authoritative sensing-hazard / detection-cost model used by Bellman and Tabular Q-learning**.

The Phase 2 builder must first determine from current code whether the hazard is:

- a scalar function of position only,
- a function of `(x,y,h)`,
- dependent on heading / glide angle,
- dependent on motion mode or speed,
- precomputed on a multi-dimensional grid,
- or evaluated directly from sensor models.

Do not collapse a multi-dimensional hazard model into a 2D `(x,y)` image without verifying that no required state dependence is lost.

Use the same local spatial support/reference frame as the terrain observation where possible, but preserve whatever additional channel/state conditioning is required by the actual hazard model.

The local hazard representation must change when the defender/sensor configuration changes. The observation must not rely on a hard-coded defender location from the cube case.

If multiple equally valid hazard encodings remain after code inspection, implement the extraction utilities needed to expose the alternatives and report the ambiguity rather than choosing silently.

### 5. Validity / Boundary Representation

A fixed local window may extend beyond the valid terrain/environment domain.

Do not silently fill such cells with values that could be confused with valid terrain or zero sensing hazard.

Provide an explicit validity/boundary channel or mask so the future DQN can distinguish:

- valid sampled locations,
- outside-domain locations,
- and any other unavailable sample class that is already meaningful in the current environment.

This validity mask is an observation feature only. It must not replace the authoritative Phase 1 action-feasibility checks.

### 6. Normalization Contract

Separate **observation structure** from **training-data normalization**.

Phase 2 must provide a deterministic normalization interface, but it must not fit normalization statistics using future unseen test terrains.

Use the following rule:

1. Reuse any physically meaningful normalization already present in current code when appropriate.
2. For inherently bounded transformations such as sine/cosine angle encoding, use the deterministic bounded representation.
3. For new terrain/hazard feature scaling that requires dataset statistics, implement the normalizer interface/configuration but defer fitting mean/std/min/max statistics to the training-terrain set in Phase 4.
4. Save every normalization parameter used by a trained DQN with the model/configuration.
5. Final unseen terrain data must never be used to fit or update normalization parameters.

During Phase 2 visualization/testing, raw and normalized versions should both be inspectable.

### 7. Discretization Metadata

The experiment varies:

- `dx = dy = dh`,
- `dpsi = dgamma`,
- Local SSE defender radius `r`.

For the current implementation, do **not** insert these computational-condition variables into the DQN state because the project has selected a condition-specific DQN design.

In particular:

- Local SSE radius `r` must not be part of the attacker observation merely because it is an experimental variable; it belongs to the defender Local SSE search.
- Spatial/angular discretization are model/configuration metadata because a separate DQN will be trained for each discretization condition.
- Cross-discretization DQN generalization is future work only and must not affect the current observation schema.

### 8. Action Feasibility Interface Preparation

Phase 2 does not implement DQN action selection or training, but ensure the observation builder can be called together with the Phase 1 feasible-transition interface for the same canonical state.

Expose sufficient state-to-observation traceability so Phase 3 can later construct a DQN action mask without modifying the observation semantics.

Do not encode the action mask into terrain/hazard channels unless explicitly justified later. Keep physical observation and action feasibility conceptually separate.

### Required Phase 2 Development Visualizations

Use the canonical cube case from Phases 0–1 only as a development/validation case.

Generate at least the following outputs.

#### 1. Global-to-local observation figure
Show in one validation figure:

- global cube terrain / environment,
- current attacker location,
- current heading if applicable,
- goal/terminal location or region,
- the physical footprint of the local DQN observation window.

#### 2. Local terrain-channel visualization
Display every terrain channel separately with:

- channel name,
- units or normalized units,
- min/max values,
- local coordinate axes/orientation.

#### 3. Local hazard-channel visualization
Display every sensing-hazard channel separately with the same local footprint and explicit conditioning/meaning.

#### 4. Validity-mask visualization
Show the boundary/validity channel for at least one state whose local window intersects an environment boundary, if such a state is reachable in the canonical test problem.

#### 5. Observation-component summary
Create a compact figure/table showing:

- ego feature names/dimensions,
- goal feature names/dimensions,
- terrain tensor shape,
- hazard tensor shape,
- validity tensor shape,
- final tensor-ready shape/size.

#### 6. Raw vs normalized input inspection
For one representative state, show raw feature ranges and the transformed/normalized values that would be supplied to the future network.

#### 7. Translation / geometry sanity visualization
Where the environment permits an equivalent shifted or differently located local configuration, visualize that the observation is based on relative/local geometry rather than a terrain identifier or absolute-map lookup artifact.

Do not claim statistical generalization from this plot; it is only a representation sanity check.

Use PNG for normal figures and HTML only when interactive 3D genuinely improves inspection.

### Required Phase 2 Tests

Implement tests for at least:

1. **Determinism**
   - identical problem/state/configuration produces identical raw observation.

2. **Shape consistency**
   - all valid attacker states for a given Phase 2 observation configuration produce the same observation schema/tensor dimensions.

3. **No terrain-ID leakage**
   - observation contains no terrain filename, terrain ID, scenario ID, or equivalent direct label.

4. **State consistency**
   - ego/motion features correspond to the actual canonical Phase 1 state.

5. **Goal consistency**
   - goal-relative features change correctly with attacker position/heading using the actual terminal definition.

6. **Terrain-query parity**
   - sampled terrain values agree with the authoritative terrain query at selected sample points within numerical precision supported by the current representation/interpolation method.

7. **Hazard-query parity**
   - sampled hazard values agree with the authoritative sensing-hazard query for the same physical/state conditions.

8. **Boundary handling**
   - out-of-domain local cells are explicitly marked by validity metadata rather than silently interpreted as valid zero terrain/hazard.

9. **Angle-wrap consistency**
   - if periodic angular variables are encoded, observations near the angular wrap boundary do not exhibit an artificial raw-angle jump in the encoded feature.

10. **Legacy behavior preservation**
    - constructing an observation does not change the Phase 1 transition set, feasibility decisions, terminal result, or `J_A` evaluation.

11. **Serialization/config reproducibility**
    - the observation configuration and any deterministic normalization parameters can be saved to JSON and reloaded to reproduce the same observation.

### Phase 2 Manifest / Logging

Under the existing result-folder convention, write a Phase 2 JSON manifest containing at least:

```text
phase
observation_builder_source
canonical_state_source
ego_feature_definition
goal_feature_definition
terrain_source
terrain_channel_definition
hazard_source
hazard_channel_definition
validity_channel_definition
local_window_extent_m
local_window_shape_cells
local_window_orientation
normalization_definition
raw_observation_schema
tensor_ready_shape
discretization_scope_decision
repository_sources_inspected
validation_test_summary
visualization_outputs
unresolved_design_items
files_changed
```

For fields representing new DQN design parameters, identify them explicitly as `new_design_choice` rather than inherited repository values.

### Implementation Restrictions

During Phase 2:

- Do not implement the DQN network.
- Do not implement replay memory, target network, epsilon exploration, TD loss, or optimizer.
- Do not train a policy.
- Do not modify Bellman DP.
- Do not modify Tabular Q-learning behavior.
- Do not modify Local SSE search.
- Do not modify attacker dynamics or feasibility.
- Do not modify `J_A` / `J_D`.
- Do not alter the sensor model to make hazard extraction easier.
- Do not hard-code the cube terrain into the observation builder.
- Do not use terrain/scenario IDs as neural features.
- Do not fit normalization statistics using unseen/final test terrains.
- Do not treat Phase 2 visualizations as evidence that DQN generalization has been achieved.
- Do not silently decide unresolved DQN design parameters by using common RL defaults.

### Explicit Design Decisions and User-Confirmation Requirements

The following items cannot all be recovered from the existing Bellman/Tabular code because some are genuinely new DQN design choices.

#### Decision A — Generalization scope across discretization settings — RESOLVED FOR CURRENT WORK

Use **A1. Terrain-generalized, condition-specific DQN** for the current implementation.

- Train a separate DQN for each selected `(spatial discretization, angular discretization)` condition.
- Each DQN generalizes across terrain/sensor cases, not across discretization definitions.
- `dx`, `dy`, `dh`, `dpsi`, and `dgamma` remain model/configuration metadata for that trained model rather than required neural observation features.
- Local SSE defender radius `r` remains outside the attacker observation.

**Future-log only:** a later extension may investigate **A2: one DQN across multiple discretization conditions**, but A2 must not influence the current Phase 2–Phase 7 implementation. Do not add cross-discretization inputs, variable-discretization action semantics, or compatibility machinery now merely to anticipate A2.

#### Decision B — Local observation window geometry — MUST ASK USER AT PHASE 2 EXECUTION

The following values are new perception-design parameters and are **not yet selected**:

```text
local_window_extent_m
local_window_shape_cells = (H, W)
local_window_orientation = heading_aligned or world_aligned
```

When Codex actually executes Phase 2:

1. first inspect the current repository and report any existing local-map/window utilities or architectural constraints that materially affect these choices,
2. then explicitly ask the user to select/confirm the Phase 2 values before hard-coding or finalizing the observation builder,
3. do not substitute standard RL defaults,
4. keep these parameters configurable in the final implementation.

Phase 2 planning may proceed without selecting them, but Phase 2 implementation/validation cannot be declared complete until the user confirms them.

#### Decision C — Final hazard-channel form — CODE-DERIVED WHEN POSSIBLE

After inspecting the actual hazard dependencies, determine whether a 2D local hazard channel is information-preserving or whether multiple altitude/angle-conditioned channels are required.

- If the current hazard model makes the faithful representation unambiguous, implement that representation and document the source.
- If multiple reduced encodings remain equally valid after repository inspection, report the alternatives and ask the user for explicit selection rather than discarding state dependence.

### Detailed Codex Command

When Phase 2 is executed in Codex, use the following instruction as the implementation contract:

> Work only on **Phase 2 — Terrain-Aware DQN State Representation**. Apply the global repository-grounding rule. Start from the completed Phase 1 common attacker-BR environment and inspect the latest `3D_0827/` / `3D_RL_DQN/` code to determine the actual canonical attacker state, terrain query, hazard/detection-cost dependencies, target/terminal representation, environment bounds, and any existing feature/normalization utilities. Do not guess whether `gamma`, speed, motion mode, or other variables belong in the state; include only variables confirmed by the current model. Implement a deterministic structured observation builder on top of the Phase 1 environment, separating ego/motion features, goal-relative features, local terrain channels, local sensing-hazard channels, and explicit validity/boundary metadata. Do not hard-code the cube case or expose terrain/scenario IDs. Support an attacker-centered local sampling window with configurable physical extent, fixed output shape, and explicitly configured orientation; do not invent the window size/orientation if it has not been selected. Preserve the full state dependence of the existing hazard model; do not flatten a multi-dimensional hazard into a 2D map unless code inspection confirms that this preserves the required information. Keep observation sampling separate from the authoritative discrete transition/feasibility logic so interpolation used for perception cannot change Bellman/Tabular dynamics. Provide a deterministic tensor-ready conversion without implementing a neural network. Separate deterministic feature transforms from dataset-fitted normalization; any normalization statistics that require training data must remain unfitted until the training-terrain phase and must never use final unseen terrains. Keep Local SSE radius `r` outside the attacker observation. Do not add spatial/angular discretization values to the neural state in the current implementation; use a condition-specific DQN for each selected discretization condition. Record cross-discretization DQN only as a future extension. Generate the required global-to-local, terrain-channel, hazard-channel, validity-mask, observation-shape, and raw-vs-normalized validation figures using the canonical cube case only for development inspection. Implement deterministic shape/query/boundary/no-ID-leakage tests and confirm that observation construction does not change Phase 1 transitions, feasibility, terminal behavior, or `J_A`. Save the complete observation schema/configuration and validation results in the Phase 2 JSON manifest. Do not implement or train DQN in this phase. At completion, report all repository-derived state/hazard/terrain facts, all new design parameters actually selected, unresolved design decisions, files changed, test results, and visualization paths.

### Exit Criteria

Phase 2 is complete only when:

1. the actual canonical attacker state and hazard dependencies have been documented from current code,
2. a deterministic terrain-aware observation builder exists on top of the Phase 1 common environment,
3. the observation cleanly separates ego, goal-relative, terrain, hazard, and validity information,
4. local terrain/hazard observations have fixed dimensions for a given model configuration and do not depend on total terrain-map size,
5. no terrain/scenario identifier leaks into the neural observation,
6. terrain and hazard sample values are validated against the authoritative current-code queries,
7. out-of-domain local cells are explicitly represented by validity metadata,
8. observation configuration is serializable/reproducible,
9. required Phase 2 visualizations and JSON manifest are produced,
10. Phase 1 dynamics/transition/feasibility/terminal/objective behavior remains unchanged,
11. no DQN network or training code has been introduced, and
12. Decision A is recorded as condition-specific DQN for current work; Decision B has been explicitly confirmed by the user during Phase 2 execution; and Decision C has been resolved from current code or escalated to the user if multiple faithful encodings remain.

### Failure Conditions

Phase 2 is not complete if:

- the observation is built from guessed legacy state/hazard definitions,
- the cube terrain is hard-coded into the DQN observation,
- terrain/scenario IDs are used as features,
- the local observation has variable tensor dimensions within one model configuration,
- hazard state dependence is discarded without verification,
- local-map interpolation changes the actual transition/collision model,
- final unseen terrains are used to fit normalization,
- Local SSE radius `r` is inserted into the attacker state without an explicit design reason,
- network/training implementation begins before the observation contract is validated,
- or unresolved new DQN design choices are silently filled with standard/default RL values.

---

## Phase 3 — Single-Terrain DQN Sanity Check

### Phase 3 Execution Decision Log

- **Gate A — development condition:** resolved from the completed Phase 0–2
  canonical case as `dx = dy = dh = 100 m` and `dpsi = 5 deg`. Local-SSE
  radius remains outside the Phase 3 attacker-BR DQN.
- **Gate D — sanity/pass criterion:** explicitly approved by the user on
  2026-10-02. Use three independent seeds. Every seed must return a feasible,
  goal-reaching trajectory under the authoritative Phase 1 checks; the median
  absolute Bellman-relative `J_A` error must be at most 10%, and every seed's
  error must be at most 20%. For this nonzero canonical Bellman objective, use
  `abs(J_A_DQN - J_A_Bellman) / abs(J_A_Bellman)`. Evaluate both objectives only
  with the authoritative common evaluator.
- **Device preparation:** the current Windows RTX 5070 environment was
  validated with `torch 2.14.0+cu130`; Phase 3 must retain automatic CPU/CUDA
  selection and log the actual runtime device.
- **Gate B — neural architecture/configuration:** approved by the user on
  2026-10-02. Use the Phase 2 `(7, 21, 21)` spatial tensor with convolution
  channels `32 -> 64 -> 64` (3x3 kernels, stride 1 then 2 then 2), flatten to a
  256-unit spatial embedding; process the 8 scalar values through `64 -> 64`;
  concatenate to 320 units and use `320 -> 256 -> 72` for the canonical action
  values. Use ReLU and plain DQN, without dueling, Double-DQN, or prioritized
  replay variants in Phase 3.
- **Gate C — training hyperparameter bundle:** approved by the user on
  2026-10-02. Train seeds `0, 1, 2` for 20,000 episodes each with `gamma=1.0`,
  replay capacity 50,000, warm-up 2,000 transitions, batch size 128, AdamW at
  `3e-4`, Huber loss, linear epsilon `1.0 -> 0.05` over 60% of the configured
  maximum environment steps, one optimizer update per environment step, hard
  target synchronization every 1,000 optimizer steps, and global gradient-norm
  clipping at 10. Evaluate every 500 episodes, prefer CUDA automatically when
  available, and select the best feasible goal-reaching checkpoint by minimum
  authoritative `J_A`.

### Phase 3 Execution Result

Phase 3 was implemented and validated on 2026-10-02. The official manifest is
`3D_0827/figure/phase_3_single_terrain_dqn/phase3_single_terrain_dqn_manifest.json`.

- The full reachable-graph audit checked 37,951 states and 494,259 transitions;
  all 72 canonical action IDs were exercised, and no nonterminal dead end was
  found.
- The Bellman-trajectory reward identity passed with zero numerical error:
  `sum(reward) = -glide_cost` and `-powered_cost + sum(reward) = -J_A`.
- All approved seeds (`0, 1, 2`) used CUDA and returned the exact Bellman
  switching state and trajectory. Bellman and every DQN seed produced
  `J_A = 0.1932564107532301`, so both the median and maximum absolute relative
  errors were 0%.
- Per-seed training times were 508.45 s, 485.35 s, and 481.16 s. Full
  deterministic switching-candidate inference times were 168.48 ms, 204.58 ms,
  and 216.83 ms. These Phase 3 timings are diagnostics only and do not establish
  the final Bellman-versus-DQN computational regime.
- The Phase 1–3 DQN suite passed 32/32 tests. The directly related legacy
  Bellman/Tabular/Gym regression suite passed 27/27 tests.
- The user-approved Phase 3 sanity criterion passed. This result validates the
  single-terrain implementation and does not constitute a terrain-generalization
  claim.

#### 25 m Follow-up Run

At the user's request, the same single-terrain experiment was repeated at
`dx = dy = dh = 25 m`, with heading spacing unchanged at `5 deg`. Outputs are
isolated under
`3D_0827/figure/phase_3_single_terrain_dqn/dx25m_dpsi5deg/` and checkpoints
under `3D_RL_DQN/checkpoints/phase3_single_terrain/dx25m_dpsi5deg/`.

- The Cartesian grid contained 3,243,240 states, of which 2,316,566 were
  goal-reachable; the implicit graph represented 64,893,102 reachable edges.
- The exact Bellman result was `J_A = 0.18788998581562277` with a 10-edge glide
  trajectory. The official-run Bellman BR time was 5.658 s.
- DQN seeds `0, 1, 2` all returned feasible goal-reaching trajectories with
  relative `J_A` errors of 1.500%, 1.966%, and 1.498%, respectively. The median
  error was 1.500% and the maximum was 1.966%, so the approved sanity criterion
  passed.
- Per-seed DQN training times were 2,630.60 s, 2,431.52 s, and 3,024.63 s
  (8,086.76 s total). Full switching-candidate inference times were 4.285 s,
  4.766 s, and 5.183 s.
- The representative DQN trajectory differs visibly from Bellman: it switches
  at a lower altitude, reaches the goal in fewer glide edges, and incurs a small
  authoritative objective penalty. The new trajectory plot therefore exposes a
  difference hidden by the exact path overlap at 100 m.
- Because a Python-level full audit over all 2.32 million reachable states would
  dominate this follow-up, the fixed action catalog was audited on a documented
  deterministic set of 10,860 reachable states containing all switching
  candidates, the Bellman trajectory, terminal states, and a stratified graph
  sample. This checked 287,379 transitions, exercised all 72 action IDs, and
  found no sampled nonterminal dead end. Bellman and DQN execution still used
  the full authoritative implicit graph; only the diagnostic audit was sampled.
- This is still one known cube terrain. It does not establish terrain
  generalization or the final Bellman-versus-DQN regime boundary.

### Objective
Implement and validate the first condition-specific DQN attacker-BR solver on the canonical simple cube terrain before any multi-terrain training or generalization claim.

The purpose of Phase 3 is **implementation sanity and solver parity**, not paper-level performance evaluation. The DQN must demonstrate that the Phase 1 common attacker-BR environment and Phase 2 observation contract can support a neural approximation of the existing Bellman attacker BR on one known terrain/configuration.

The core question for this phase is:

> Given one fixed computational condition and one known cube terrain, can the DQN learn a feasible attacker policy whose authoritative existing `J_A` approaches the Bellman `J_A`, without changing the underlying attacker problem?

No terrain-generalization claim is allowed in Phase 3.

### Phase Boundary

Phase 3 may implement:

- DQN model construction,
- condition-specific fixed action indexing,
- feasible-action masking,
- experience replay,
- target-network logic,
- exploration policy,
- training/evaluation loops,
- checkpointing,
- diagnostic logging,
- single-terrain Bellman/DQN comparison.

Phase 3 must **not** implement:

- multi-terrain training,
- unseen-terrain evaluation,
- cross-discretization DQN,
- Local-SSE replacement/integration with DQN,
- final Monte Carlo performance experiments,
- new objective terms or reward shaping that changes the optimal attacker objective,
- terrain/scenario IDs as neural inputs.

### Repository-Grounded Preconditions

Before coding, Codex must inspect the latest completed Phase 1 and Phase 2 implementation plus the latest `3D_0827/` and `3D_RL_DQN/` repository state.

Resolve and record from current code, rather than guessing:

1. canonical cube case/configuration used for development validation,
2. Phase 1 attacker-BR transition/feasibility/terminal interfaces,
3. Phase 2 structured observation schema and tensor-ready conversion,
4. actual attacker state variables and terminal semantics,
5. actual Bellman stage-cost / trajectory-cost decomposition,
6. authoritative existing `J_A` evaluator,
7. current Tabular Q-learning reward/cost mapping, if useful for parity checks,
8. the existing result-folder convention,
9. any existing PyTorch/device/checkpoint utilities in `3D_RL_DQN/`,
10. any existing reproducibility/seed utilities.

Do not infer missing values from old snapshots, chat history, or generic DQN tutorials. If a required repository-derived value is absent, report it as unresolved.

### Condition-Specific DQN Contract

Current work uses **A1: condition-specific DQN**.

For one selected Phase 3 development condition,

```text
spatial condition:  dx = dy = dh = fixed value
angular condition:  dpsi = dgamma = fixed value
```

train one DQN whose observation shape and discrete action catalog remain fixed for that condition.

Do not include `dx`, `dy`, `dh`, `dpsi`, or `dgamma` as neural inputs in the current implementation.

Do not implement A2 cross-discretization support. Preserve only the existing future-extension log that A2 may be studied later.

The Phase 3 development discretization must be selected from the previously defined study domain unless current repository constraints show otherwise. If the repository contains no authoritative single development/default condition, Codex must ask the user to select the Phase 3 condition before training.

Local-SSE neighbor radius `r` is not part of the Phase 3 attacker-BR DQN and must not be inserted into the DQN observation/action definition.

### Fixed Discrete Action Catalog

DQN requires a fixed output dimension for one condition. Build that action space from the **Phase 1 authoritative feasible-transition semantics**, not from an independently invented RL action definition.

Codex must:

1. inspect the Phase 1 transition generator,
2. identify the solver-independent semantic descriptor of a candidate transition/action,
3. construct a deterministic canonical action-ID catalog for the selected discretization condition,
4. verify that each action ID maps back to exactly the same physical/discrete transition semantics used by the common attacker environment,
5. generate a state-dependent feasible-action mask from the Phase 1 feasibility logic.

The exact action descriptor must come from current code. Examples such as relative grid-index increments or angular-index changes may be used only if they match the actual transition model; do not assume those forms in advance.

At action selection time, infeasible actions must never be executed.

The same feasible-action mask must also be applied when computing the bootstrap maximum for the DQN target. An infeasible next-state action must not contribute to the target merely because the neural network assigned it a high Q value.

If a state has no feasible action, handle it according to the existing Phase 1 failure/terminal semantics. Do not invent a new terminal penalty unless it is mathematically required by the existing objective and explicitly documented.

### Reward / Cost Equivalence

The DQN training signal must preserve the attacker objective used by Bellman.

Codex must inspect the current Bellman recurrence / stage-cost decomposition and construct the RL reward from that authoritative cost structure such that maximizing DQN return produces the same policy ordering as minimizing the existing attacker objective.

Where the existing objective is an undiscounted additive finite-trajectory cost and code inspection confirms exact equivalence, the natural transformation may be of the form

```text
reward = - authoritative_stage_cost
```

with corresponding terminal handling. However, do **not** impose this form or a particular discount factor unless current code confirms it.

The DQN discount factor must be selected to preserve objective equivalence wherever possible. Do not choose a conventional value such as `0.99` merely because it is common in DQN.

Required equivalence validation:

- take one or more known feasible trajectories,
- compute the authoritative existing `J_A`,
- compute the cumulative RL training return under the proposed transformation,
- verify and document the mathematical/numerical relationship between the two,
- verify that policy ordering is preserved.

Final Bellman-vs-DQN comparison must always use the **authoritative existing `J_A` evaluator**, not the raw neural-network Q value or training return.

### DQN Network Interface

The network must consume the completed Phase 2 tensor-ready observation and output one Q value per canonical Phase 3 action ID.

Do not redesign the Phase 2 observation contract inside Phase 3.

The exact neural architecture is a **new DQN design decision**, not a repository fact. After inspecting the finalized Phase 2 observation schema, Codex must present the user with the architecture required/proposed for that schema and obtain confirmation before finalizing training architecture parameters.

If the Phase 2 observation contains both spatial channels and scalar/ego features, preserve their semantic separation unless the user explicitly approves a different encoding. Do not silently flatten a spatial representation solely for implementation convenience.

Architecture/configuration choices that require explicit user confirmation before full Phase 3 training include, as applicable:

- model family/branch structure appropriate to the Phase 2 observation,
- hidden/channel dimensions,
- activation choice if not already specified,
- optimizer,
- learning rate,
- replay-buffer capacity,
- batch size,
- target-network update rule/frequency,
- exploration schedule,
- replay warm-up requirement,
- gradient clipping if used,
- total training budget or stopping criterion.

Codex may prepare these as configuration fields before values are chosen, but must not silently populate unresolved new DQN hyperparameters with tutorial defaults and call Phase 3 complete.

### Training Loop Requirements

Implement a standard off-policy DQN training loop consistent with the approved Phase 3 configuration and the existing objective.

At minimum, each stored transition must preserve the information required for masked Bellman updates:

```text
observation_t
action_id
reward
observation_t_plus_1
done / terminal status
next_feasible_action_mask
```

Include any additional environment status required by the current code, but do not duplicate terrain/scenario identity as a neural feature.

Training must separate:

- **training mode**: exploration + replay updates,
- **evaluation mode**: deterministic/greedy policy execution with learning disabled.

Evaluation episodes used for Bellman comparison must not update network weights, replay state, normalization statistics, or exploration state.

If Phase 2 normalization requires fitted statistics, Phase 3 may fit them using only the canonical Phase 3 training case because this phase is explicitly single-terrain development. Save those statistics with the checkpoint. Do not reuse any future Phase 5 unseen test terrain to fit them.

### Reproducibility and Checkpointing

Use explicit random seeds for all stochastic components supported by the current stack, including as applicable:

- Python,
- NumPy,
- PyTorch CPU,
- PyTorch CUDA,
- environment/random start logic if present,
- replay sampling.

If no authoritative project-wide DQN seed exists, expose the seed as a Phase 3 configuration field and obtain/record the selected value rather than hiding it in code.

Save at least:

1. DQN configuration JSON,
2. observation schema/config reference,
3. action-catalog manifest,
4. normalization parameters if applicable,
5. final model checkpoint,
6. best-evaluation checkpoint if a best-model rule is used,
7. training/evaluation metric history,
8. Phase 3 run manifest.

A checkpoint must contain enough metadata to reject loading it under an incompatible discretization, observation schema, or action catalog.

### Runtime Logging for Phase 3

Phase 3 is not the final computation-regime experiment, but log enough timing information for debugging and later accounting.

At minimum record separately:

```text
dqn_training_runtime_sec
dqn_evaluation_inference_runtime_sec
bellman_br_runtime_sec
```

Also preserve the Phase 0/Phase 1 authoritative attacker-BR runtime field where the existing result convention requires it.

Do not yet define the final amortized DQN-vs-Bellman runtime comparison rule; that belongs to the later performance phases.

GPU may be used for DQN training. Device choice and CUDA availability must be logged, but GPU utilization is not an official comparison metric under the current study definition.

### Required Phase 3 Evaluation Metrics

For the same canonical cube case and same selected discretization condition, evaluate at minimum:

- DQN goal/success status under existing terminal semantics,
- trajectory feasibility,
- authoritative Bellman `J_A`,
- authoritative DQN `J_A`,
- absolute `J_A` difference,
- relative `J_A` error where numerically well-defined,
- Bellman attacker-BR runtime,
- DQN training runtime,
- DQN evaluation/inference runtime.

The relative error calculation must handle zero or near-zero Bellman `J_A` safely; do not divide blindly by `J_A^Bellman` when it is numerically unsuitable.

Tabular Q-learning may be shown as a regression/reference result on the same cube case when the existing pipeline makes this straightforward, but Phase 3 pass/fail is primarily Bellman-vs-DQN because the purpose is to validate the new DQN implementation.

Do not introduce `J_D` or Local-SSE conclusions into the Phase 3 DQN sanity criterion; Local-SSE integration occurs later.

### Required Development Visualizations

All Phase 3 figures are development/sanity outputs, not final paper figures.

Generate at least:

1. **Episode return / training-objective history**
   - x-axis: training episode or approved update unit,
   - y-axis: DQN training return / objective quantity,
   - clearly separate training values from deterministic evaluation values.

2. **TD-loss history**
   - show the actual optimization loss used for DQN updates.

3. **Evaluation success history**
   - deterministic evaluation success/goal-reach metric over training checkpoints.

4. **Bellman-relative `J_A` approximation history**
   - evaluate checkpoints using the authoritative `J_A` evaluator,
   - show absolute and/or relative gap over training.

5. **Bellman vs DQN trajectory comparison**
   - same cube terrain,
   - same defender/sensor configuration,
   - same start/goal,
   - same discretization,
   - PNG for ordinary plots,
   - HTML only if interactive 3D inspection is genuinely required.

6. **Action-mask sanity visualization**
   - for at least one representative state, show the canonical action IDs split into feasible and infeasible actions,
   - confirm the selected DQN action is feasible.

7. **Compact result summary**
   - selected computational condition,
   - Bellman `J_A`,
   - DQN `J_A`,
   - approximation error,
   - Bellman BR runtime,
   - DQN training runtime,
   - DQN inference runtime,
   - success/feasibility status.

### Required Phase 3 Tests

Implement automated tests where practical for:

1. **Action-catalog determinism**
   - same condition produces the same action-ID mapping.

2. **Action-catalog compatibility**
   - each action ID maps to the authoritative Phase 1 transition semantics.

3. **Feasible-action masking**
   - infeasible actions cannot be selected during exploration, greedy evaluation, or target bootstrap.

4. **Observation compatibility**
   - DQN input matches the Phase 2 schema exactly.

5. **Reward/cost equivalence**
   - known trajectories preserve the documented relationship between cumulative RL return and authoritative attacker cost.

6. **Terminal handling**
   - terminal transitions do not bootstrap beyond the existing terminal semantics.

7. **Replay transition integrity**
   - stored transitions preserve state/action/reward/next-state/terminal/mask consistency.

8. **Checkpoint compatibility**
   - incompatible observation/action/discretization metadata causes a clear load-time rejection rather than silent use.

9. **Evaluation isolation**
   - deterministic evaluation does not update network weights or replay contents.

10. **Legacy regression**
    - adding DQN code does not change Bellman, Tabular, Phase 1 environment behavior, or authoritative `J_A` evaluation.

### Phase 3 User-Confirmation Gates

Phase 3 contains new neural/RL design parameters that cannot be discovered from legacy code. Codex must stop before full training and explicitly obtain user confirmation when the following are not already fixed by a completed earlier decision:

#### Gate A — Single development discretization
Select the one `(spatial discretization, angular discretization)` condition used for the Phase 3 sanity run.

Use the project's allowed study domain, but do not arbitrarily choose one if no canonical development condition exists in current code.

#### Gate B — Neural architecture/configuration
After the finalized Phase 2 observation schema is known, present the proposed architecture/configuration needed to encode it and obtain confirmation.

#### Gate C — Training hyperparameter bundle
Present the DQN training hyperparameter bundle in one compact configuration for approval rather than asking about individual constants piecemeal where possible.

#### Gate D — Phase 3 pass criterion
Obtain a user-approved sanity threshold before declaring DQN approximation successful. At minimum this criterion should state:

- required trajectory/goal success behavior,
- acceptable Bellman-relative `J_A` error or equivalent tolerance,
- whether the criterion must hold for one fixed seed or multiple repeated seeds.

Do not invent a percentage threshold such as 80%, 90%, or 95% from the broader research motivation.

These gates may be resolved during actual Phase 3 execution after Phase 2 is complete; they do not need to be guessed during planning.

### Phase 3 JSON Manifest

Under the existing result-folder convention, save a Phase 3 JSON manifest containing at least:

```text
phase
terrain_case
spatial_discretization_m
angular_discretization_deg
observation_schema_id
action_catalog_id
network_config
training_config
random_seed
device
bellman_J_A
dqn_J_A
J_A_absolute_error
J_A_relative_error
bellman_br_runtime_sec
dqn_training_runtime_sec
dqn_evaluation_inference_runtime_sec
dqn_success
trajectory_feasible
checkpoint_paths
visualization_paths
test_results
status
```

Use actual repository field names/conventions where equivalents already exist instead of creating redundant schemas.

### Detailed Codex Command

When Codex actually executes Phase 3, use the following instruction as the implementation contract:

> Work only on **Phase 3 — Single-Terrain DQN Sanity Check**. Apply the global repository-grounding rule and begin from the completed Phase 1 common attacker-BR environment and completed Phase 2 observation builder. Inspect the latest `3D_0827/` and `3D_RL_DQN/` code and current Phase 1/2 manifests before modifying anything. Use the canonical cube case only as a development/sanity environment and do not make terrain-generalization claims. Keep the current work condition-specific: train one DQN for one explicitly selected `(dx=dy=dh, dpsi=dgamma)` condition, do not put discretization values or Local-SSE radius `r` into the neural state, and do not implement cross-discretization DQN. Derive a deterministic fixed action-ID catalog from the Phase 1 authoritative transition semantics for that condition and use Phase 1 feasibility logic to create state-dependent action masks. Apply the mask both when selecting behavior/evaluation actions and when maximizing over next-state actions in the DQN target. Do not invent an independent RL motion model. Derive the training reward from the authoritative Bellman stage/trajectory cost so maximizing expected DQN return preserves the Bellman attacker objective; do not assume `reward=-cost`, `gamma=0.99`, or any other conventional DQN setting without verifying objective equivalence from current code. Validate the reward/`J_A` relationship numerically on known feasible trajectories, and always evaluate final DQN trajectories using the existing authoritative `J_A` evaluator. Consume the finalized Phase 2 observation contract without redesigning it. Before full training, resolve the Phase 3 user-confirmation gates: the single development discretization if no canonical one exists, the neural architecture appropriate to the actual Phase 2 observation schema, the complete DQN training hyperparameter bundle, and the Bellman-relative sanity/pass threshold including the required seed policy. Implement the approved DQN network, replay buffer, target-network logic, exploration policy, masked update loop, deterministic evaluation loop, seed handling, checkpoint compatibility metadata, and JSON logging. Keep evaluation isolated from training. Use GPU/CUDA for DQN training when the approved/current environment supports it and record the selected device. Log DQN training time and inference time separately from Bellman BR time, but do not yet define final amortized runtime conclusions. Generate development plots for training return, TD loss, deterministic evaluation success, Bellman-relative `J_A` gap, Bellman-vs-DQN trajectory comparison, action-mask sanity, and a compact result summary. Implement automated tests for action mapping, masking, observation compatibility, reward/cost equivalence, terminal handling, replay integrity, checkpoint compatibility, evaluation isolation, and legacy Bellman/Tabular regression. Do not implement multi-terrain training, unseen-terrain testing, Local-SSE DQN integration, final Monte Carlo evaluation, objective changes, or A2 cross-discretization support. At completion report all repository-derived facts, all user-confirmed DQN design choices, files changed, test results, checkpoint/config paths, visualization paths, unresolved issues, and whether the user-approved Phase 3 sanity criterion was met.

### Exit Criteria

Phase 3 is complete only when:

1. one explicitly selected condition-specific DQN configuration is recorded,
2. a deterministic fixed action catalog is derived from Phase 1 semantics,
3. infeasible-action masking is applied consistently in behavior selection, evaluation, and target bootstrap,
4. reward/cost equivalence to the existing attacker objective is documented and tested,
5. DQN training runs successfully on the canonical cube case,
6. deterministic DQN evaluation returns a trajectory that satisfies the existing feasibility and terminal rules,
7. Bellman and DQN trajectories are both evaluated with the same authoritative `J_A` implementation,
8. the user-approved Phase 3 Bellman-relative sanity criterion is evaluated,
9. required runtime diagnostics are recorded,
10. required checkpoints/configuration/manifest files are saved,
11. required development visualizations are produced,
12. automated Phase 3 tests pass or any failures are explicitly unresolved,
13. Bellman, Tabular Q-learning, Phase 1 environment behavior, and existing objective logic remain unchanged,
14. no terrain-generalization, Local-SSE, or final computational-regime conclusion is claimed.

Phase 3 is not complete if conventional DQN defaults are silently substituted for unresolved new design choices, if the DQN uses an action/reward model different from Bellman, if infeasible actions enter the target maximum, if `J_A` comparison is performed using raw Q values instead of the authoritative evaluator, or if a successful cube result is presented as evidence of terrain generalization.

---

## Phase 4 — Multi-Terrain Generalized DQN Training

### Objective
Extend the validated Phase 3 condition-specific DQN from one canonical cube case to a **single shared DQN trained across multiple simple terrain/scenario instances**, while preserving the exact same attacker-BR problem semantics, observation contract, action catalog, objective, and discretization condition.

The purpose of Phase 4 is to answer the implementation-level question:

> Can one fixed-condition DQN learn a reusable attacker-BR policy across multiple simple terrain geometries and associated sensing conditions, without retraining a separate network for every terrain?

Phase 4 is the first phase that intentionally trains across multiple terrain instances. It is **not** the final unseen-complex-terrain performance experiment and must not use any Phase 5 test terrain for training, normalization, model selection, hyperparameter tuning, or stopping decisions.

### Phase Boundary

Phase 4 may implement:

- multi-terrain training-set management,
- simple-terrain train/validation split management,
- per-episode terrain/scenario sampling,
- multi-terrain replay-buffer population,
- training-distribution logging,
- training-set-only normalization fitting where required by Phase 2,
- periodic deterministic evaluation across multiple training/validation cases,
- Bellman-reference generation/caching for those development cases,
- generalized-DQN checkpointing and model selection,
- multi-terrain training visualizations and diagnostics.

Phase 4 must **not** implement:

- Phase 5 complex unseen-terrain testing,
- cross-discretization DQN (A2),
- Local-SSE integration with DQN,
- final Monte Carlo computational-regime sweeps,
- modification of `J_A`, attacker dynamics, feasibility, terminal logic, sensing model, or Bellman recurrence,
- terrain/scenario/defender IDs as neural-network inputs,
- terrain-specific network branches or separate per-terrain models that defeat the purpose of one generalized DQN,
- use of future test terrain statistics for normalization or tuning.

### Repository-Grounded Preconditions

Before coding, Codex must inspect the latest completed Phase 0–3 implementation plus the current `3D_0827/` and `3D_RL_DQN/` repository state.

Resolve from current code rather than guessing:

1. the finalized Phase 1 common attacker-BR environment and solver interfaces,
2. the finalized Phase 2 observation schema and normalization mechanism,
3. the finalized Phase 3 DQN architecture, action catalog, feasible-action masking, replay implementation, reward/cost mapping, and checkpoint format,
4. the currently approved Phase 3 DQN hyperparameter bundle,
5. existing simple-terrain generators/configurations already present in the repository,
6. existing defender/sensor configuration generators or candidate-location logic,
7. which scenario variables are fixed by the current P1b formulation and which are already configurable,
8. existing terrain/scenario serialization or configuration formats,
9. existing result-directory convention,
10. existing reproducibility/seed utilities.

Do not infer numeric values from old repository snapshots, prior chat text, or generic DQN examples. If a value is already defined by current code, use and record that value. If it is a new Phase 4 design choice, expose it as such and obtain user confirmation where required below.

### Condition-Specific DQN Contract Remains Active

Phase 4 continues **A1: condition-specific DQN**.

For one approved computational condition,

```text
spatial condition:  dx = dy = dh = fixed value
angular condition:  dpsi = dgamma = fixed value
```

train one shared DQN across multiple terrain/scenario instances.

The following must remain identical across all Phase 4 training and validation cases for one model:

- spatial discretization,
- angular discretization,
- observation tensor shape/schema,
- action-ID catalog,
- DQN network architecture,
- objective definition,
- transition/feasibility semantics.

Do not implement A2 cross-discretization support. Preserve only the existing future-extension log.

Local-SSE neighbor radius `r` remains outside the attacker-BR DQN and is not a Phase 4 training variable.

### Multi-Terrain Dataset Contract

Create explicit, persistent manifests for three logically distinct terrain groups:

```text
TRAIN_SIMPLE
VALIDATION_SIMPLE
RESERVED_PHASE5_TEST
```

Only `TRAIN_SIMPLE` may contribute learning transitions or fitted training statistics.

`VALIDATION_SIMPLE` is used only for deterministic development evaluation, checkpoint/model-selection diagnostics, and overfitting detection. It must not update network weights or replay state.

`RESERVED_PHASE5_TEST` contains identifiers/configuration references for the future complex unseen terrains if they already exist at this stage. Phase 4 must not load their geometry/hazard data for learning, normalization, tuning, checkpoint selection, or early stopping. If Phase 5 terrains have not yet been created, reserve the namespace/manifest mechanism now and populate it later.

For procedural terrain generation, save the generator configuration and random seed for every generated terrain. For file-based terrain definitions, save stable file/config identifiers. Use hashes where practical to detect accidental overlap or mutation.

Add an automated leakage check that rejects any terrain/scenario appearing in more than one split.

### Training-Terrain Design Gate — USER CONFIRMATION REQUIRED

The number, families, parameter ranges, and generation rules for the **simple terrain examples used for Phase 4 training and validation are new experiment-design choices**, not values that should be invented from generic RL practice.

Before generating the final Phase 4 dataset, Codex must:

1. inspect which simple terrain generators/types already exist in the latest repository,
2. summarize the available terrain families and tunable parameters,
3. propose a concrete `TRAIN_SIMPLE` and `VALIDATION_SIMPLE` composition,
4. ask the user to approve or modify that composition,
5. record the approved design in the Phase 4 dataset manifest.

Do not silently decide the number of terrains, terrain-family mixture, geometric parameter ranges, or train/validation split size.

### Defender / Sensing-Condition Training Contract

The DQN observation is expected to include sensing-hazard information according to the finalized Phase 2 design. Therefore Phase 4 must determine, from the current P1b code, how defender configuration `d` changes the attacker hazard field and how those configurations are generated or enumerated.

The DQN must not receive a raw defender-case ID or terrain ID as a neural feature.

If the existing problem formulation already specifies a fixed defender/sensor configuration for the attacker-BR subproblem, preserve that behavior unless the user explicitly approves variation.

If multiple defender/sensor configurations are required to teach the DQN a reusable hazard-conditioned BR policy, the selection/range/distribution of those configurations is a **new Phase 4 training-distribution choice** and must be presented to the user for confirmation before final training.

Store defender/sensor configuration as scenario metadata for reproducibility and diagnostics even when it is not a neural input.

### Non-Target Scenario Variables

Do not introduce unnecessary randomization.

Attacker start state, goal definition, mission constraints, sensor modality parameters, and any other existing P1b quantities must remain exactly as defined by the current code unless one of the following is true:

1. current code already treats that quantity as part of the scenario distribution, or
2. the user explicitly approves adding it as a Phase 4 generalization variable.

Terrain generalization does not automatically authorize start/goal, physics, objective, or sensor-model randomization.

### Episode-Level Scenario Sampling

Each training episode must select a scenario from the approved `TRAIN_SIMPLE` distribution before environment reset.

The sampling policy must be configurable and explicitly logged. Examples such as uniform terrain sampling, weighted family sampling, or curriculum sampling are possible design choices, but Codex must not choose one silently.

Before final Phase 4 training, Codex must show the user the proposed sampling distribution if it is not already dictated by the approved dataset design.

At minimum, log per episode:

```text
terrain_instance_id   # metadata only; never neural input
defender/scenario_id  # metadata only; never neural input
seed
episode_return
terminal/success status
trajectory length
```

plus any approved diagnostics already present in Phase 3.

### Replay Buffer Across Terrains

Reuse the validated Phase 3 DQN/replay implementation unless a change is mathematically or technically necessary.

The replay buffer may mix transitions from different `TRAIN_SIMPLE` terrain instances because the network is intentionally shared across terrains.

However:

- terrain/scenario identity may be stored as non-neural metadata for diagnostics,
- terrain/scenario identity must not be appended to the DQN observation merely to make memorization easier,
- the stored observation itself must contain only the approved Phase 2 state representation,
- the stored feasible-action mask must continue to follow Phase 1 semantics,
- transitions from validation or Phase 5 test terrains must never enter the training replay buffer.

Add diagnostics showing how many replay transitions originate from each training terrain/family so that severe sampling imbalance can be detected.

### Observation and Normalization Rules

Do not redesign the Phase 2 observation schema in Phase 4.

All terrain instances under one condition-specific model must produce the same observation structure and tensor shape.

If Phase 2 normalization uses fitted dataset statistics, fit them **only from `TRAIN_SIMPLE` data** and freeze them before deterministic validation.

Validation and future Phase 5 test observations must use the frozen training-derived normalization parameters and must not update them.

If Phase 2 uses physically defined fixed scaling rather than fitted statistics, preserve that mechanism unchanged.

Save the finalized normalization configuration/statistics with the generalized DQN checkpoint.

### Network / Action / Reward Invariance

Phase 4 starts from the validated Phase 3 model contract.

Unless a repository-grounded incompatibility is discovered:

- preserve the Phase 3 network architecture,
- preserve the condition-specific action catalog,
- preserve feasible-action masking,
- preserve reward/cost equivalence to the Bellman attacker objective,
- preserve terminal handling,
- preserve deterministic evaluation logic.

Do not alter architecture or reward shaping simply because training now spans multiple terrains.

If Phase 4 genuinely requires a change to the Phase 3 architecture or training hyperparameters, Codex must explain why and obtain user confirmation before making the change.

### Phase 4 Hyperparameter Rule

Use the user-approved Phase 3 DQN hyperparameter configuration as the initial Phase 4 configuration.

Any Phase 4-specific change to, for example:

- learning rate,
- replay capacity,
- batch size,
- target-network update,
- exploration schedule,
- total training budget,
- optimizer,
- architecture size,
- evaluation frequency,
- checkpoint-selection rule,
- early-stopping rule,

must be documented as a new training-design choice rather than silently tuned against future Phase 5 data.

If these values are not already sufficient to run multi-terrain training, Codex must propose the necessary modifications and request user confirmation.

### Bellman Reference Set for Development Evaluation

Create a deterministic Phase 4 evaluation suite composed only of approved `TRAIN_SIMPLE` and `VALIDATION_SIMPLE` cases.

For each evaluation case, obtain the Bellman attacker BR through the Phase 1 common interface and save/cache:

```text
scenario configuration
Bellman trajectory
Bellman J_A
Bellman attacker-BR runtime
status / feasibility metadata
```

Bellman remains the authoritative fixed-discretization attacker-BR reference.

Reuse cached Bellman results only when the scenario configuration, discretization, objective version, and environment signature match exactly. Otherwise recompute.

Do not require Bellman computation for every stochastic training episode. Use a fixed evaluation suite at configured intervals to avoid turning development evaluation into the dominant training cost.

### Deterministic Multi-Terrain Evaluation

At configured evaluation intervals, disable exploration and learning and evaluate the current DQN on the fixed Phase 4 evaluation suite.

For each case, record at minimum:

```text
DQN success / failure
DQN trajectory
DQN authoritative J_A
Bellman authoritative J_A
raw J_A difference
Bellman-relative J_A error if mathematically well-defined
DQN inference runtime
```

Always recompute DQN trajectory cost using the authoritative existing `J_A` evaluator.

Do not use neural Q values or episode return as the final approximation-quality metric.

If the relative-error denominator can be zero or numerically unstable under the current objective, do not invent an ad hoc denominator. Record raw objective differences and ask for/derive an appropriate normalized metric later.

### Generalization Diagnostics Inside Phase 4

Phase 4 must distinguish at least:

1. performance on `TRAIN_SIMPLE`,
2. performance on held-out `VALIDATION_SIMPLE` from the approved simple-terrain distribution.

This allows detection of terrain memorization before moving to complex unseen testing.

The Phase 4 result must not claim successful final terrain generalization solely because `TRAIN_SIMPLE` performance is high.

If training error decreases while validation performance does not improve or degrades, flag potential overfitting explicitly.

### Checkpointing and Model Selection

Save enough information to reproduce and reject incompatible checkpoints.

At minimum, the generalized checkpoint bundle must include:

1. model weights,
2. condition-specific discretization metadata,
3. Phase 2 observation schema/config,
4. action-catalog signature,
5. reward/objective signature,
6. normalization configuration/statistics,
7. approved training/validation terrain manifests,
8. defender/scenario distribution configuration,
9. Phase 4 DQN hyperparameter configuration,
10. random seeds,
11. training/evaluation history,
12. repository/config version information where available.

The exact rule for selecting the "best" generalized checkpoint is a **Phase 4 design decision**. If no user-approved rule already exists, Codex must propose one based only on the Phase 4 validation suite and request confirmation before using it as the official model-selection criterion.

Phase 5 test terrain performance must never influence checkpoint selection.

### Runtime Logging for Phase 4

Phase 4 is still a development/training phase, not the final computational-regime experiment, but log the components needed for later accounting.

At minimum record:

```text
generalized_dqn_total_training_runtime_sec
phase4_evaluation_runtime_sec
per_case_dqn_inference_runtime_sec
per_case_bellman_reference_runtime_sec
```

Do not yet collapse these into a final Bellman-vs-DQN amortized-use rule. That accounting belongs to the later experimental phases.

GPU may be used for DQN training. Record the training device/CUDA status for reproducibility, but GPU utilization remains outside the current official performance metric definition.

### Required Visualizations / Outputs

Produce development visualizations sufficient to inspect whether multi-terrain training is functioning correctly.

At minimum:

#### 1. Training Terrain Gallery
Show the approved `TRAIN_SIMPLE` terrain instances/families with stable labels.

Use PNG for static terrain views. Use HTML only where interactive 3D inspection is materially necessary.

#### 2. Validation Terrain Gallery
Show the `VALIDATION_SIMPLE` terrains separately so train/validation membership is visually obvious.

#### 3. Scenario Sampling Distribution
Plot the number/fraction of training episodes or replay transitions contributed by each terrain/family and, where varied, each defender/sensing configuration class.

#### 4. Aggregate Learning Curve
Plot approved aggregate training metrics over training progress.

#### 5. Train-vs-Validation Bellman Gap
Plot Bellman-referenced authoritative `J_A` approximation performance separately for `TRAIN_SIMPLE` and `VALIDATION_SIMPLE`.

#### 6. Per-Terrain Evaluation Plot
For each Phase 4 evaluation terrain/scenario, show DQN `J_A`, Bellman `J_A`, success/failure, and objective difference over evaluation checkpoints.

#### 7. Representative Trajectory Comparisons
For selected training and validation terrains, overlay:

- Bellman trajectory,
- deterministic DQN trajectory,
- terrain,
- relevant sensing-hazard visualization.

#### 8. Overfitting Diagnostic
Produce a compact plot/table showing whether training performance improves while held-out simple-terrain performance stagnates or degrades.

All non-3D plots must be saved as PNG under the existing result-folder convention.

### Required JSON Outputs

Extend the existing JSON result convention rather than creating an unrelated format.

At minimum save:

```text
phase4_dataset_manifest.json
phase4_training_config.json
phase4_generalized_model_manifest.json
phase4_training_history.json
phase4_evaluation_history.json
phase4_bellman_reference_manifest.json
```

The exact filenames may follow the current repository convention, but all equivalent information must be present.

### Automated Validation Tests

Add tests for at least the following:

1. all Phase 4 terrain splits are disjoint,
2. validation/test terrains never enter replay,
3. one shared DQN checkpoint is used across all Phase 4 terrain instances for a fixed condition,
4. observation schema and tensor shape are identical across all training/validation terrains,
5. the Phase 3 action catalog remains valid across all Phase 4 cases under the same discretization,
6. feasible-action masking remains identical to Phase 1 semantics,
7. reward/cost equivalence remains unchanged from Phase 3,
8. normalization statistics are fitted only from training data when fitted normalization is used,
9. evaluation mode does not update weights, replay, exploration state, or normalization statistics,
10. terrain/scenario IDs are absent from neural input tensors,
11. the same feasible trajectory receives the same authoritative `J_A` regardless of which solver wrapper evaluates it,
12. saved checkpoints reject incompatible observation/action/discretization signatures,
13. Phase 0/1 Bellman regression remains unchanged,
14. Phase 3 canonical-cube sanity behavior does not regress beyond the user-approved tolerance unless explicitly documented.

### Failure / Stop Conditions

Do not declare Phase 4 complete if any of the following occurs:

- train/validation/test leakage,
- separate terrain-specific DQN models are silently used instead of one shared model,
- terrain or defender IDs are injected as neural features,
- observation shape changes across terrain instances,
- action semantics differ between terrain instances under the same condition,
- reward/objective definition changes relative to Bellman,
- validation data updates training statistics or weights,
- the official checkpoint is chosen using Phase 5 test performance,
- new hyperparameters or sampling rules are silently selected without required user approval,
- only training-terrain performance is shown and held-out simple-terrain behavior is unknown.

### Phase 4 User-Confirmation Gates

The following are new Phase 4 design choices and must be confirmed by the user at execution time if they have not already been resolved:

1. **Simple terrain training/validation composition**
   - terrain families/types,
   - number of terrain instances,
   - geometric parameter ranges,
   - train/validation split.

2. **Defender/sensing-condition variation**
   - whether `d` is varied during Phase 4 training,
   - allowed defender/sensor configurations,
   - sampling distribution across those configurations.

3. **Episode/scenario sampling policy**
   - uniform, weighted, curriculum, or another approved distribution.

4. **Any Phase 4 changes to Phase 3 DQN hyperparameters/training budget**
   - only if the Phase 3 configuration is insufficient for multi-terrain training.

5. **Generalized-checkpoint selection criterion**
   - the validation metric/rule used to identify the official Phase 4 pretrained DQN.

Codex must first inspect the latest repository and propose concrete options grounded in available terrain/sensor generators before asking the user to choose. Do not ask the user for repository facts that Codex can determine directly from code.

### Exit Criteria

Phase 4 is complete only when:

1. the approved `TRAIN_SIMPLE` and `VALIDATION_SIMPLE` datasets are created and saved with reproducible manifests,
2. leakage checks pass and Phase 5 test data remains untouched,
3. one condition-specific shared DQN is trained across the approved multi-terrain training distribution,
4. Phase 1 transition/feasibility/terminal semantics remain unchanged,
5. Phase 2 observation contract remains unchanged and fixed-shape across terrains,
6. Phase 3 action/reward/masking semantics remain unchanged unless an explicitly approved modification was necessary,
7. deterministic evaluation is performed on both training and held-out simple validation terrains,
8. Bellman reference `J_A` values are available for the fixed Phase 4 evaluation suite,
9. generalized DQN trajectories are re-evaluated by the authoritative `J_A` implementation,
10. train-vs-validation behavior is visualized and overfitting can be assessed,
11. the user-approved model-selection rule identifies an official generalized Phase 4 checkpoint,
12. all required configuration, dataset, normalization, checkpoint, runtime, evaluation, and visualization artifacts are saved,
13. automated Phase 4 tests pass or unresolved failures are explicitly documented,
14. no claim about complex unseen-terrain performance, Local SSE performance, or final Bellman-vs-RL computational regime is made.

Phase 4 is specifically an implementation/training milestone demonstrating reusable learning across an approved family of simple terrains under one fixed discretization condition. Final evaluation on complex, unseen terrains belongs exclusively to Phase 5.

## Phase 5 — Unseen-Terrain Generalization Test

### Objective
Evaluate whether the **frozen Phase 4 terrain-generalized DQN** can approximate the attacker best response on **complex terrain instances that were never used for training or Phase 4 validation**, without terrain-specific retraining or fine-tuning.

Phase 5 is an **attacker-BR generalization test**. It compares, on exactly the same unseen attacker-BR cases:

- Bellman DP — exact discrete attacker BR at the selected discretization condition,
- Tabular Q-learning — terrain-specific RL approximation retrained for each unseen case,
- Phase 4 DQN — one pretrained generalized model used without terrain-specific retraining.

The purpose is to determine whether the DQN has learned a reusable terrain/hazard-conditioned attacker policy rather than merely memorizing the Phase 4 simple training geometries.

Phase 5 is **not** the Local-SSE integration phase and **not** the final computational-condition Monte Carlo sweep across all discretizations. Those belong to Phase 6 and Phase 7 respectively.

### Phase Boundary

Phase 5 may implement:

- frozen unseen-test-suite manifests,
- complex unseen-terrain generation/loading,
- Bellman attacker-BR evaluation on unseen cases,
- per-case Tabular Q-learning retraining/evaluation,
- frozen DQN inference on unseen cases,
- common feasibility and `J_A` re-evaluation,
- attacker-BR runtime instrumentation,
- repeated stochastic Tabular trials where approved,
- aggregate generalization statistics,
- unseen-case trajectory and performance visualizations,
- strict train/validation/test leakage checks.

Phase 5 must **not** implement:

- DQN fine-tuning on test terrains,
- DQN replay-buffer updates from test trajectories,
- normalization-statistics updates from test terrains,
- architecture or hyperparameter tuning using Phase 5 outcomes,
- cross-discretization DQN (A2),
- DQN integration into the Local SSE defender search,
- `J_D` comparison,
- Local-SSE neighbor-radius `r` experiments,
- final full-domain Monte Carlo sweeps over all `dx`, `dpsi`, and `r`,
- changes to `J_A`, dynamics, feasibility, sensing model, or Bellman recurrence.

### Repository-Grounded Preconditions

Before implementing or running Phase 5, Codex must inspect the latest completed Phase 0–4 implementation and current `3D_0827/` and `3D_RL_DQN/` repositories.

Resolve from the actual current code/configuration rather than guessing:

1. the finalized common attacker-BR interface from Phase 1,
2. the finalized Phase 2 observation schema and frozen normalization configuration,
3. the finalized Phase 3 action catalog and feasible-action masking,
4. the official Phase 4 generalized DQN checkpoint and its condition metadata,
5. the exact Phase 4 `TRAIN_SIMPLE` and `VALIDATION_SIMPLE` manifests,
6. the reserved Phase 5 test-manifest mechanism created in Phase 4,
7. current complex-terrain generators or terrain configurations available in the repositories,
8. current attacker start/goal/sensor/defender configuration mechanisms,
9. the authoritative common trajectory-feasibility checker,
10. the authoritative `J_A` trajectory evaluator,
11. existing Tabular Q-learning training settings and seed-handling behavior,
12. current runtime/result-directory conventions.

Do not infer any of these from old snapshots, prior chat text, or generic RL practice. If current code does not resolve a needed quantity, report it as unresolved or expose it as a user-confirmed Phase 5 design choice.

### Condition-Specific DQN Contract Remains Active

Phase 5 continues **A1: condition-specific DQN**.

For each Phase 5 evaluation batch, use one frozen Phase 4 DQN associated with one exact computational condition:

```text
spatial discretization:  dx = dy = dh = fixed value
angular discretization:  dpsi = dgamma = fixed value
```

Bellman, Tabular Q-learning, and DQN must all solve the **same discretized attacker problem** under that condition.

Do not evaluate a DQN checkpoint at a different spatial/angular discretization than the one for which it was trained unless a future phase explicitly implements A2.

Local-SSE neighbor radius `r` is irrelevant to Phase 5 attacker-BR evaluation and must not be treated as a DQN input or Phase 5 variable.

### Frozen DQN Test Contract

Before the first official Phase 5 test run for a condition:

1. select the official Phase 4 checkpoint using the Phase 4-approved checkpoint rule,
2. save its checkpoint ID/path/hash,
3. save the associated observation schema,
4. save the associated normalization parameters,
5. save the condition metadata,
6. save the DQN architecture/configuration metadata,
7. switch the model to deterministic evaluation mode.

During Phase 5 official evaluation:

- no gradient updates,
- no replay-buffer insertion,
- no epsilon/exploratory actions,
- no target-network updates,
- no normalization updates,
- no terrain-specific fine-tuning,
- no test-terrain-specific network selection.

The same frozen checkpoint must be used for all Phase 5 cases belonging to that condition-specific evaluation batch.

If test results motivate architecture, state, reward, hyperparameter, or training-distribution changes, those changes must be made by returning to the appropriate earlier phase. The already-inspected Phase 5 terrains must then be treated as development data; a **new untouched test set** is required for a new official generalization claim. Pure implementation-bug fixes may be rerun on the same test suite only if the bug did not use test performance to redesign the algorithm; document the bug and fix explicitly.

### Unseen Complex-Terrain Test-Set Gate — USER CONFIRMATION REQUIRED

The exact definition of **complex unseen terrain** is an experiment-design choice and must not be invented silently.

Before creating/populating the official Phase 5 test set, Codex must:

1. inspect the current repository for available terrain families/generators and their tunable geometry parameters,
2. show which terrain families/parameter ranges were already used in `TRAIN_SIMPLE` and `VALIDATION_SIMPLE`,
3. identify available terrain families/configurations that can serve as genuinely unseen and more complex test cases,
4. propose a concrete `RESERVED_PHASE5_TEST` composition,
5. explain why each proposed test family/parameter range is unseen relative to Phase 4,
6. ask the user to approve or modify the proposed complex test set,
7. freeze the approved test manifest before official model evaluation.

The user-approved test manifest must specify, as applicable:

- terrain family/type,
- number of terrain instances,
- generator/configuration parameter ranges,
- terrain-generation seeds,
- file/config identifiers,
- any explicit complexity-control parameters already supported by the repository.

Do not invent a terrain-complexity scalar if the repository does not already define one. A future analysis may introduce one separately, but Phase 5 does not require an artificial complexity index.

### Train / Validation / Test Separation

The following must remain disjoint:

```text
TRAIN_SIMPLE
VALIDATION_SIMPLE
RESERVED_PHASE5_TEST
```

Before official Phase 5 execution, perform a leakage audit using stable identifiers and hashes where practical.

Reject the run if:

- a test terrain appears in either Phase 4 split,
- a procedural test terrain reuses an identical generator seed/configuration already used in Phase 4,
- test-terrain observations contributed to fitted normalization statistics,
- test results were used to select among Phase 4 checkpoints.

Store the leakage-audit result in JSON.

### Scenario-Variable Control

The primary novelty tested in Phase 5 is **terrain generalization**.

Therefore, do not silently introduce additional out-of-distribution changes in attacker start state, goal, physics, objective, sensor model, or defender/sensing configuration.

Codex must inspect the approved Phase 4 training distribution and classify each non-terrain scenario variable as:

```text
FIXED
VARIED_WITHIN_TRAINING_DISTRIBUTION
NOT_TRAINED_FOR_GENERALIZATION
```

For the main Phase 5 terrain-generalization test:

- preserve fixed variables exactly,
- if a variable such as defender configuration was intentionally varied during Phase 4, evaluate it only according to the user-approved Phase 5 scenario design,
- do not simultaneously test an unseen sensing-model or unseen physics regime unless the user explicitly creates a separate experiment for that purpose.

This prevents a DQN failure from being incorrectly attributed to terrain generalization when multiple unrelated distribution shifts were introduced at once.

### Defender / Hazard Handling

For every unseen terrain case and defender configuration `d`:

1. generate/query the sensing hazard using the same authoritative implementation used by Bellman and Tabular Q-learning,
2. construct the DQN hazard observation using the frozen Phase 2 observation pipeline,
3. do not pass terrain ID or defender-case ID as neural inputs,
4. verify that Bellman, Tabular, and DQN are evaluated against the same hazard field/problem instance.

If Phase 4 used multiple defender/sensing configurations, Codex must propose the Phase 5 defender-case evaluation design after inspecting the actual Phase 4 distribution and obtain user confirmation if a new distribution/selection rule is required.

### Common Solver Comparison Contract

For every approved unseen test case, evaluate all three methods through the common attacker-BR interface:

```text
Bellman DP
Tabular Q-learning
Frozen generalized DQN
```

#### Bellman DP

- solve the exact discrete attacker BR under the selected Phase 5 condition,
- use the existing Bellman recurrence unchanged,
- return the Bellman trajectory and authoritative `J_A`,
- use Bellman as the reference solution for approximation error.

#### Tabular Q-learning

- retrain from the approved initial condition for each unseen terrain/scenario case,
- do not reuse a Q-table trained on another terrain unless the existing approved Tabular baseline explicitly defines such behavior,
- use the approved existing Tabular training configuration rather than tuning it on Phase 5 cases,
- include case-specific training plus final policy/trajectory extraction in the attacker-BR runtime,
- evaluate the returned trajectory using the common authoritative `J_A` evaluator.

#### DQN

- load the official frozen Phase 4 checkpoint once for the evaluation batch,
- perform deterministic inference only,
- do not update weights or normalization,
- return the trajectory through the same attacker-BR result contract,
- evaluate the returned trajectory using the same authoritative `J_A` evaluator.

### Trajectory Feasibility and Objective Re-Evaluation

Do not trust solver-internal reward, Bellman table values, or neural Q-values as the final comparison metric.

For every returned trajectory from every method:

1. run the common trajectory-feasibility checker,
2. verify goal/terminal success using the authoritative terminal logic,
3. recompute `J_A` through the common objective evaluator,
4. record any mismatch between solver-internal reported cost and common re-evaluated cost as a diagnostic error.

The main Phase 5 comparison uses the re-evaluated `J_A`.

If a method fails to produce a feasible goal-reaching trajectory:

- record the run as failure,
- preserve the reason/status,
- do not fabricate a finite `J_A` unless the existing authoritative evaluator already defines a valid failure/terminal penalty,
- include the failure in success-rate statistics.

### Bellman-Relative Approximation Metrics — USER CONFIRMATION IF NOT ALREADY DEFINED

Phase 5 must record the raw objective values:

```text
J_A_bellman
J_A_tabular
J_A_dqn
```

and raw differences:

```text
Delta_J_A_tabular = J_A_tabular - J_A_bellman
Delta_J_A_dqn     = J_A_dqn     - J_A_bellman
```

For a normalized percentage/accuracy-style metric, first inspect whether the current Bellman-vs-Tabular comparison already defines an approved formula.

- If an approved repository-grounded formula already exists, reuse it.
- If no normalized formula exists, Codex must present the proposed normalization to the user before using terms such as `relative error`, `accuracy`, or `% optimality gap` in official Phase 5 outputs.

Do not silently define `accuracy = 1 - relative_error` or choose an epsilon denominator without approval.

The eventual Bellman-vs-RL acceptance threshold is **not** selected in Phase 5 unless the user explicitly chooses one. Phase 8 derives the final computational-regime rule from measured evidence.

### Runtime Accounting

Record attacker-BR runtime consistently for all methods.

#### Bellman

```text
bellman_br_runtime_sec
```

Use the same Phase 0/1 timing boundary from BR invocation to returned attacker-BR result.

#### Tabular Q-learning

```text
tabular_br_runtime_sec
```

Include the complete terrain-specific attacker-BR process required for that unseen case:

```text
case-specific training + final deterministic policy/trajectory extraction
```

#### DQN

Record at least:

```text
dqn_br_inference_runtime_sec
```

Measure deterministic attacker-BR inference from the already-loaded frozen model to the returned BR result. If GPU is used, synchronize appropriately before/after timing so asynchronous CUDA execution does not under-report runtime.

Also record, as diagnostics/provenance rather than automatically adding them to every test case:

```text
dqn_model_load_runtime_sec       # one-time load cost, if measured
dqn_phase4_training_runtime_sec  # inherited offline training cost/provenance
```

Do not silently amortize Phase 4 DQN training cost across Phase 5 test cases. The final accounting rule for offline generalized training versus online per-case solve cost is a separate experimental interpretation/design decision and must remain recoverable from separately logged quantities.

### DQN Training-Cost Accounting Gate — USER CLARIFICATION REQUIRED BEFORE FINAL PAPER COMPARISON

Phase 5 can proceed by logging DQN offline training time and online inference time separately.

Before the project later reports a single combined DQN computation-time number for Bellman-vs-RL regime classification, the user must define how generalized DQN training cost is treated, for example whether it is:

- reported separately as offline setup cost,
- amortized over a specified number of downstream BR queries,
- included in a one-shot end-to-end comparison,
- or presented using more than one accounting view.

Phase 5 must **not** choose that interpretation silently.

### Stochastic Evaluation and Seed Gate — USER CONFIRMATION REQUIRED

Bellman may be deterministic, while Tabular Q-learning is stochastic and DQN training was stochastic even though Phase 5 DQN inference should be deterministic.

Before the official Phase 5 evaluation, Codex must inspect the actual solver behavior and propose:

- number of Tabular retraining seeds per unseen case,
- whether repeated DQN inference runs are necessary for timing only,
- number of timing repetitions/warm-up runs if required for stable runtime measurement.

The user must approve the official seed/repetition plan before final Phase 5 aggregate statistics are generated.

Do not use a single stochastic Tabular result as an aggregate generalization conclusion unless the user explicitly approves that design.

### Test-Suite Size versus Phase 7

Phase 5 should use a **bounded, frozen unseen test suite sufficient to validate terrain generalization and expose failure modes**.

Do not turn Phase 5 into the full final Cartesian experiment over:

```text
all spatial discretizations
× all angular discretizations
× all Local-SSE radii
× all terrains
× all Monte Carlo seeds
```

That full computational-condition experiment belongs to Phase 7 after Local-SSE integration.

Phase 5 may use multiple unseen terrain instances and repeated stochastic solver seeds, but its purpose is to validate the generalized attacker-BR mechanism before integrating it into the Stackelberg search.

### Required Per-Run JSON Output

For each solver run, preserve the project result convention and record at least:

```text
phase
condition_id
spatial_discretization_m
angular_discretization_deg
terrain_test_id
terrain_hash_or_generator_config
scenario_or_defender_id
solver_method                 # bellman / tabular_q / dqn
solver_seed                   # null if deterministic/non-applicable
success
failure_reason                # if applicable
trajectory_result_path
J_A
bellman_reference_J_A
Delta_J_A_vs_bellman
normalized_J_A_metric         # only after metric definition is approved
attacker_br_runtime_sec
dqn_checkpoint_id             # DQN only
dqn_normalization_id          # DQN only
tabular_training_config_id    # Tabular only
status
```

Record any solver-internal-vs-common-objective mismatch as a diagnostic field.

### Required Aggregate JSON Output

For each condition-specific Phase 5 test batch, aggregate by solver and terrain/scenario as applicable:

- number of attempted runs,
- number of successful feasible goal-reaching trajectories,
- success rate,
- mean/median/std `J_A` where statistically applicable,
- mean/median/std raw Bellman objective difference,
- approved normalized objective-error statistics if defined,
- mean/median/std attacker-BR runtime,
- per-terrain breakdown,
- Tabular seed variability,
- DQN checkpoint and training-manifest provenance.

Do not average failed trajectories into objective statistics using invented penalty values.

### Required Visualization / Output

For representative unseen complex terrains, generate:

1. terrain geometry with attacker start and goal,
2. sensing-hazard visualization using the authoritative hazard representation,
3. Bellman trajectory,
4. Tabular Q-learning trajectory,
5. frozen-DQN trajectory,
6. combined trajectory overlay when visually readable,
7. per-method `J_A` comparison,
8. per-method attacker-BR runtime comparison.

Aggregate Phase 5 figures must include:

1. Bellman-relative raw `J_A` difference distribution for Tabular and DQN,
2. approved normalized approximation-error distribution if available,
3. success rate by method,
4. attacker-BR runtime distribution by method,
5. per-unseen-terrain objective/error summary,
6. Tabular stochastic variability where multiple seeds are used.

If a repository-defined terrain-complexity parameter exists and the approved test design varies it systematically, an additional performance-vs-complexity plot may be generated. Do not invent a new complexity metric solely for this figure.

Use `.png` for normal plots. Use `.html` only when interactive 3D terrain/trajectory visualization is genuinely useful.

### Visualization Integrity

Every trajectory-comparison figure must identify:

- terrain/test-case ID,
- computational condition,
- defender/scenario ID where applicable,
- solver method,
- success/failure state,
- authoritative `J_A`,
- attacker-BR runtime.

Do not plot an infeasible or failed trajectory as if it were a valid solution. Mark failure explicitly.

### Validation Tests

Phase 5 automated/diagnostic validation must verify at minimum:

1. no Phase 5 terrain appears in `TRAIN_SIMPLE` or `VALIDATION_SIMPLE`,
2. the official DQN checkpoint hash remains unchanged throughout evaluation,
3. DQN parameter gradients/optimizer steps are disabled during Phase 5,
4. replay-buffer contents are not modified by Phase 5 DQN evaluation,
5. normalization parameters remain frozen,
6. DQN uses the same discretization condition as its checkpoint metadata,
7. Bellman, Tabular, and DQN receive identical attacker/scenario/hazard inputs for each comparison case,
8. all solver trajectories are checked by the same feasibility/terminal code,
9. all successful solver trajectories are re-evaluated by the same authoritative `J_A` function,
10. Tabular starts each terrain-specific official retraining run from the approved initialization rather than carrying a learned Q-table from another test terrain,
11. DQN does not perform terrain-specific retraining,
12. runtime timers follow the defined method-specific boundaries,
13. CUDA synchronization is used when required for valid DQN GPU timing,
14. all per-run and aggregate JSON outputs are reproducible from saved manifests/seeds/configuration,
15. failed trajectories remain failures rather than being silently discarded from success-rate reporting.

### Failure Conditions

Phase 5 is invalid if any of the following occurs:

- a Phase 5 terrain was used during Phase 4 training, normalization, checkpoint selection, or hyperparameter tuning,
- DQN is fine-tuned on an unseen test terrain before its official result is recorded,
- terrain ID or test-case ID is added as a neural input,
- Bellman/Tabular/DQN use different dynamics, objectives, hazard fields, or discretization conditions for the same comparison case,
- Tabular is allowed to reuse terrain-specific knowledge while its role is defined as retrain-per-case,
- DQN is evaluated under a discretization condition different from its condition-specific checkpoint,
- `J_A` is compared using solver-specific internal values instead of the common evaluator,
- a normalized accuracy formula is invented without repository support or user approval,
- DQN offline training cost is silently hidden inside or silently added to per-case inference time,
- failed runs are removed from success statistics,
- Phase 5 results are used to retune the official Phase 4 model and then re-presented as untouched test results.

### Phase 5 User Decision Gates

Before official Phase 5 evaluation is declared complete, resolve the following user decisions:

1. **Unseen complex-terrain test-set composition**
   - Codex must inspect available terrain generators/configurations and propose the test families, parameter ranges, instance count, and seeds.

2. **Defender/sensing scenario design**, if Phase 4 trained across multiple defender configurations
   - Codex must propose how those configurations are represented in Phase 5 without creating an unintended additional distribution shift.

3. **Normalized Bellman-relative `J_A` metric**, if the repository does not already contain an approved definition
   - raw `J_A` and raw difference can always be recorded; `% error` / `accuracy` terminology requires an approved formula.

4. **Stochastic seed/repetition plan**
   - number of Tabular retraining seeds per unseen case and timing repetitions.

5. **Generalization acceptance criterion**, if Phase 5 is intended to act as a scientific gate before Phase 6
   - define acceptable success rate and/or Bellman-relative `J_A` approximation level; do not infer the threshold from prior examples such as “80% accuracy.”

6. **Final DQN training-cost accounting rule**
   - may remain unresolved during Phase 5 data collection because offline training and online inference are logged separately, but must be resolved before final paper-level Bellman-vs-RL computation-time classification.

### Exit Criteria

Phase 5 is technically complete when all of the following hold for the approved condition-specific test batch:

1. an untouched, user-approved complex unseen-terrain manifest is frozen,
2. leakage checks against Phase 4 train/validation data pass,
3. the official frozen Phase 4 DQN checkpoint is evaluated without updates,
4. Bellman attacker BR is successfully evaluated wherever the discrete problem is solvable,
5. Tabular Q-learning is retrained/evaluated per approved test case and seed plan,
6. DQN inference is evaluated on the same cases without terrain-specific retraining,
7. all returned trajectories pass through the same feasibility/terminal checks,
8. all successful trajectories are re-evaluated with the same authoritative `J_A` evaluator,
9. attacker-BR runtimes are logged using the approved boundaries,
10. per-run JSON and aggregate JSON outputs are complete,
11. required representative and aggregate visualizations are generated,
12. success/failure behavior and Bellman-relative objective differences are explicitly reported,
13. no Local-SSE or `J_D` conclusions are claimed in Phase 5,
14. no Phase 5 test information has been used to adapt the frozen model.

If the user has defined a separate **scientific generalization acceptance criterion**, report whether that criterion is met. A failure to meet the scientific threshold does not invalidate the Phase 5 experiment; it means the generalized DQN is not yet acceptable for the intended approximation role and must return to an earlier design/training phase before being treated as validated.

### Expected Deliverables

Phase 5 should leave the project with:

- frozen Phase 5 unseen-test manifest,
- leakage-audit JSON,
- per-case Bellman results,
- per-case/per-seed Tabular results,
- per-case frozen-DQN results,
- common `J_A` and feasibility evaluations,
- attacker-BR runtime results,
- aggregate generalization summary JSON,
- representative terrain/hazard/trajectory figures,
- aggregate error/success/runtime figures,
- documented user-approved Phase 5 design decisions,
- explicit statement of whether generalized attacker-BR performance is technically/scientifically acceptable for Phase 6 integration.

### Detailed Codex Command

Implement **Phase 5 only** according to all requirements above. Start by inspecting the latest Phase 0–4 outputs and current repository code. Do not begin official unseen-terrain evaluation until all required user decision gates that affect test-set integrity have been presented and resolved. Do not implement Phase 6 Local-SSE integration or Phase 7 computational sweeps.

---

## Phase 6 — DQN Attacker BR Integration into Local SSE

### Objective
Integrate the Phase 1 common attacker-BR interface into the existing Local Strong Stackelberg Equilibrium (Local SSE) solver so that the **attacker BR backend is the only solver component that changes** among:

1. exact Bellman DP,
2. terrain/case-specific Tabular Q-learning,
3. frozen condition-specific generalized DQN.

The purpose of Phase 6 is to determine how attacker-BR approximation propagates through the existing defender local search and changes the final defender strategy, attacker trajectory, `J_A`, and `J_D`.

At the selected discretization condition, use the terminology:

- **Bellman-based Local SSE** = exact Local SSE with respect to the existing discretized Bellman follower problem and existing local defender-search definition,
- **Tabular-based approximate Local SSE** = Local-SSE search driven by Tabular approximate attacker BRs,
- **DQN-based approximate Local SSE** = Local-SSE search driven by frozen DQN approximate attacker BRs.

Do not label the Tabular- or DQN-based result as an exact SSE merely because the defender local-search stopping rule is satisfied under the approximate BR backend.

### Phase Boundary

Phase 6 may implement:

- a solver-backend selector inside the existing Local-SSE pipeline,
- Bellman / Tabular / DQN attacker-BR adapters through the Phase 1 common interface,
- propagation of the current defender/sensor configuration into the attacker BR environment,
- per-BR-call and cumulative attacker-BR runtime logging,
- total Local-SSE runtime logging,
- Local-SSE search-trace logging,
- final `J_A` and `J_D` comparison,
- final defender-strategy and attacker-trajectory comparison,
- Local-SSE regression/parity tests,
- representative Local-SSE visualizations.

Phase 6 must **not** implement:

- new DQN architecture design,
- DQN retraining or fine-tuning,
- new terrain observation channels,
- cross-discretization DQN (A2),
- full Monte Carlo sweeps over all spatial/angular discretizations and all `r`,
- final Bellman-vs-RL regime classification,
- changes to the existing defender local-search algorithm,
- changes to Strong Stackelberg tie-breaking,
- changes to `J_A`, `J_D`, dynamics, sensing, feasibility, or terminal definitions,
- fallback from a failed RL BR to Bellman unless the user explicitly authorizes such a diagnostic mode.

The full computational-condition sweep belongs to Phase 7. Phase 6 is an **integration and correctness phase** using a limited, user-approved set of configurations.

### Repository-Grounded Preconditions

Before implementing Phase 6, Codex must inspect the latest current repository and completed Phase 0–5 outputs. Do not infer missing numerical values from this planning document.

Resolve from the actual code/configuration:

1. the current Local-SSE entry point(s),
2. the exact defender strategy representation,
3. defender candidate generation,
4. the existing Chebyshev-neighborhood implementation and meaning of `r`,
5. the Local-SSE initial defender strategy / initialization rule,
6. the exact Local-SSE stopping condition,
7. the existing Strong Stackelberg tie-breaking rule,
8. the exact `J_D` evaluator and its sign/minimization/maximization convention,
9. how the current defender configuration `d` is supplied to the sensing/hazard model,
10. whether the Local-SSE implementation currently caches attacker BRs or objective values,
11. whether any attacker-BR warm start or result reuse currently occurs across nearby defender candidates,
12. the existing result-directory convention,
13. the Phase 1 common attacker-BR interface,
14. the Phase 4 official DQN checkpoint/configuration,
15. the Phase 5 condition(s) for which DQN generalization was actually validated,
16. the current Tabular training configuration established in Phase 3/4/5,
17. the authoritative `J_A` evaluator shared by Bellman and RL,
18. the existing Local-SSE visualization/output utilities, if any.

If any of these differ from assumptions in this document, use the actual current implementation as the source of truth and record the discrepancy. Do not silently rewrite the legacy Local-SSE behavior to match this plan.

### Fixed Scientific Comparison Principle

For a given Phase 6 comparison run, the following must be identical across Bellman, Tabular, and DQN modes:

- terrain/scenario,
- attacker initial condition,
- goal/terminal specification,
- sensor model,
- defender initial strategy,
- defender feasible strategy set,
- Local-SSE neighbor definition,
- neighbor radius `r`,
- candidate ordering where the existing algorithm depends on ordering,
- local-search stopping condition,
- Strong Stackelberg tie-breaking,
- spatial discretization `Δs = dx = dy = dh`,
- angular discretization `Δθ = dpsi = dgamma`,
- `J_A` definition,
- `J_D` definition.

The only intended algorithmic difference is:

```text
attacker BR backend = bellman | tabular_q | dqn
```

Because an approximate BR changes the defender objective landscape, the three local-search trajectories are **allowed to diverge after the common initialization**. Do not force them to visit the same defender candidates merely to make the paths visually similar.

### Condition-Specific DQN Constraint

Phase 6 continues to use the A1 design selected in Phase 2.

For one Phase 6 run:

```text
one (Δs, Δθ) condition
    -> matching Phase 4/5 DQN checkpoint
    -> one Local-SSE comparison
```

Do not apply a DQN checkpoint trained for one discretization condition to another condition.

Before loading the checkpoint, verify that its metadata matches at minimum:

- spatial discretization,
- angular discretization,
- observation schema/version,
- action-catalog/version,
- normalization configuration,
- model architecture/configuration.

Abort the DQN run on an incompatible signature rather than silently adapting the checkpoint.

### Local-SSE Backend Architecture

Refactor or wrap the existing Local-SSE solver so that it requests attacker BRs through one stable call boundary, conceptually:

```text
br_result = solve_attacker_br(
    problem_instance=current_defender_condition,
    solver_method={bellman | tabular_q | dqn},
    solver_config=...
)
```

The Local-SSE solver must consume a backend-neutral result containing the fields established in Phase 1, including at minimum:

- solve status,
- attacker trajectory,
- terminal/success status,
- authoritative or re-evaluable `J_A`,
- method/configuration metadata.

The Local-SSE code must not contain method-specific objective definitions such as a separate DQN `J_A` or Tabular `J_A`.

### Defender-Condition Propagation

At every defender candidate `d` evaluated by Local SSE:

1. build/update the attacker BR problem using that exact `d`,
2. regenerate/query the sensing/hazard representation exactly as required by the existing model,
3. keep terrain, attacker start, goal, dynamics, and discretization unchanged unless the existing Local-SSE formulation explicitly changes them,
4. call the selected attacker BR backend,
5. validate the returned attacker trajectory,
6. recompute `J_A` using the authoritative evaluator,
7. compute `J_D` using the existing defender evaluator and existing SSE tie-breaking semantics.

For DQN, the current `d` must affect the DQN through the approved Phase 2 sensing/hazard observation mechanism. Do not provide a defender ID or terrain ID as an undocumented neural input.

### Bellman Backend

For each defender candidate requested by the Local-SSE algorithm:

- solve the attacker BR with the existing exact Bellman DP at the selected discretization,
- preserve the existing recurrence, feasibility logic, terminal logic, and objective,
- return the result through the Phase 1 common interface,
- recompute/verify final `J_A` with the authoritative evaluator if this is already part of the common validation path.

Bellman provides the Phase 6 reference Local SSE at the selected discretization and `r`.

### Tabular Q-Learning Backend

For every new defender candidate `d` that requires a BR evaluation, Tabular Q-learning must be treated as a **case-specific approximate solver**.

Default Phase 6 semantics:

- initialize/retrain Tabular Q-learning for the current attacker-BR case using the approved Phase 3/5 configuration,
- do not carry a learned Q-table from one defender candidate to another unless the user explicitly approves a separate warm-start experiment,
- include training plus final deterministic policy/trajectory extraction in attacker-BR runtime,
- validate the returned trajectory using the common feasibility/terminal checks,
- recompute final `J_A` using the same authoritative evaluator used for Bellman and DQN.

If current legacy code already performs cross-candidate Q-table warm starting or reuse, Codex must report it before changing the behavior. Do not silently preserve or remove such reuse because it materially changes both approximation quality and timing.

### DQN Backend

For every defender candidate `d` requested by Local SSE:

- load the matching official Phase 4/5 condition-specific checkpoint once per Local-SSE run unless repository architecture requires otherwise,
- keep all network weights frozen,
- keep normalization statistics frozen,
- perform no optimizer step,
- perform no replay-buffer update,
- perform no test-time fine-tuning,
- generate the current observation using the Phase 2 observation builder and the current defender-dependent hazard/sensing state,
- use the Phase 3 feasible-action mask at every decision,
- use deterministic evaluation policy,
- validate the returned trajectory,
- recompute `J_A` using the authoritative evaluator.

The DQN attacker-BR runtime for Local SSE is online inference time only. Phase 4 offline training time must remain stored separately as provenance and must not be silently folded into one Phase 6 run.

### BR Failure Handling

Do not hide solver failures.

If Bellman, Tabular, or DQN fails to return a valid terminal attacker trajectory for a defender candidate:

- record the backend, defender candidate, state/configuration signature, failure type, and runtime,
- do not fabricate `J_A` or `J_D`,
- do not insert an arbitrary large penalty unless that exact failure/penalty rule already exists in the authoritative formulation,
- do not automatically call Bellman as an RL fallback,
- mark the corresponding Local-SSE run as partial/failed if the existing search cannot validly continue.

A retry rule for stochastic Tabular Q-learning may use only the user-approved seed/repetition policy. DQN evaluation is deterministic for a fixed checkpoint, scenario, and configuration unless the approved policy explicitly contains stochasticity.

### Strong Stackelberg Tie-Breaking Preservation

Codex must locate and preserve the current Strong Stackelberg tie-breaking implementation.

Where multiple attacker BRs have equivalent attacker objective according to the existing tolerance/rule, the same defender-favorable tie-breaking semantics used by the current exact formulation must be applied wherever the backend exposes the relevant candidate set/information.

If the existing Tabular/DQN interface returns only one approximate trajectory and therefore cannot reproduce exact multi-BR tie handling, do **not** invent an equivalent mechanism. Record this as an approximation limitation in Phase 6 metadata and use the current approved approximate-solver behavior.

### Local-SSE Termination Validation

For each method, record why the Local-SSE search terminated according to the existing algorithm.

At minimum record:

- final defender strategy `d*`,
- local iteration count,
- evaluated defender candidates,
- accepted defender moves / search path,
- final neighborhood radius `r`,
- stopping reason,
- final attacker trajectory,
- final `J_A`,
- final `J_D`.

For Bellman, verify the existing local-optimality/SSE stopping condition exactly as implemented.

For Tabular/DQN, verify the **same algorithmic stopping condition using that approximate BR backend**, but label the output as approximate-BR-based Local SSE rather than exact SSE.

### Runtime Instrumentation

Phase 6 official runtime metrics remain limited to the two runtime categories already approved globally:

1. attacker-BR runtime,
2. total Local-SSE runtime.

Because one Local-SSE solve invokes multiple attacker BRs, record:

```text
per_br_call_runtime_sec[]
cumulative_attacker_br_runtime_sec
local_sse_total_runtime_sec
num_attacker_br_calls
```

Interpretation by backend:

**Bellman**
```text
per BR call = Bellman BR solve time
```

**Tabular**
```text
per BR call = Tabular training + final deterministic trajectory query
```

**DQN**
```text
per BR call = frozen online DQN trajectory inference for that defender condition
```

Do not yet add CPU utilization, RAM utilization, GPU utilization, or detailed subsection timing to the official comparison metrics.

If DQN model-loading time is non-negligible, record it as diagnostic metadata. Load the model outside repeated BR calls when possible and do not intentionally reload it for every defender candidate merely to inflate runtime.

### Caching and Reuse Audit

Before official Phase 6 timing, inspect the current Local-SSE implementation for:

- BR caching keyed by defender strategy,
- hazard-map caching,
- Bellman cost-to-go reuse,
- Tabular Q-table reuse,
- previous-iteration warm starts,
- repeated evaluation of identical defender candidates.

Record all detected behavior.

Do not create solver-specific caching that changes the scientific comparison without explicit justification. If existing legacy caching materially advantages one backend or cannot be applied consistently, report the issue before official comparative timing and request user direction rather than guessing a new policy.

### Objective Evaluation

All successful final trajectories must be evaluated by the unchanged authoritative objective implementations.

Record:

```text
J_A_final
J_D_final
```

for each backend.

Bellman provides the reference values:

```text
J_A_Bellman
J_D_Bellman
```

Always store raw differences for the approximate methods:

```text
ΔJ_A_method = J_A_method - J_A_Bellman
ΔJ_D_method = J_D_method - J_D_Bellman
```

If Phase 5 established an approved normalized error/accuracy formula, reuse that exact formula. If not, do not invent a new “accuracy percentage” in Phase 6.

### Final Strategy Comparison

For each backend record the final solution tuple conceptually as:

```text
(d*_method, a*_method, J_A_method, J_D_method)
```

where `a*_method` is the returned final attacker trajectory/BR under the method's final defender strategy.

Compare at minimum:

- whether the final defender strategies are identical,
- if not identical, their difference using the existing defender-grid representation,
- Bellman-relative `J_A` difference,
- Bellman-relative `J_D` difference,
- final attacker trajectory differences,
- total Local-SSE runtime,
- cumulative attacker-BR runtime,
- number of BR calls.

Do not invent a defender-strategy distance formula if the repository does not already define one. Raw grid/index/coordinate differences are sufficient until a metric is approved.

### Optional Post-Hoc Bellman Audit of Approximate Final Defender Strategies

A useful diagnostic, but **not mandatory for Phase 6 completion**, is:

1. take the final defender strategy selected by Tabular or DQN,
2. hold that defender strategy fixed,
3. compute one exact Bellman attacker BR at that `d`,
4. evaluate the resulting exact `J_A` / `J_D`.

This separates:

- error caused by choosing a different defender strategy,
- error caused by an approximate follower response at that strategy.

Because this adds Bellman computation, Codex must present the expected purpose/cost and obtain user approval before enabling it as an official Phase 6 diagnostic if it is not already available cheaply from cached evaluations.

### Required JSON Outputs

Follow the current project result-directory convention discovered from the repository. Use JSON for machine-readable outputs.

At minimum generate one configuration manifest containing:

```text
phase
terrain/scenario identifier
spatial discretization
angular discretization
neighbor radius r
initial defender strategy
Local-SSE algorithm/version
Strong Stackelberg tie-breaking configuration
attacker-BR backend
backend configuration/checkpoint
seed, where applicable
```

For each Local-SSE run record at minimum:

```text
status
solver_method
num_local_iterations
num_defender_candidates_evaluated
num_attacker_br_calls
cumulative_attacker_br_runtime_sec
local_sse_total_runtime_sec
final_defender_strategy
final_attacker_trajectory artifact/reference
J_A_final
J_D_final
stopping_reason
failure information, if any
```

Also save the Local-SSE search trace in JSON with the existing defender representation and evaluated objectives sufficient to reproduce the Phase 6 diagnostic plots.

### Required Visualization

Phase 6 visualizations are integration/diagnostic outputs, not yet the full Phase 7 paper-level Monte Carlo figures.

Generate at minimum:

1. **Defender local-search trace / evaluated landscape**
   - show evaluated defender candidates available from the actual search,
   - indicate initial defender strategy,
   - accepted search path,
   - final defender strategy,
   - objective quantity used by the existing defender search.
   - Do not exhaustively compute an unevaluated global defender landscape solely for visualization.

2. **Bellman-based Local SSE final result**
   - final defender/sensor placement,
   - final attacker trajectory,
   - terrain/sensing context.

3. **Tabular-based approximate Local SSE final result**
   - same visualization semantics.

4. **DQN-based approximate Local SSE final result**
   - same visualization semantics.

5. **Final strategy comparison**
   - compare final defender placements from the three backends on the same scenario.

6. **Final attacker trajectory comparison**
   - overlay the three final trajectories when geometrically meaningful.

7. **Objective comparison**
   - final `J_A` and `J_D` for Bellman, Tabular, and DQN.

8. **Runtime comparison**
   - cumulative attacker-BR runtime,
   - total Local-SSE runtime,
   - number of BR calls.

Use PNG for normal plots. Use HTML only where interactive 3D visualization is genuinely required.

### Regression and Automated Tests

Add or extend tests that verify at minimum:

1. **Backend selector test**
   - the Local-SSE solver can request Bellman, Tabular, or DQN through the common interface without changing defender-search code.

2. **Common scenario test**
   - all backends receive the same terrain, start, goal, defender initialization, discretization, and `r`.

3. **Defender propagation test**
   - changing candidate `d` changes the sensing/hazard input seen by each attacker-BR backend as intended.

4. **Bellman regression test**
   - Bellman-based Local SSE through the new interface reproduces the pre-Phase-6 Bellman Local-SSE result within the existing numerical/tie tolerances.

5. **Objective identity test**
   - final trajectories from all methods are evaluated by the same `J_A` and `J_D` implementations.

6. **No-DQN-training test**
   - DQN weights, optimizer state, normalization statistics, and replay state do not change during Local-SSE evaluation.

7. **Tabular independence test**
   - unless explicitly approved otherwise, Q-table state is not leaked/reused across distinct defender BR cases.

8. **Discretization-signature test**
   - incompatible DQN checkpoints are rejected.

9. **Failure transparency test**
   - failed/invalid attacker BRs do not receive fabricated objective values or silent Bellman fallback.

10. **Runtime-accounting test**
    - cumulative attacker-BR time is derived from the actual BR calls and total Local-SSE time encloses the complete local-search execution.

11. **Result-schema test**
    - required Phase 6 JSON fields are present and parseable.

### User Decision Gates Before Official Phase 6 Execution

Codex must inspect the latest repository and Phase 5 outputs first, then ask the user only for decisions that cannot be resolved from the code.

The following are expected user gates:

1. **Phase 6 integration-test discretization condition**
   - select one `(Δs, Δθ)` condition for which a compatible DQN checkpoint exists and Phase 5 validation has been completed.
   - If only one valid condition exists, present it and request confirmation rather than inventing another.

2. **Phase 6 integration scenario(s)**
   - choose the terrain/scenario used for Local-SSE integration testing.
   - Prefer a limited case or small set already validated in Phase 5; Phase 6 is not the final Monte Carlo experiment.

3. **Local-SSE neighbor radius `r` for the integration test**
   - inspect whether the current Local-SSE implementation/config has an established development/default radius.
   - Present the actual existing value if one exists; otherwise ask the user to choose one.
   - Do not sweep `r = 1..5` in Phase 6 unless explicitly requested; that belongs to Phase 7.

4. **Initial defender strategy if not fixed by the current formulation/config**
   - use the current authoritative initialization if one exists.
   - If several valid initialization modes exist, present them and ask which Phase 6 test should use.

5. **Tabular stochastic repetition/seed rule for end-to-end Local-SSE comparison**
   - reuse an approved Phase 5 rule if one exists.
   - If not, ask how many end-to-end Tabular Local-SSE repetitions/seeds are required for Phase 6 validation.

6. **Caching/warm-start policy only if the repository reveals a material ambiguity**
   - if existing Local-SSE/Bellman/Tabular code uses method-specific cache or warm-start behavior that affects fairness or timing, report the exact behavior and request a decision before official timing.

7. **Optional final-defender Bellman audit**
   - ask only if enabling the optional post-hoc diagnostic would add substantial computation and the user wants it included.

Do not ask the user for values already unambiguously specified in the current code or completed earlier phases.

### Exit Criteria

Phase 6 is technically complete when all of the following hold for the approved integration configuration(s):

1. the existing Local-SSE solver uses the Phase 1 common attacker-BR boundary,
2. Bellman, Tabular, and DQN can be selected without duplicating the defender local-search implementation,
3. Bellman-based Local SSE reproduces the legacy/reference Local-SSE behavior,
4. Tabular is evaluated as a case-specific retrained attacker-BR approximation according to the approved seed policy,
5. DQN uses the matching frozen condition-specific checkpoint with no updates,
6. every defender candidate passes its exact current `d`/hazard condition to the selected attacker BR,
7. all successful attacker trajectories pass common feasibility and terminal validation,
8. all successful final results use the same authoritative `J_A` and `J_D` evaluators,
9. Strong Stackelberg tie-breaking behavior is preserved or any unavoidable approximate-solver limitation is explicitly documented,
10. per-BR and cumulative attacker-BR runtimes are recorded,
11. total Local-SSE runtime is recorded,
12. final defender strategy, attacker trajectory, `J_A`, `J_D`, and stopping reason are stored in JSON,
13. required Local-SSE search/result/comparison visualizations are generated,
14. RL BR failures are transparent and no silent exact-solver fallback occurs,
15. no Phase 7 parameter sweep or Phase 8 regime classification is performed.

A Phase 6 integration run may be technically successful even if Tabular/DQN produces a materially different approximate Local SSE from Bellman; that difference is an experimental outcome, not an implementation failure, provided all interface, feasibility, objective, and logging checks pass.

### Expected Deliverables

Phase 6 should leave the project with:

- one common Local-SSE attacker-BR backend interface,
- Bellman Local-SSE adapter/integration,
- Tabular Local-SSE adapter/integration,
- frozen-DQN Local-SSE adapter/integration,
- Phase 6 configuration manifest(s),
- per-method Local-SSE result JSON,
- Local-SSE search-trace JSON,
- cumulative BR and total Local-SSE timing outputs,
- final `J_A` / `J_D` comparison,
- final defender/attacker comparison visualizations,
- documented user-approved Phase 6 design decisions,
- a stable Local-SSE evaluation pipeline ready for the Phase 7 computational-condition Monte Carlo sweep.

### Detailed Codex Command

Implement **Phase 6 only** according to all requirements above. Begin by inspecting the latest Phase 0–5 outputs and current `3D_0827/` / `3D_RL_DQN/` code. Preserve the existing Local-SSE defender search, objective functions, tie-breaking, dynamics, sensing model, and discretization semantics. Insert only the common attacker-BR backend boundary needed to select Bellman, Tabular Q-learning, or frozen DQN. Resolve all required user decision gates before official comparative execution. Do not implement the Phase 7 parameter sweep or Phase 8 classification.

---

## Phase 7 — Computational-Condition Monte Carlo Sweep

### Objective
Run the final empirical scaling study across the approved computational-condition domain and quantify, under **fresh unseen complex terrain/scenario realizations**, how Bellman DP, case-specific Tabular Q-learning, and condition-specific generalized DQN differ in:

- attacker-BR computation time,
- total Local-SSE computation time,
- final Local-SSE attacker objective `J_A`,
- final Local-SSE defender objective `J_D`,
- Bellman-relative approximation error/accuracy using the user-approved metric definition,
- success/failure rate.

Phase 7 is the main **paper-level Monte Carlo data-generation phase**. It does not yet convert the results into the final Bellman-vs-RL decision rule; that interpretation/classification belongs to Phase 8.

### Phase Boundary

Phase 7 may implement:

- the official computational-condition matrix,
- fresh Phase-7 Monte Carlo unseen-terrain/scenario generation/loading,
- condition-specific Bellman / Tabular / frozen-DQN Local-SSE runs,
- repeated stochastic trials according to an approved seed protocol,
- exact timing instrumentation using Phase 0/5/6 timing boundaries,
- resumable/batched experiment execution,
- per-run JSON logging,
- aggregate Monte Carlo statistics,
- publication-oriented scaling/error plots,
- explicit timeout/censoring records for computationally unresolved conditions.

Phase 7 must **not**:

- change the Bellman recurrence,
- retune Tabular Q-learning on Phase-7 test performance,
- retune DQN architecture/hyperparameters on Phase-7 test performance,
- fine-tune a DQN on Phase-7 unseen terrains,
- introduce A2 cross-discretization DQN,
- change `J_A`, `J_D`, Local-SSE logic, tie-breaking, dynamics, sensing, or feasibility semantics,
- derive or announce the final Bellman-vs-RL regime-selection boundary; that is Phase 8.

### Repository-Grounding / Preflight Audit

Before launching the official sweep, inspect the current `3D_0827/`, `3D_RL_DQN/`, and Phase 0–6 artifacts and produce a Phase-7 preflight manifest that resolves or reports:

1. current Bellman, Tabular, and DQN BR entry points,
2. current Local-SSE entry point,
3. authoritative `J_A` and `J_D` evaluators,
4. current Local-SSE Chebyshev-neighbor implementation and meaning of `r`,
5. approved Strong Stackelberg tie-breaking behavior,
6. approved Phase-5/6 Bellman-relative objective-error metric definitions, if already resolved,
7. approved Tabular seed/training protocol,
8. approved DQN checkpoint/training protocol,
9. current timing boundaries,
10. existing result-directory convention,
11. available terrain/scenario generators and all train/validation/test manifests from earlier phases,
12. hardware/device configuration relevant to reproducibility.

Do not reconstruct any unresolved implementation detail from this planning document. Use the latest code and actual manifests.

### Official Computational-Condition Domain

The intended computational variables are:

#### Spatial discretization

```text
dx = dy = dh ∈ {100, 50, 25, 12.5, 10, 5} m
```

#### Angular discretization

```text
dpsi = dgamma ∈ {1.25, 2.5, 5, 10} deg
```

#### Local-SSE defender neighbor-search radius

```text
r ∈ {1, 2, 3, 4, 5}
```

Define one computational condition as:

```text
c = (spatial_discretization_m,
     angular_discretization_deg,
     neighbor_radius_r)
```

The nominal full factorial contains:

```text
6 × 4 × 5 = 120 computational conditions
```

Do **not** assume that all 120 are computationally feasible simply because they are listed. The official Phase-7 matrix must be frozen only after the user approves the execution design described in the user-confirmation gates below.

### A1 Condition-Specific DQN Rule — Remains in Force

Phase 7 continues to use **A1 condition-specific DQN** only.

For every evaluated pair:

```text
(dx=dy=dh, dpsi=dgamma)
```

DQN evaluation must use a checkpoint trained for that exact discretization pair.

Do not feed the discretization values into one universal DQN and do not implement cross-discretization A2 in Phase 7.

The Local-SSE neighbor radius `r` does **not** define a different attacker policy discretization, so the same approved DQN checkpoint for a fixed `(spatial, angular)` pair may be reused across `r = 1..5`, provided the attacker problem semantics are unchanged.

#### Condition-Coverage Preparation

Phase 4 may initially have produced a generalized DQN for only one development discretization condition. Before Phase 7 can compare DQN over the approved condition matrix, every `(spatial, angular)` pair included in the official Phase-7 matrix must have an approved condition-specific generalized checkpoint.

For any missing pair, reuse the already approved Phase 2–5 pipeline **without redesigning it**:

1. instantiate the same observation contract at the requested discretization,
2. instantiate the corresponding Phase-1 action/transition semantics,
3. train on the same approved simple-terrain training distribution or its discretization-consistent equivalent,
4. validate using the same approved held-out simple-terrain protocol,
5. freeze one official checkpoint according to the previously approved checkpoint-selection rule,
6. record training runtime and all provenance.

Do not treat this as a new architecture-search phase. Any condition-specific architecture or hyperparameter modification that is not mechanically required by changed input/output dimensions must be explicitly approved and documented before official Phase-7 evaluation.

### Fresh Phase-7 Monte Carlo Evaluation Set

Phase 7 should not silently reuse training or validation terrain instances.

Create a dedicated evaluation set, conceptually:

```text
PHASE7_MC_TEST
```

that is disjoint from:

```text
TRAIN_SIMPLE
VALIDATION_SIMPLE
```

and uses unseen complex terrain/scenario realizations.

Prefer fresh terrain instances/seeds relative to Phase 5 as well, especially if Phase-5 outcomes influenced debugging, model selection, or design decisions. If Phase 5 remained a strictly untouched evaluation-only set and the user intentionally wants to reuse its terrain distribution, Phase 7 may reuse the **approved distribution definition** while generating new independent instances/seeds.

Before official execution, save an immutable Monte Carlo manifest containing, as applicable:

- terrain family/type,
- terrain-generator parameters,
- terrain-generation seed,
- defender/sensing configuration,
- attacker start/goal configuration,
- any other scenario randomization explicitly approved by the user.

No Phase-7 test terrain may be used for DQN normalization fitting, weight updates, hyperparameter tuning, checkpoint selection, or Tabular hyperparameter tuning.

### Monte Carlo Experimental Unit

Define one official Monte Carlo unit as:

```text
(condition c,
 scenario/terrain realization k,
 method m,
 stochastic-seed replicate s if applicable)
```

where:

```text
m ∈ {bellman, tabular_q, dqn}
```

For Bellman, `s` may be absent if the implementation is deterministic.

For Tabular Q-learning, each approved stochastic replicate must retrain according to the frozen case-specific Tabular protocol.

For DQN, deterministic inference uses the frozen checkpoint. If the official study includes multiple independently trained DQN checkpoints, their training-seed identity must be treated explicitly as a higher-level replicate rather than silently pooled with terrain Monte Carlo seeds.

### Controlled Scenario Variables

The Phase-7 computational-condition study is intended to isolate the effect of discretization and Local-SSE neighborhood size while averaging over approved unseen terrain/scenario variation.

Therefore:

- keep physics/model/objective definitions fixed,
- use the same scenario realization across Bellman, Tabular, and DQN for a given Monte Carlo index,
- use the same initial defender strategy when the Local-SSE algorithm requires one,
- use the same Strong Stackelberg tie-breaking,
- use the same sensing/hazard configuration for all three methods within a paired trial,
- do not introduce an additional out-of-distribution model change only for one method.

This is a **paired experimental design** wherever possible: every method should solve the same `(condition, scenario)` instance so Bellman-relative objective differences and runtime comparisons are meaningful.

### Method Contract for Each Monte Carlo Trial

#### Bellman

For each `(condition, scenario)`:

1. run the existing Local-SSE solver with Bellman as the attacker-BR backend,
2. treat the Bellman-based Local SSE at that discretization as the exact discrete reference for that condition,
3. record all Phase-6 Local-SSE outputs and timing.

#### Tabular Q-learning

For the same `(condition, scenario)`:

1. use the same Local-SSE solver,
2. replace only the attacker-BR backend with Tabular Q-learning,
3. retrain per BR case according to the frozen Tabular protocol unless the approved existing baseline defines otherwise,
4. run all approved stochastic replicates,
5. recompute final `J_A` / `J_D` using the common authoritative evaluators.

#### DQN

For the same `(condition, scenario)`:

1. load the official generalized checkpoint corresponding to that `(spatial, angular)` pair,
2. freeze weights and normalization,
3. use deterministic inference inside the same Local-SSE search,
4. do not fine-tune on the Phase-7 scenario,
5. recompute final `J_A` / `J_D` with the common evaluators.

### Official Measurements

The official Phase-7 comparison remains limited to the user-approved metrics:

#### Timing

```text
cumulative_attacker_br_runtime_sec
local_sse_total_runtime_sec
```

Also retain per-BR-call timing as diagnostic data because it is needed to understand how the total Local-SSE time is accumulated, but do not add CPU/RAM/GPU utilization as official comparison metrics.

#### Final Local-SSE objectives

```text
J_A_final
J_D_final
```

#### Bellman-relative objective differences

Always save raw paired differences:

```text
Delta_J_A = J_A_method - J_A_bellman
Delta_J_D = J_D_method - J_D_bellman
```

If Phase 5/6 has already established a user-approved normalized error/accuracy definition, reuse it exactly.

If no approved normalized definition exists by Phase 7 execution time, stop before producing official `accuracy`, `% error`, or `optimality gap` plots and request user clarification. Do not invent a denominator or threshold.

#### Success/failure

Record:

```text
status
trajectory_success
local_sse_success
failure_type
stopping_reason
```

Do not silently discard failures from aggregate statistics.

### Runtime Measurement Protocol

Use the already established timing boundaries from Phases 0, 5, and 6.

For DQN GPU inference, synchronize the device around timed regions where needed to prevent asynchronous execution from biasing runtime.

For repeated timing trials:

- use the user-approved warm-up/repetition protocol,
- distinguish algorithm runtime from one-time experiment-orchestration overhead,
- keep any DQN model-load time separately recoverable,
- do not silently add or remove offline DQN training time from the online Local-SSE runtime.

### DQN Offline Training-Cost Accounting

For every condition-specific DQN checkpoint, preserve:

```text
dqn_offline_training_runtime_sec
```

as provenance.

Phase 7 must **not** silently fold that offline training time into each online BR or Local-SSE solve.

Store online inference/Local-SSE runtime and offline training cost separately so Phase 8 can apply the user-approved accounting interpretation, such as:

- offline setup cost reported separately,
- one-shot end-to-end cost,
- amortized cost over a specified number of future BR/Local-SSE queries.

If the accounting interpretation has already been explicitly approved in an earlier phase, preserve it as an additional derived view but never delete the raw separated quantities.

### Computationally Intractable / Timeout Conditions

The experiment itself is intended to identify conditions where exact Bellman computation becomes costly. Therefore, a very slow Bellman run must not automatically be treated as a coding failure.

However, the sweep needs a reproducible operational rule for runs that cannot reasonably finish.

Before official execution, the user must approve an experiment budget policy, which may include:

- maximum wall-clock time per Local-SSE run,
- maximum wall-clock time per attacker-BR call if needed,
- maximum retry count,
- operational memory-failure handling even though memory is not an official performance metric.

When a run exceeds the approved bound or terminates for resource reasons:

- record it as `TIMEOUT`, `RESOURCE_FAILURE`, or another explicit status,
- preserve the condition/scenario/method metadata,
- record elapsed time up to termination,
- do not fabricate `J_A` / `J_D`,
- do not exclude the result from feasibility/scalability reporting.

Phase 8 may later use the existence of censored/incomplete Bellman regions as evidence of computational limits, but it must not pretend an exact runtime/objective was observed where none was obtained.

### Resumable Experiment Runner

Because the full Phase-7 matrix can be large and expensive, implement the official runner as resumable and idempotent.

Requirements:

1. each run has a deterministic unique run ID based on condition/scenario/method/seed/provenance,
2. completed valid runs are not recomputed unless explicitly requested,
3. failed/timeout runs remain visible and are not overwritten silently,
4. interrupted batches can resume from the remaining run manifest,
5. raw per-run JSON is written before aggregate analysis,
6. aggregate statistics are regenerated from raw records rather than from hand-edited tables.

Do not allow the orchestration layer to alter solver behavior.

### Required Per-Run JSON

Follow the existing project result-folder convention. Each official run must preserve at least:

```text
phase
run_id
method
spatial_discretization_m
angular_discretization_deg
neighbor_radius_r
terrain/scenario_id
terrain/scenario_seed
method_seed_or_checkpoint_seed, if applicable
initial_defender_strategy
attacker_br_backend_config/checkpoint
status
trajectory_success
local_sse_success
num_local_iterations
num_defender_candidates_evaluated
num_attacker_br_calls
cumulative_attacker_br_runtime_sec
local_sse_total_runtime_sec
J_A_final
J_D_final
Delta_J_A_vs_bellman, when reference exists
Delta_J_D_vs_bellman, when reference exists
normalized_error_or_accuracy, only if approved
stopping_reason
failure_type
repository/config/checkpoint provenance
```

Bellman-relative fields for Tabular/DQN must be paired to the Bellman result from the exact same `(condition, scenario)` instance.

### Aggregate Statistics

For each computational condition and method, aggregate over the approved Monte Carlo design.

At minimum compute, where defined:

- sample count,
- success/failure count and rate,
- mean runtime,
- median runtime,
- runtime variance or standard deviation,
- mean/median final `J_A`,
- mean/median final `J_D`,
- mean/median Bellman-relative `Delta_J_A`,
- mean/median Bellman-relative `Delta_J_D`,
- user-approved normalized accuracy/error statistics,
- uncertainty summary using the user-approved reporting convention.

Do not assume a confidence-interval method without checking whether the user has already specified one. If not, Codex may compute basic descriptive statistics first and must request approval before labeling a particular interval (e.g., 95% t-interval, bootstrap CI) as the official paper uncertainty measure.

### Required Visualizations

Generate publication-oriented Phase-7 figures from the raw JSON/aggregate data. The exact final styling can be refined later, but the data products must support at least:

#### 1. Runtime vs spatial discretization
Separate or faceted plots for Bellman, Tabular, and DQN at controlled angular resolution / `r`, or an approved aggregation.

#### 2. Runtime vs angular discretization
At controlled spatial discretization / `r`.

#### 3. Runtime vs Local-SSE radius `r`
At controlled spatial/angular discretization.

#### 4. Bellman-relative `J_A` error vs each computational variable
Use the approved normalized metric where available; otherwise show raw `Delta_J_A`.

#### 5. Bellman-relative `J_D` error vs each computational variable
Use the approved normalized metric where available; otherwise show raw `Delta_J_D`.

#### 6. Runtime-saving vs objective-error plot
Show the empirical speed/accuracy tradeoff without declaring a preferred regime yet.

#### 7. Computational-condition heatmaps
For example:

```text
(spatial discretization, angular discretization) -> runtime
(spatial discretization, angular discretization) -> objective error
```

with separate panels/facets by `r` where appropriate.

#### 8. Monte Carlo variability plots
Show the approved mean/median and uncertainty/dispersion for runtime and objective error.

#### 9. Timeout/intractability coverage map
Show which condition cells completed, timed out, or failed for each method.

Use PNG for normal plots. Use HTML only for a genuinely useful interactive 3D result surface; do not create HTML merely because the data have three axes.

### Validation Tests

Before accepting Phase-7 results, verify at minimum:

1. every run uses an approved computational condition,
2. every DQN result uses a checkpoint whose `(spatial, angular)` signature exactly matches the run,
3. no A2 cross-discretization model is used,
4. Phase-7 terrains/scenarios are absent from DQN training/validation datasets,
5. Bellman/Tabular/DQN paired trials use the same scenario realization and Local-SSE initialization,
6. Local-SSE `r` is identical across methods in each paired trial,
7. `J_A` and `J_D` are recomputed by the common authoritative evaluators,
8. Bellman-relative differences are paired to the exact same condition/scenario,
9. Tabular stochastic seeds follow the approved protocol,
10. DQN weights/normalization are never updated during Phase-7 evaluation,
11. runtime boundaries are consistent across repeats,
12. timeout/failure runs remain in the raw dataset,
13. aggregate statistics reproduce exactly from raw JSON,
14. resuming an interrupted batch does not duplicate completed trials,
15. Phase 0/1/6 regression tests remain valid.

### Failure / Stop Conditions

Do not declare Phase 7 complete if:

- the official condition matrix was not user-approved,
- required condition-specific DQN checkpoints are missing,
- Phase-7 terrain leakage occurs,
- DQN or Tabular is tuned using Phase-7 outcomes,
- different methods solve different scenario realizations under the same nominal trial,
- `r` changes meaning between methods,
- normalized accuracy/error is reported using an unapproved formula,
- timeout runs are silently dropped,
- DQN offline training time is silently mixed with online runtime,
- raw JSON results cannot reproduce the reported aggregate plots/statistics,
- the final Bellman-vs-RL regime boundary is chosen during Phase 7.

### Phase 7 User-Confirmation Gates

The following choices cannot be inferred safely from the existing repository and must be confirmed before official Phase-7 execution if not already resolved:

1. **Official condition matrix**
   - whether to run the full `6 × 4 × 5 = 120` factorial,
   - or an explicitly approved reduced matrix if the full design is operationally prohibitive.

2. **Condition-specific DQN coverage**
   - whether every selected `(spatial, angular)` pair must receive its own independently trained generalized DQN checkpoint before the sweep,
   - and whether the already approved Phase-4 training protocol is reused identically across all pairs except mechanically required dimensional changes.

3. **Phase-7 Monte Carlo test distribution and sample size**
   - terrain families/complexity range,
   - number of independent terrain/scenario realizations per condition,
   - scenario-generation seeds,
   - whether defender/sensing configuration is fixed or sampled from an approved distribution.

4. **Stochastic replication protocol**
   - number of Tabular retraining seeds per `(condition, scenario)`,
   - whether multiple independent DQN training seeds/checkpoints per discretization pair are required for the official study,
   - number of timing repetitions/warm-ups where needed.

5. **Operational timeout/resource budget**
   - maximum allowed runtime per Local-SSE run and/or BR call,
   - retry policy,
   - status used for censored/intractable runs.

6. **Official normalized `J_A` / `J_D` accuracy-error definition**
   - only if it was not already resolved in Phase 5/6.

7. **Official uncertainty reporting convention**
   - descriptive mean/std only,
   - or a specific confidence-interval method for paper figures.

8. **DQN offline-training cost interpretation**
   - this may remain unresolved through raw Phase-7 data collection because offline and online times are stored separately,
   - but it must be resolved before Phase-8 regime classification if a combined computation-cost rule will be reported.

Codex must first use current code and prior manifests to fill every repository-grounded fact. It should ask the user only for genuine experiment-design choices that remain unresolved.

### Exit Criteria

Phase 7 is complete only when:

1. the user-approved computational-condition matrix is frozen,
2. every included `(spatial, angular)` DQN condition has an approved compatible generalized checkpoint,
3. the dedicated Phase-7 unseen Monte Carlo manifest is frozen and leakage checks pass,
4. the approved stochastic-replication and timeout protocols are frozen,
5. Bellman, Tabular, and DQN Local-SSE runs are executed according to the paired design for all scheduled trials or explicitly recorded as timeout/failure,
6. raw per-run JSON is complete and resumable,
7. attacker-BR and total Local-SSE runtimes are recorded consistently,
8. final `J_A` and `J_D` are recorded through the common evaluators,
9. Bellman-relative raw differences and any approved normalized metrics are computed correctly,
10. Monte Carlo descriptive/uncertainty statistics are generated from raw data,
11. required runtime, error, heatmap, variability, and completion/timeout figures are generated,
12. no test-data tuning or A2 cross-discretization DQN is introduced,
13. DQN offline training and online inference/Local-SSE costs remain separately recoverable,
14. all Phase-7 validation tests pass or unresolved failures are explicitly documented,
15. the resulting dataset is sufficient for Phase 8 to derive the Bellman-vs-RL computational-regime classification without rerunning or redefining the Phase-7 experiment.

### Detailed Codex Command

Implement **Phase 7 only** according to all requirements above. Begin by inspecting the latest Phase 0–6 artifacts and current `3D_0827/` / `3D_RL_DQN/` code. Resolve repository-grounded values directly from code/manifests, then present only the unresolved Phase-7 experiment-design gates to the user. After those gates are approved, freeze the condition matrix and Monte Carlo manifest, ensure all required A1 condition-specific DQN checkpoints exist, and run the paired Bellman/Tabular/DQN Local-SSE sweep with resumable raw JSON logging. Generate aggregate scaling/error/variability figures, but do **not** derive the final Bellman-vs-RL regime-selection rule; reserve that for Phase 8.

---

## Phase 8 — Bellman vs RL Computational-Regime Classification

### Objective
Use the frozen Phase-7 Monte Carlo dataset to answer the primary research question:

> **At what computational conditions should the framework use exact Bellman DP, and at what conditions is an RL-based attacker-BR approximation computationally preferable while remaining within an approved approximation-quality threshold?**

The computational condition is

\[
c=(\Delta_s,\Delta_\theta,r),
\]

where

\[
\Delta_s=\Delta x=\Delta y=\Delta h,
\qquad
\Delta_\theta=\Delta\psi=\Delta\gamma,
\]

and `r` is the Local-SSE Chebyshev neighbor-search radius.

Phase 8 is an **analysis/classification phase only**. It must not retrain any model, modify the solver, tune on Phase-7 outcomes, regenerate the official Monte Carlo dataset, or alter the approved objective definitions.

### Primary Comparison
The final regime-selection question is primarily:

```text
Bellman DP vs generalized DQN approximation
```

Tabular Q-learning remains in the Phase-8 analysis as a **case-specific RL baseline / historical fast-prototyping comparator**, but it is not automatically treated as the final generalized RL method because it requires retraining for each terrain/case.

If the user later explicitly requests a three-way production rule among Bellman, Tabular, and DQN, construct it as a separate secondary analysis from the same frozen Phase-7 data. Do not silently replace the primary Bellman-vs-DQN research question with a three-way selector.

### Non-Negotiable Grounding Rules
Before analysis, Codex must load and verify the latest:

- Phase-7 raw per-run JSON,
- Phase-7 aggregate tables,
- condition matrix,
- Monte Carlo manifest,
- timeout/failure log,
- objective-metric definition manifest,
- DQN checkpoint/training-cost provenance,
- any user-approved normalized `J_A` / `J_D` error formulas,
- any user-approved uncertainty/statistical reporting convention.

Do not infer missing thresholds or formulas from examples in prior discussion. In particular, examples such as "80% accuracy" are explanatory examples only unless the user explicitly approves them as official thresholds.

### Phase-8 Analysis Inputs
For each observed condition

\[
c=(\Delta_s,\Delta_\theta,r),
\]

aggregate the paired Monte Carlo results for at least:

```text
Bellman:
    success/completion status
    cumulative_attacker_br_runtime_sec
    local_sse_total_runtime_sec
    J_A
    J_D

DQN:
    success/completion status
    cumulative_attacker_br_runtime_sec
    local_sse_total_runtime_sec
    J_A
    J_D
    frozen checkpoint/config provenance

Tabular baseline:
    success/completion status
    cumulative_attacker_br_runtime_sec
    local_sse_total_runtime_sec
    J_A
    J_D
```

Retain both raw paired samples and the approved aggregate statistics. Never classify using only a plot image when raw JSON is available.

### Accuracy / Approximation Metrics
Always retain raw paired differences:

\[
\Delta J_A = J_A^{RL}-J_A^{Bellman},
\]

\[
\Delta J_D = J_D^{RL}-J_D^{Bellman}.
\]

If an official normalized error/accuracy definition was approved in Phase 5–7, use it unchanged. Otherwise, Phase 8 must stop before official classification and ask the user to approve the exact normalized metric.

Conceptually denote the approved approximation errors as:

\[
E_A(c), \qquad E_D(c).
\]

Do not define their formulas by assumption in this phase.

### Required Classification Logic
Use a **quality gate first, computation comparison second**.

#### Step 1 — RL approximation-quality admissibility
For each observed condition `c`, determine whether DQN satisfies the user-approved quality requirements.

Conceptually:

\[
\text{RL\_admissible}(c)
=
\Big(E_A(c)\le \epsilon_A\Big)
\land
\Big(E_D(c)\le \epsilon_D\Big)
\land
\Big(S_{DQN}(c)\ge S_{min}\Big),
\]

where:

- `epsilon_A` = user-approved `J_A` approximation-error threshold,
- `epsilon_D` = user-approved `J_D` approximation-error threshold,
- `S_min` = user-approved minimum DQN success/completion requirement.

This equation is a structural template only. The exact aggregate statistic used for `E_A`, `E_D`, and `S_DQN` must follow the user-approved Phase-8 confidence/robustness convention.

If DQN fails the quality gate, classify the condition as **Bellman-preferred due to approximation quality**, provided Bellman itself is computationally available.

#### Step 2 — Computational comparison
Only for conditions where RL is quality-admissible, compare Bellman and DQN computation cost using the user-approved runtime basis.

Maintain both:

\[
T^{BR}_{B}(c),\qquad T^{BR}_{DQN}(c),
\]

and

\[
T^{SSE}_{B}(c),\qquad T^{SSE}_{DQN}(c).
\]

Compute transparent quantities such as:

\[
R_T^{SSE}(c)=\frac{T^{SSE}_{B}(c)}{T^{SSE}_{DQN}(c)},
\]

and, if useful,

\[
\Delta T^{SSE}(c)=T^{SSE}_{B}(c)-T^{SSE}_{DQN}(c).
\]

Do not invent a minimum speedup ratio. If the user wants a minimum computational benefit such as `R_T >= R_min`, Codex must request and record the approved `R_min`.

#### Step 3 — Base regime labels
Support at least the following internal labels:

```text
BELLMAN_QUALITY_REQUIRED
    DQN fails the approved approximation-quality gate and Bellman completes.

DQN_COMPUTATIONALLY_PREFERRED
    DQN passes the approved quality gate and satisfies the approved computation-saving rule.

BELLMAN_COMPUTATIONALLY_PREFERRED
    DQN passes the quality gate, but Bellman is equal/faster under the approved runtime rule.

CONDITION_DEPENDENT
    The Monte Carlo uncertainty/replication results do not support a stable assignment under the approved confidence rule.

DQN_ONLY_PRACTICAL
    Bellman reaches the approved timeout/resource limit while DQN passes the approved quality/success requirement under whatever reference comparison remains valid.

UNRESOLVED_NO_EXACT_REFERENCE
    Bellman does not complete and there is insufficient exact-reference information to verify DQN approximation quality for that condition.

BOTH_FAILED_OR_INVALID
    Neither method provides a valid result under the approved protocol.
```

The final paper-facing labels may be simplified after user review, but the raw internal reason code must be retained.

### Bellman Timeout / No-Reference Handling
A Bellman timeout is itself an important tractability result, but it creates a reference-quality problem.

Therefore:

- Do not call DQN "accurate" merely because it returns a trajectory when Bellman times out.
- If Bellman completed for enough paired scenarios/nearby repetitions at the same exact condition to evaluate the approved DQN-quality criterion, use only the valid paired exact-reference subset and report coverage explicitly.
- If no adequate exact Bellman reference exists at a condition, classify it as `UNRESOLVED_NO_EXACT_REFERENCE` rather than inventing an accuracy estimate.
- If the user later approves a secondary surrogate/reference strategy, record it as a separate analysis and never relabel it as exact Bellman validation.

### DQN Offline Training-Cost Treatment
The Phase-4 DQN training cost must remain separate from online BR/Local-SSE inference cost unless the user approves a specific deployment accounting rule.

Support the following accounting views without choosing one automatically:

#### View A — Online-query comparison

\[
T_{DQN,online}=T_{inference/SSE}
\]

Use when asking whether a previously trained generalized DQN is preferable for a new terrain/query.

#### View B — One-shot total cost

\[
T_{DQN,total}=T_{train}+T_{online}
\]

Use only if the experiment is explicitly interpreted as a single-use deployment.

#### View C — Amortized repeated-query cost

For `N` future queries using the same compatible condition-specific generalized DQN:

\[
\bar T_{DQN}(N)
=
\frac{T_{train}}{N}+T_{online}.
\]

If this view is approved, calculate the empirical break-even query count where DQN becomes computationally cheaper than Bellman, if such a positive finite break-even exists.

Do not choose `N`, one-shot, or online-only interpretation without user approval.

### Observed-Condition Classification First
The official primary classification must be defined on **observed Phase-7 condition cells only**.

For each tested

\[
(\Delta_s,\Delta_\theta,r),
\]

produce one classification record with:

```text
spatial_discretization_m
angular_discretization_deg
neighbor_radius_r
bellman_runtime_summary
dqn_runtime_summary
bellman_completion_rate
dqn_success_rate
J_A_error_summary
J_D_error_summary
quality_gate_pass
computation_gate_pass
regime_label
reason_code
uncertainty_flag
reference_coverage
```

Do not interpolate, extrapolate, smooth, or fit an artificial boundary across unobserved conditions for the official result unless the user explicitly approves a secondary boundary-model analysis.

### Optional Secondary Boundary Summary
After the cell-level empirical classification is complete, Codex may summarize an interpretable regime boundary **only if the observed data support it** and only after user approval.

Examples of acceptable summaries if empirically justified:

```text
For fixed angular resolution and r, Bellman remains preferable down to X m,
while DQN becomes preferable at finer spatial discretization.
```

or

```text
At r >= k and Delta_s <= X, Bellman exceeded the runtime budget while DQN
remained within the approved objective-error threshold.
```

Do not force monotonicity. If the empirical regime map is non-monotonic, report the non-monotonic result rather than manufacturing a clean threshold.

### Tabular Q-Learning Secondary Analysis
Retain Tabular results to quantify the limitation that motivated DQN:

- per-case retraining cost,
- final `J_A/J_D` approximation quality,
- total Local-SSE runtime,
- comparison against Bellman and DQN.

Recommended Phase-8 interpretation outputs include:

```text
Bellman vs DQN -> primary deployment/regime question
Bellman vs Tabular -> prototype / retraining-cost baseline
Tabular vs DQN -> effect of terrain-generalized reuse
```

Do not let Tabular-specific results alter the DQN classification thresholds unless the user explicitly requests that rule.

### Required Outputs
Create machine-readable Phase-8 outputs in the existing result-folder convention, preferably JSON for canonical data.

At minimum produce:

```text
phase8_regime_config.json
phase8_condition_classification.json
phase8_summary.json
```

The configuration JSON must record:

- source Phase-7 dataset/version,
- source commit/config signatures where available,
- approved `J_A` / `J_D` error formulas,
- `epsilon_A`,
- `epsilon_D`,
- success threshold,
- uncertainty/confidence rule,
- runtime basis,
- any minimum speedup rule,
- DQN training-cost accounting view,
- timeout handling rule,
- whether any secondary boundary fitting was authorized.

### Required Visualizations
Generate at least the following once the user-approved classification rule is frozen:

#### 1. Computational-regime classification map
For each `r`, show a 2D grid:

```text
x-axis: spatial discretization
 y-axis: angular discretization
 cell: regime label
```

This is the primary visualization for the research question.

#### 2. Runtime-vs-accuracy tradeoff with regime labels
Show observed DQN/Bellman tradeoff points with the approved quality threshold(s) overlaid.

#### 3. Bellman/DQN runtime-ratio heatmap

\[
(\Delta_s,\Delta_\theta)\rightarrow R_T
\]

faceted by `r`.

#### 4. `J_A` approximation-error heatmap
Faceted by `r`.

#### 5. `J_D` approximation-error heatmap
Faceted by `r`.

#### 6. Completion/timeout map
Show Bellman completed / timed out / invalid and DQN success/failure status.

#### 7. Final regime summary table
For every observed condition or compact empirically supported group, report:

- condition,
- Bellman computation result,
- DQN computation result,
- approximation-quality result,
- final regime label,
- reason.

#### 8. Optional amortization/break-even plot
Only if the user approves amortized DQN training-cost accounting.

Use PNG for standard figures. HTML is unnecessary unless a genuinely useful interactive 3D surface is explicitly requested.

### Validation Tests
Before accepting Phase 8, verify at minimum:

1. Phase-7 raw dataset is frozen and unchanged by Phase 8,
2. every classification row maps to an observed Phase-7 condition cell,
3. no Phase-8 model retraining or fine-tuning occurs,
4. no threshold was inferred from an illustrative example,
5. the approved `J_A` / `J_D` formulas are applied identically across all cells,
6. paired Bellman/DQN statistics use matching scenario realizations where exact comparison is claimed,
7. Bellman timeout cells are not assigned fake exact-reference accuracy,
8. DQN training cost is not silently merged with online runtime,
9. regime labels are reproducible from `phase8_regime_config.json` and Phase-7 raw JSON alone,
10. classification figures reproduce from the machine-readable classification JSON,
11. no unobserved condition is presented as empirically classified,
12. any secondary fitted/interpolated boundary is visually and numerically distinguished from observed cell classifications,
13. Tabular analysis remains secondary unless the user explicitly changes the research question,
14. raw reason codes are retained even if paper-facing labels are simplified.

### Failure / Stop Conditions
Do not declare Phase 8 complete if:

- official `J_A` / `J_D` approximation thresholds are missing,
- the official DQN success criterion is missing,
- the uncertainty/robustness rule for deciding whether a condition passes the quality gate is missing,
- the runtime basis for the primary regime decision is unspecified,
- the DQN offline-training-cost interpretation is required by the intended conclusion but has not been approved,
- Phase-7 exact-reference coverage is insufficient but the analysis nevertheless labels DQN accuracy as verified,
- the classification rule changes from cell to cell without an explicitly documented reason,
- a fitted boundary is presented as observed fact,
- Phase-8 analysis modifies or tunes the DQN using Phase-7 results.

### Phase 8 User-Confirmation Gates
The following are genuine research-design choices and must be confirmed by the user before the final regime map is declared official if they have not already been resolved in earlier phases:

1. **Official `J_A` approximation-quality threshold**
   - exact metric/formula,
   - approved threshold `epsilon_A`.

2. **Official `J_D` approximation-quality threshold**
   - exact metric/formula,
   - approved threshold `epsilon_D`.

3. **DQN success/completion requirement**
   - e.g., whether every paired test must succeed or whether a specified minimum Monte Carlo success rate is acceptable.

4. **Robustness / uncertainty rule for a condition cell**
   - whether the quality threshold applies to the mean,
   - median,
   - a chosen upper quantile,
   - confidence-bound criterion,
   - or another approved aggregate rule.

5. **Primary runtime basis**
   - attacker-BR runtime,
   - total Local-SSE runtime,
   - or a hierarchical rule using both.
   - Both are still reported regardless of which one drives the final regime label.

6. **Minimum computation-saving requirement**
   - whether any DQN speed advantage is sufficient once quality passes,
   - or whether a minimum speedup ratio / absolute time saving is required.

7. **DQN training-cost accounting interpretation**
   - online-only pretrained deployment,
   - one-shot `training + inference`,
   - amortized repeated-query deployment,
   - or multiple views reported side by side with one identified as the primary interpretation.

8. **Handling of Bellman timeout cells without adequate exact-reference coverage**
   - default design is `UNRESOLVED_NO_EXACT_REFERENCE`;
   - any stronger claim requires a separately approved reference strategy.

9. **Whether a compact fitted boundary is desired**
   - default is observed-cell classification only,
   - any interpolation/regression/decision-tree boundary is secondary and requires approval.

Codex must first read all prior phase manifests and reuse already approved definitions. It should ask the user only for gates that remain genuinely unresolved.

### Exit Criteria
Phase 8 is complete only when:

1. the frozen Phase-7 dataset and manifests are loaded and verified,
2. all required Phase-8 thresholds and decision rules are user-approved and recorded,
3. the primary Bellman-vs-DQN regime logic is frozen before generating the final classification,
4. every observed condition cell receives a reproducible regime label or an explicit unresolved/failure status,
5. Bellman timeout/no-reference conditions are handled without fabricating approximation accuracy,
6. DQN offline and online costs remain explicitly separable and are interpreted according to the approved deployment view,
7. Tabular Q-learning is summarized as the retraining-based RL baseline without replacing the primary Bellman-vs-DQN question,
8. machine-readable regime configuration, per-condition classification, and summary JSON files are generated,
9. the computational-regime map, runtime/accuracy plots, error heatmaps, timeout map, and final summary table are generated,
10. all outputs reproduce from the frozen Phase-7 data plus the frozen Phase-8 regime configuration,
11. no model retraining/tuning occurs in Phase 8,
12. no unobserved computational condition is presented as empirically verified,
13. any optional fitted boundary is clearly secondary and separately identified,
14. the final outputs directly answer the research question: **under which observed discretization / Local-SSE-neighborhood conditions Bellman DP is retained versus when generalized RL approximation is computationally preferred under the approved approximation-quality requirement.**

### Detailed Codex Command
Implement **Phase 8 only**. Begin by loading the latest frozen Phase-7 raw JSON, aggregate results, manifests, timeout/failure records, and all previously approved metric definitions. Do not rerun or modify the solver and do not retrain or fine-tune any RL model. Resolve all prior approved definitions from the repository/manifests first, then present only the still-unresolved Phase-8 user-confirmation gates. After those gates are approved, freeze `phase8_regime_config.json` and classify each **observed** `(spatial discretization, angular discretization, Local-SSE radius)` condition using the quality-gate-first, computation-comparison-second logic described above. Generate machine-readable classification JSON and the required regime/error/runtime/timeout visualizations. Treat Bellman timeout cells without adequate exact reference as unresolved rather than fabricating DQN accuracy. Keep DQN offline training cost distinct from online runtime according to the approved accounting interpretation. Use Tabular Q-learning as the secondary retraining-based RL baseline. Do not interpolate or fit an official boundary beyond observed Phase-7 cells unless the user explicitly authorizes a clearly labeled secondary boundary analysis.

---

# Phase 16.4 Extension Design Record — Environment, Mission, Sensor, and Early-Stopping Variation

## Status and Scope

This section records the user/team design discussion and the repository-grounded audit performed on 2026-10-08. It is a requirements and brainstorming record only. It does not authorize implementation, retraining, or modification of the frozen Phase-16 results. No item in this section is to be implemented until the user gives an explicit implementation instruction.

The proposed extensions are:

1. variation of environment/terrain type,
2. variation of launch/start point,
3. variation of goal point or goal region,
4. variation of a fixed single-sensor position,
5. a training stopping rule based on approved approximation quality and stable validation performance.

## Frozen Physical Interpretation of the Current Attacker

The Attacker is an abstract standoff-weapon or mothership/daughtership model. Only the daughtership is tracked as the Attacker after separation.

The current hybrid model remains a single irreversible transition from powered mode to glide mode.

The physical/modeling interpretation is:

- the powered vehicle travels from the launch point to one switching point,
- the powered trajectory is a straight line,
- at the switching point the daughtership/standoff weapon separates, or the Attacker irreversibly shuts down propulsion,
- after switching, the Attacker is unpowered,
- the glide phase can only descend and is subject to the existing glide-control, turn, terrain, energy, and reachability constraints,
- after entering glide mode, the Attacker may exploit any later terrain occlusion encountered along the glide trajectory,
- LOS may change from hidden to visible to hidden multiple times during one glide trajectory without causing another mode switch.

Complex terrain, multiple obstacles, corridors, and repeated LOS changes do not by themselves invalidate the one-switch model. LOS state changes and physical mode switching are distinct events.

### Alternative Fixed-Wing Interpretation

A fixed-wing UAS that repeatedly turns its engine off and on to minimize probability of detection is a materially different reversible hybrid model. That model would require multiple switches plus physical rules such as minimum dwell time, restart delay/cost, fuel/energy accounting, acoustic transients, or an explicit switching penalty. Without these constraints, the optimizer could produce unrealistic step-to-step chattering.

The reversible fixed-wing model is a possible later extension or sensitivity study. It must not silently replace the current one-way abstract standoff/daughtership model.

## Current Powered-Phase and Switching Contract Confirmed from Code

The existing implementation confirms the following:

- StraightPoweredPhaseModel connects the fixed mission launch point to a switching point with one constant-speed straight segment.
- Powered feasibility checks that this segment does not intersect terrain and that it produces a valid horizontal arrival heading.
- Switching candidates are sampled from the sensor-centered terrain LOS tangent surface and snapped to the glide lattice.
- Candidates are retained only if they are powered-feasible and lie in the goal-backward-reachable glide set.
- The powered contribution to J_A currently uses zero hazard and normalized powered-flight time only.
- A DQN episode starts at a sampled switching state in glide mode. It does not simulate the powered segment step by step.
- Final switching-state selection adds the authoritative powered cost to the learned or exact glide cost.

The user confirmed that switching candidates should remain restricted to the LOS tangent surface associated with terrain visible from the sensor. Terrain hidden behind the visible horizon is not intended to contribute a separate initial switching surface, although it may provide later occlusion during the glide trajectory.

## Environment / Terrain Variation Requirements

### Training Terrain Families

Training is intended to include defined terrain families and/or procedurally generated arrangements of multiple boxes. Corridor generation may be constructed naturally from box arrangements.

The requested corridor/wall cases include all of the following:

1. parallel-wall straight corridor,
2. L-shaped corridor,
3. U-shaped corridor,
4. branching corridor with possible dead ends,
5. walls of different heights.

Additional procedural variation may include box count, box position, footprint, height, spacing, gap width, corridor width, orientation, and topology/connectivity.

### Expected Qualitative Behavior

No outcome is assumed in advance.

- A corridor aligned with the sensor may expose a glide attacker for a long interval and increase detection probability.
- A corridor whose walls occlude the sensor may instead provide a protected route.
- A wall may allow an intelligent attacker to remain behind it.
- An intelligent defender may favor the end of a wall or a position with visibility of multiple sides, but this is a hypothesis to test rather than a conclusion to encode.
- Defender-disadvantage cases in which the sensor sees very little of the reachable region remain valid scenarios and must not be discarded merely because J_D is small.

### Final Unseen Test Terrain

The intended final generalization test is not limited to another box layout. Candidate final tests include an unseen urban closed mesh and a mountainous height map or DEM-like terrain.

Training/validation/test separation must be performed by terrain geometry or terrain-generation seed, not only by new sensor or goal coordinates on a training terrain. A coordinate-only split measures position interpolation but does not establish terrain generalization.

## Current Observation Contract Confirmed from Code

The current DQN observation is not an unrestricted global terrain map.

It contains:

- a heading-aligned 2,000 m by 2,000 m attacker-centered window,
- a fixed 21 by 21 spatial grid,
- 100 m sample spacing,
- one terrain-clearance channel,
- five sensor/hazard geometry channels,
- one in-domain validity channel,
- four ego scalar features,
- four goal-vector and goal-distance scalar features.

The current attacker graph domain is approximately 1,600 m by 800 m. Although the observation window's total width exceeds each map dimension, the window is centered on the Attacker and extends only 1,000 m in each heading-aligned direction. Terrain at the far side of the domain can therefore fall outside the observation when the Attacker is near an edge.

The fixed 100 m observation sampling is too coarse to reliably represent sub-100 m corridors, narrow gaps, thin urban buildings, or fine mountain ridges. A goal vector alone does not reveal a distant wall, branch, or dead end outside the terrain window.

### Observation Design Options Still Open

Before corridor/mesh implementation, the user must select or approve an observation design. Candidate designs are:

1. retain the 2 km window and increase its resolution, such as 41 by 41 or 81 by 81,
2. use a high-resolution local channel plus a low-resolution global channel,
3. use a map-wide resampled global channel,
4. adopt another representation suitable for both procedural boxes and arbitrary mesh/height-map terrain.

A multiscale local-plus-global observation is the current working suggestion, not an approved implementation decision. The minimum corridor width that must be represented is also unresolved and should drive the local sampling resolution.

## Goal and Start Variation

### Current Goal and Start Behavior

The current Phase-16 configuration fixes the launch/start at (-8, 0, 0) map units and the goal at (8, 0, 0) map units.

The current goal is a continuous physical point. It is not snapped to the lattice. A lattice state is terminal when it lies inside the inclusive three-dimensional 25 m goal sphere.

The fixed goal aligns with every tested 10/25/50/100 m lattice. An arbitrary continuous goal will not have that property. At 50 m or 100 m, some sampled goal points can have no lattice state inside the 25 m terminal sphere, producing no terminal state and no goal-backward-reachable graph.

### Requested Variation

- Define an allowed physical goal region.
- Sample a goal within that region according to the final approved scenario-generation design.
- Vary the physical launch/start position within an approved launch region.
- Continue to provide the relative goal vector to the DQN observation.

### Goal Terminal Definition Still Open

The cross-resolution goal rule must be frozen before implementation. Options include:

1. restrict goals to coordinates aligned with every tested lattice,
2. snap each physical goal to the condition-specific lattice,
3. define the physical goal region itself as the terminal set,
4. increase terminal tolerance as a function of discretization.

Using the same physical goal region as the terminal set is the current working suggestion because it preserves the mission definition across discretizations. It is not yet approved.

### Computational Consequence of Goal Variation

The current goal-backward-reachable graph is sensor-independent and is shared across all sensors for one terrain because the goal is fixed. Goal variation changes the terminal set and requires a different reachability graph for each terrain-instance and goal-definition pair.

The existing Phase-16.4 runtime cache is keyed only by terrain category. A goal-varying implementation must change scene/graph reuse to be keyed by at least terrain instance plus goal definition. Start and sensor variations may reuse that graph; goal variation may not.

## Sensor Variation

The sensor remains a single sensor, stationary for the entire episode, sampled from an approved potential-sensor region, and fixed after the episode begins. Multiple sensors are a later extension and are outside the present change.

The current code does not sample a continuous sensor region. It uses nine fixed locations from x = {3.25, 5.0, 6.75}, y = {-3.75, 0.0, 3.75}, and z = 0 map units. Continuous or larger discrete sensor variation requires a new scenario-generation contract.

## Proposed Scenario Sampling Strategy

Naive fully independent uniform sampling of terrain, start, goal, and sensor is not the preferred default. It can produce a training set dominated by trivial or geometrically redundant cases.

The working proposal is constrained sampling with geometry-based stratification:

1. choose terrain family,
2. choose or generate a terrain instance,
3. choose a geometry/difficulty bin,
4. sample a goal from the allowed goal region,
5. sample a launch point from the allowed launch region,
6. sample a sensor from the allowed potential-sensor region,
7. apply the approved geometric and mission-validity checks.

Candidate bins include low visibility, mixed visibility, high visibility, and defender-disadvantage or near-zero-visibility cases.

Difficulty bins should be defined by inexpensive geometry metrics rather than post-hoc Bellman J_A or J_D so that training-scenario generation does not require solving every proposed case first. Defender-disadvantage cases remain in the distribution with an explicit quota and separate reporting.

The exact sampling weights, geometry metric, and whether tuples are generated on demand or drawn from a finite scenario bank remain open decisions.

### Finite Scenario Bank vs On-the-Fly Generation

Fully regenerating terrain and goal geometry every episode is computationally expensive because goal changes require a new reachability graph. A finite bank of approved terrain/goal/start/sensor tuples, with graph sharing across tuples that have the same terrain and goal, is the current working suggestion. The size and resampling policy of that bank are unresolved.

## Failure-Episode Contract — Current Mismatch and Required Decision

The user stated that failure episodes should be learned rather than silently discarded. The current code does not implement that behavior.

Current behavior is:

- switching candidates outside the goal-backward-reachable set are removed,
- DQN actions are restricted to successors that remain goal-backward reachable,
- a scenario with no admissible switching state raises during setup,
- an infeasible Bellman reference raises during setup,
- official training therefore does not expose the policy to physically valid but goal-unreachable branches.

To train meaningful failure episodes, the action/state contract would need to include physically feasible goal-unreachable transitions and define failure termination and cost for altitude exhaustion, unreachable dead ends, horizon exhaustion, collision if represented, and absence of an admissible switching candidate.

An explicit failure terminal cost is necessary. With the current negative stage-cost reward, terminating early without a failure penalty could be incorrectly preferred because it avoids future cost.

Before implementation, the user must distinguish:

1. invalid mission geometry that should be rejected or resampled,
2. a physically impossible mission that should be represented as failure,
3. a feasible mission in which the learned policy chooses a failing branch,
4. a mission with no LOS-tangent switching candidate.

Any expanded failure action space and terminal cost must be applied identically to Bellman and DQN if exact approximation-quality comparison is retained.

## Powered-Phase Occlusion Issue Introduced by Start/Sensor Variation

The current powered objective always supplies zero hazard. The powered feasibility test checks terrain collision, but it does not explicitly verify that every point of the launch-to-switch segment remains hidden from the sensor.

The fixed baseline geometry was designed around a terrain-hidden powered phase and LOS-tangent switching. Once launch and sensor locations vary, that assumption is no longer automatically guaranteed.

Before implementation, one of the following must be approved:

1. sample only scenarios/candidates whose complete powered segment satisfies the intended occlusion assumption,
2. calculate powered-phase acoustic/visual hazard when the segment is exposed,
3. allow exposure with an explicitly defined powered-phase penalty.

The current user-stated physical intent favors option 1, but this has not yet been frozen as an implementation rule.

## Early-Stopping Requirements

### Confirmed Direction

The research comparison should measure the wall-clock training time required to first reach the approved approximation accuracy, rather than training every seed for an identical fixed episode count. Different seeds may therefore stop after different episode counts.

Raw per-episode training J_A alone is not sufficient when terrain, start, goal, and sensor vary. Scenario difficulty changes the absolute J_A scale and can create a false plateau. The stopping decision should use a fixed held-out validation suite and Bellman-relative J_A error.

The approved quality thresholds remain:

- validation median Bellman-relative J_A error no greater than 10%,
- validation maximum Bellman-relative J_A error no greater than 20%.

Training may stop successfully only when both are true:

1. the approved quality thresholds are satisfied,
2. recent validation performance is stable within the approved change band.

A low-quality plateau is not successful convergence. If validation plateaus without meeting the accuracy thresholds, training continues unchanged to the maximum episode budget. If the maximum budget is reached without satisfying the quality gate, the seed is recorded as a training failure.

### Working Parameter Proposal — Not Yet Frozen

The current candidate parameters are:

- evaluation interval: 1,500 episodes,
- minimum training budget: 12,000 episodes,
- stability window: latest four validation evaluations,
- maximum allowed change: 1%,
- quality median threshold: 10%,
- quality every-case threshold: 20%,
- failure-plateau behavior: continue to maximum episodes.

The user agreed to reduce the old 6,000-episode evaluation spacing, require a minimum training budget, retain the 10%/20% quality gate, continue low-quality plateaus to the maximum budget, permit different seed stopping episodes, and compare time-to-approved-accuracy. The exact 1,500/12,000/four-evaluation/1% tuple remains a working proposal until explicitly frozen.

## Required Reporting for the Extension

When implementation is later authorized, report at minimum:

- actual stopping episode per seed,
- wall-clock time to the first checkpoint satisfying the full stop rule,
- whether a seed stopped successfully or hit the maximum budget,
- validation median and maximum Bellman-relative J_A error at stop,
- per-terrain-family and per-difficulty-bin quality,
- goal-reaching/failure counts under the approved expanded failure contract,
- train/validation/test geometry identities and generation seeds,
- start, goal, and sensor sampling provenance,
- observation coverage/resolution used by the trained model,
- reusable graph/cache counts by terrain-goal pair.

## Open User Decisions Before Implementation

No implementation should begin until the following are resolved or explicitly deferred:

1. minimum corridor/gap width that the observation must resolve,
2. local/global/multiscale observation architecture,
3. point-goal versus physical goal-region terminal definition,
4. exact start and goal regions,
5. exact potential-sensor region and sampling distribution,
6. geometry-based difficulty metric and bin weights,
7. finite scenario-bank size versus on-the-fly generation,
8. invalid mission versus learnable failure semantics,
9. expanded failure action space and terminal failure cost,
10. powered-segment occlusion validation versus powered hazard modeling,
11. exact early-stopping interval, minimum budget, patience/window, and stability percentage,
12. urban mesh and mountain height-map ingestion/observation backend.

## Implementation Gate

This design record does not authorize code changes. Answering the open questions also does not by itself authorize implementation. Codex must wait for an explicit user instruction to implement, retrain, or run the extended experiment.

---


# Development Rule

Detailed Codex instructions will be constructed **one phase at a time**.

For every phase, the detailed command must explicitly define:

1. Objective
2. Existing code assumptions
3. Files / modules allowed to change
4. Files / modules that must remain unchanged
5. Mathematical / algorithmic definition
6. Required implementation tasks
7. Logging / instrumentation requirements
8. Visualization requirements
9. Validation tests
10. Failure conditions
11. Exit criteria
12. Expected deliverables

No later phase should be implemented before the current phase command has been reviewed and accepted.

---

# Current Status

- Overall phase architecture: CREATED
- Detailed Phase 0 command: IMPLEMENTED AND VALIDATED (2026-10-01)
- Detailed Phase 1 command: IMPLEMENTED AND VALIDATED (2026-10-02)
- Detailed Phase 2 command: IMPLEMENTED AND VALIDATED (2026-10-02)
- Detailed Phase 3 command: IMPLEMENTED AND VALIDATED (2026-10-02)
- Detailed Phase 4 command: IMPLEMENTED; PERFORMANCE TEST STOPPED AFTER EPISODE 6,000 CHECKPOINT (2026-10-05)
  - user-approved development split: three `TRAIN_SIMPLE` terrains and two held-out `VALIDATION_SIMPLE` terrains,
  - user-approved sensing distribution: 3 x 3 common sensor positions, producing 27 training and 18 validation scenarios,
  - one shared condition-specific DQN; terrain/scenario identifiers remain metadata only,
  - official condition: 25 m spatial and 5 degree angular discretization,
  - official budget: 60,000 episodes per seed for seeds 0, 1, and 2; evaluation every 1,500 episodes,
  - validation-first checkpoint selection and the existing 10% median / 20% every-case Bellman-relative thresholds,
  - CUDA required for the official run; scenario-qualified replay and periodic exact resume checkpoints implemented,
  - `Phase16_DQN_Terrain_Generalization.ipynb` added as the single switch-controlled Phase 16.1–16.4 entry point,
  - full DQN test suite passed (45 tests) and the 100 m CPU Phase 16.4 smoke pipeline completed,
  - all 45 official 25 m Bellman references are cached,
  - official seed 0 reached the durable episode-1,500 checkpoint; validation was 18/18 goal-reaching with 14.70% median and 100.08% maximum Bellman-relative `J_A` error, so the approved final quality criterion was not yet met at that early checkpoint,
  - evaluation now uses batched switching-candidate inference, bounded observation/terrain caches, deterministic MDP-cache cleanup, and terrain-at-a-time validation loading after the first evaluation exposed excessive runtime and memory growth,
  - the new evaluator matched the prior evaluator exactly for success, switching state, trajectory, `J_A`, and successful-candidate count; on one actual 25 m CUDA case it reduced evaluation from 46.70 s to 36.25 s (1.29x),
  - the official CUDA run resumed seed 0 from episode 1,500 after optimization and was stopped at the user's request immediately after the durable episode-3,000 checkpoint because the projected three-seed runtime was excessive,
  - episode 3,000 validation remained 18/18 goal-reaching, with 15.73% median and 136.25% maximum Bellman-relative `J_A` error; the validation-selected best checkpoint therefore remains episode 1,500,
  - Phase 16.4 is incomplete: seed 0 did not reach 60,000 episodes and seeds 1 and 2 were not started.
  - an exact batched AABB CUDA observation backend was subsequently added for the five approved box terrains; sampled CPU/CUDA observation tensors were bitwise equal across all five terrain categories and the full DQN suite passed 46 tests,
  - the CUDA smoke pipeline completed with unchanged quality outputs,
  - a user-approved 3,000-episode performance continuation ran seed 0 from episode 3,000 to 6,000 and then stopped automatically; its two 45-case evaluations took 399.34 s and 391.93 s versus 2,133.15 s for the preceding CPU-observation evaluation,
  - the 3,000-episode continuation took 1,590.29 s of trainer time (26.50 min), an estimated 3.30x end-to-end improvement over two prior-style intervals; checkpoint evaluation improved about 5.39x while non-evaluation training cost per environment step improved about 5%,
  - a warmed post-change profile found that one representative 5.62 s validation case spent 3.92 s in CPU feasible-action masks, 1.11 s in CPU transitions, 0.445 s in GPU observation construction, and 0.078 s in DQN forward passes; the authoritative CPU graph lookup is now the main steady-state evaluation bottleneck,
  - the CPU graph lookup was then changed to bounded terrain-level structural action-row caches shared across sensor scenarios, with batched authoritative cache-miss construction and lightweight structural evaluation transitions; sensor-specific hazard/cost rows and final authoritative trajectory scoring were preserved,
  - the episode-6,000 checkpoint reproduced objective, trajectory, switching state, and successful-candidate count exactly for all 45 cases; full evaluation fell from 391.93 s to 213.09 s (1.84x), while the nine stepped-pyramid sensor cases fell from 74.97 s to 31.20 s (2.40x),
  - with gradient-training time conservatively left unchanged, the projected 60,000-episode three-seed design is now about 20.78 h total (6.93 h per seed), down from 27.16 h after the observation-only optimization,
  - episode 6,000 validation was 18/18 goal-reaching with 13.02% median and 97.95% maximum Bellman-relative `J_A` error; it became the validation-selected best checkpoint but still failed the approved 10% median / 20% every-case criterion,
  - the original 60,000-episode, three-seed design remains incomplete and is not running.
- Detailed Phase 5 command: CREATED — pending user review / execution-time unseen-test design gates
- Detailed Phase 6 command: CREATED — pending user review / execution-time integration gates
- Detailed Phase 7 command: CREATED — pending user review / execution-time Monte Carlo design gates
- Detailed Phase 8 command: CREATED — pending user review / final regime-threshold design gates
