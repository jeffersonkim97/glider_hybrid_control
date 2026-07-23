# Hierarchical Stackelberg Security Problem

## Phase 0–7: Acceptance Criteria and Implementation Checklist

**Status:** final Phase 0 acceptance contract. This document defines completion; it does not claim that future implementation work is already complete.

> **SUPERSEDED (2026-07-23):** Following advisor review, the Attacker best response is computed exclusively by Bellman dynamic programming (`p1b_4D.bellman.select_authoritative_bellman_response`). The CasADi/IPOPT NLP refinement stage, candidate filtering/Top-K warm-start selection, and the "Bellman → filter → warm-start → multi-start NLP → selection" pipeline described in ATT-03 through ATT-05 and elsewhere below are **deprecated and disconnected** from the authoritative pipeline (`attacker_nlp.py`, `candidate_filtering.py` are retained only as offline experimental comparison code, never imported by `stackelberg_solver.py` or the notebook). Read NLP/warm-start/Top-K acceptance items below as historical design context, not current pass conditions. See `stackelberg_solver.py`, `bellman.py`, and the notebook's Phase 7–8 cells for the current architecture.

Every future phase and the final notebook shall satisfy all applicable requirements in this document together with the governing Phase 0–1 through Phase 0–6 specifications. A phase is incomplete when required evidence is missing, even if its primary computation appears to run.

## 1. Acceptance status model

Every acceptance item has one of four statuses:

- **Not started** — no implementation evidence exists.
- **In progress** — implementation or evidence is incomplete.
- **Passed** — all stated conditions and evidence requirements pass.
- **Blocked** — an explicit external dependency prevents evaluation; the blocker and affected criteria are recorded.

“Not applicable” is permitted only when the criterion itself explicitly allows it. Warnings do not replace failures. A mandatory failed check makes its phase incomplete.

For each item, acceptance evidence shall record:

- criterion ID;
- status;
- validation or test identifier;
- result artifact and schema version;
- configuration/run identity;
- measured value and configured tolerance when numerical;
- diagnostic message;
- reviewer or automated-check provenance.

## 2. Final project completion

The project is complete only when the notebook implements and demonstrates the full sequence:

**Continuous Defender Optimization → Nested Attacker Optimization → Best-found Attacker Response → Defender Evaluation → Final Stackelberg Solution**

Completion requires:

1. all 15 fixed notebook phases are present in their prescribed order;
2. every required computational phase executes successfully from explicit dependencies;
3. all mandatory acceptance criteria pass;
4. all required JSON/NPZ bundles pass import and compatibility validation;
5. all five required figures are reproduced solely from imported bundles;
6. the final result retains complete configuration, solver, validation, and dependency provenance;
7. no result is described as globally optimal without an independent proof.

## 3. Mathematical consistency criteria

### MATH-01 — Stackelberg order

**Pass condition:** the Defender selects continuous (z_{mathrm{sensor}}) first; the complete Attacker solver evaluates that decision afterward.

**Evidence:** final solver call/dependency trace showing one full Attacker solve per effective Defender evaluation.

**Failure:** simultaneous decision treatment, reversed move order, or an Attacker response computed for a different Defender decision.

### MATH-02 — Attacker objective

**Pass condition:** Bellman, filtering/ranking where objective-based, NLP, response selection, validation, and reporting use the same normalized (J_A), weights, and component definitions.

**Evidence:** exported objective-contract ID plus independently reconstructed component totals for Bellman and NLP records.

**Failure:** weight changes for diversity, inconsistent normalization, or selecting under a different objective.

### MATH-03 — Defender objective

**Pass condition:** every (J_D) uses mission PoD from the associated Best-found Attacker Response and normalized LOS coverage area.

**Evidence:** Defender Evaluation records with response IDs and reproducible component totals.

**Failure:** use of a Bellman-only trajectory, unselected NLP result, non-LOS coverage definition, or mismatched response.

### MATH-04 — Hybrid mission

**Pass condition:** the strategy and metrics preserve powered and glide phases, the prescribed detection allocation, and mission time equal to powered time plus glide time.

**Evidence:** phase-specific detection/time components and boundary validation.

### MATH-05 — Trajectory boundary and continuity

**Pass condition:** the glide begins at the refined switching point and converges to ((z_{mathrm{goal}},0)) within configured tolerance; switching position/profile values are mutually consistent.

**Evidence:** switching-point residual and terminal goal residual.

### MATH-06 — Terrain and LOS

**Pass condition:** terrain, sensor height, LOS geometry, occlusion, LOS tangent, and coverage conform to the mathematical specification.

**Evidence:** geometry validation report and reference-case checks.

## 4. Attacker-solver criteria

### ATT-01 — Multi-start Bellman

**Pass condition:** multiple configured starts are evaluated under the unchanged Attacker objective and produce a traceable candidate set containing switching point, path, speed, gamma, objective, validation, and metadata.

### ATT-02 — Candidate filtering

**Pass condition:** filtering, duplicate removal, deterministic ranking, and Top-(K) selection produce accepted/rejected IDs and reasons without changing objective weights.

### ATT-03 — Bellman-to-NLP interface

**Pass condition:** every selected candidate maps to a dimensionally valid NLP warm start with preserved parent ID, boundary conditions, trajectory, speed, and gamma.

### ATT-04 — Multi-start CasADi NLP

**Pass condition:** every supplied warm start has an explicit attempted-solution record, including failures; feasible results contain refined switching point, trajectory, speed, gamma, controls, objectives, status, residuals, and validation.

### ATT-05 — Best-found response

**Pass condition:** selection considers all feasible refined NLP solutions and returns the minimum unchanged (J_A) according to the fixed deterministic selection rule.

**Evidence for ATT-01 through ATT-05:** Bellman Candidate, Filtered Candidate, Warm Start, NLP Solution, and Best-found Response artifacts with complete lineage.

**Mandatory failure:** a Bellman candidate is returned or consumed as the final Attacker response.

## 5. Defender and Stackelberg criteria

### DEF-01 — Continuous decision

**Pass condition:** (z_{mathrm{sensor}}) remains a continuous outer decision. Sampling for initialization or evaluation does not redefine the feasible strategy as a finite set.

### DEF-02 — Nested black-box evaluation

**Pass condition:** each uncached Defender evaluation executes the complete Bellman → filter → warm-start → multi-start NLP → selection sequence. Cache reuse requires identical full identity.

### DEF-03 — Refined-response exclusivity

**Pass condition:** the Defender Evaluation interface accepts only a validated Best-found Attacker Response.

### DEF-04 — Algorithm independence

**Pass condition:** the public Defender optimizer interface is independent of any named optimization algorithm.

### GAME-01 — Final solution

**Pass condition:** the Final Stackelberg Solution contains the returned continuous Defender position, associated refined Attacker strategy, switching point, trajectory/profiles, both objective breakdowns, final metrics, validation, termination, and complete lineage.

**Evidence:** Defender Optimization and Final Stackelberg Solution bundles plus an objective/identity reconstruction report.

## 6. Modularity criteria

### MOD-01 — Independent components

Every computational responsibility exists behind a reusable module interface.

### MOD-02 — Single responsibility

Geometry does not optimize; optimization does not plot or write files; validation does not repair; export does not recompute; visualization does not compute.

### MOD-03 — Explicit data flow

All required inputs arrive through function arguments or typed dependency objects. No scientific result depends on hidden notebook variables or mutable globals.

### MOD-04 — Independent testability

Every module can be tested with explicit fixtures or dependency artifacts without executing unrelated future phases.

### MOD-05 — Notebook thinness

Notebook cells call reusable functions and contain no long algorithm implementation.

**Evidence:** module/interface inventory, dependency audit, test inventory, and notebook-cell responsibility review.

## 7. Computational-phase criteria

For each computational phase:

1. explicit input schemas are validated before computation;
2. the single assigned computation runs;
3. output shape, units, physical constraints, and internal consistency are validated;
4. the phase returns primary result, validation, metadata, status, and export bundle;
5. no plotting occurs;
6. unsuccessful required validation prevents authoritative-success export;
7. diagnostics identify invalid inputs, states, candidates, or solver outcomes.

**Evidence:** one phase acceptance record per computational phase, including positive and expected-failure tests.

## 8. Required optimization validation

Every Bellman candidate, refined NLP solution, Best-found Attacker Response, Defender Evaluation, and final solution shall carry all applicable checks:

| Validation | Pass condition |
|---|---|
| Terrain clearance | trajectory maintains configured clearance; no penetration beyond tolerance |
| LOS feasibility | switching point and LOS-dependent constraints satisfy the declared geometry contract |
| Goal convergence | final position reaches ((z_{mathrm{goal}},0)) within configured tolerance |
| Dynamic residual | transcription/integration residual is finite and within configured tolerance |
| Switching consistency | refined switching point equals the glide initial state within tolerance |
| Objective consistency | stored total equals recomputed normalized weighted components within tolerance |
| Solver status | termination/status is explicitly classified and acceptable for selection |

Additional state, control, time, detection-probability, and bound checks remain mandatory where applicable.

Tolerances are centralized configuration values with units and provenance. Validation shall not silently clip or repair the object being checked.

## 9. Export and import criteria

### DATA-01 — Phase-owned export bundles

Every important computational result supplies a complete export bundle.

### DATA-02 — JSON and NPZ separation

JSON contains metadata/scalars/references; large multidimensional arrays reside in referenced NPZ payloads.

### DATA-03 — Required bundles

Geometry, Projected Cost, Projected Cost-to-Go, Bellman Candidate, NLP Solution, and Stackelberg Solution bundles all exist and pass schema validation.

### DATA-04 — No regeneration

Later phases reference earlier results and do not regenerate them for export or plotting.

### DATA-05 — Import validation

Import verifies schema, files, keys, dtype, shapes, axes, units, IDs, integrity, configuration, run, and dependency compatibility.

### DATA-06 — Plot reproducibility

A clean visualization-only execution using persisted JSON/NPZ files recreates all required figures without loading computational modules.

**Evidence:** bundle validation report, dependency graph, and clean visualization-only run report.

## 10. Visualization criteria

### VIS-01 — Imported data only

Every numerical coordinate, heatmap, path, mask, and marker originates from imported bundles.

### VIS-02 — No computational calls

Visualization invokes no geometry constructor, cost-map generator, Bellman solver, CasADi NLP, response selector, or Defender optimizer.

### VIS-03 — Required figures

All five figures exist:

1. Geometry;
2. Projected Cost Map;
3. Projected Cost-to-Go Map;
4. Projected Cost-to-Go with all Bellman candidates and all NLP paths;
5. Final Stackelberg Solution.

### VIS-04 — Terrain rendering

In every applicable figure, terrain:

- has highest z-order;
- is drawn last;
- is filled solid white;
- hides heatmaps and paths beneath it;
- retains a visible outline.

### VIS-05 — Complete presentation

Figures have title, labeled axes and units, applicable colorbar, legend, consistent colormap, line styles, and marker styles. Bellman and NLP paths are visually distinct.

**Evidence:** Visualization Manifests, source bundle IDs, automated artist/style inspection where practical, and rendered figure review.

## 11. Reproducibility criteria

### REP-01 — Fixed seeds

Every randomized or multi-start initializer accepts and exports an explicit seed or deterministic seed stream.

### REP-02 — Repeatability

Two clean executions with identical explicit inputs, artifacts, configuration, software contract, and seeds produce identical scientific outputs.

### REP-03 — Order independence

Outputs do not depend on hidden notebook history, unordered-container traversal, stale caches, or undeclared parallel scheduling.

### REP-04 — Provenance

Every authoritative bundle identifies configuration, dependencies, schema, solver settings/tolerances, seeds, and required software/backend provenance.

**Evidence:** repeat-run comparison report with declared equality or configured numerical-equivalence rules.

## 12. Backward compatibility criteria

Future work shall extend existing modules and schemas without:

- breaking public call contracts;
- renaming required exported objects or fields;
- changing existing field meaning;
- silently changing units, axes, normalization, or objective semantics;
- reordering fixed notebook phases.

An authorized incompatible change requires a version increment, reason, migration path, affected-phase inventory, and updated validation evidence.

## 13. Extensibility criteria

The architecture shall admit:

- multiple sensors;
- multiple attackers;
- moving sensors;
- moving targets;
- higher-order aircraft dynamics;
- additional detection models.

Acceptance requires an interface-extension analysis showing that each capability can be introduced through compatible typed collections, replaceable model contracts, or versioned optional fields without changing the 15-phase structure or the Defender → Attacker nesting.

This criterion does not require all extensions to be implemented in the initial system.

## 14. Phase completion gate

A notebook phase is complete only when:

1. its required module/interface exists;
2. its public interface is documented;
3. required positive behavior is implemented;
4. invalid-input and failure behavior is tested;
5. internal validation passes;
6. its result envelope is complete;
7. its export bundle is schema-valid;
8. applicable reproducibility checks pass;
9. dependency and backward-compatibility audits pass;
10. no prohibited responsibility is present;
11. acceptance evidence is recorded.

An upstream phase failure invalidates downstream acceptance evidence that depends on that failed result.

## 15. Final implementation checklist

All boxes remain unchecked until implementation evidence passes the corresponding criteria.

- [ ] Modular architecture implemented — MOD-01 through MOD-05
- [ ] Nested Stackelberg structure implemented — MATH-01, DEF-02, GAME-01
- [ ] Continuous Defender optimization implemented — DEF-01 through DEF-04
- [ ] Multi-start Bellman implemented — ATT-01
- [ ] Candidate filtering implemented — ATT-02
- [ ] Bellman-to-NLP interface implemented — ATT-03
- [ ] Multi-start CasADi NLP implemented — ATT-04
- [ ] Best-found Attacker Response implemented — ATT-05
- [ ] Defender evaluation implemented — MATH-03, DEF-03
- [ ] Final Stackelberg solver implemented — GAME-01
- [ ] JSON export implemented — DATA-01 through DATA-03
- [ ] NPZ export implemented — DATA-02 through DATA-03
- [ ] Visualization separated from computation — VIS-01 through VIS-02
- [ ] Five required figures generated — VIS-03 through VIS-05
- [ ] Validation reports generated — Section 8 and phase-specific checks
- [ ] Public interfaces documented — MOD-03 and Phase 0–5 contract
- [ ] Configuration centralized — Phase 0–4 configuration policy
- [ ] Results reproducible — REP-01 through REP-04
- [ ] Backward compatibility demonstrated — Section 12
- [ ] Extension analysis completed — Section 13

## 16. Phase 0 completion

Phase 0 is complete when the governing architecture, mathematics, notebook organization, coding standards, module interfaces, data/visualization policy, and this acceptance specification are all present, mutually consistent, and contain no algorithm implementation.

Phase 0 completion authorizes later implementation work; it does not satisfy or pre-check any implementation checklist item.

This Phase 0–7 document is the final acceptance authority for the complete notebook.
