# Hierarchical Stackelberg Security Problem

## Phase 0–5: Authoritative Module Interface Specification

**Status:** interface contract only. This document contains no algorithms, optimization code, or implementation.

> **SUPERSEDED (2026-07-23):** Following advisor review, Phase 9 (Bellman to NLP Interface) and Phase 10 (Attacker CasADi NLP) below — `BellmanWarmStartSet` and `RefinedNlpSolutionSet` — are **deprecated and disconnected** from the authoritative pipeline. The Attacker best response is now produced directly from `BellmanCandidateSet` by `p1b_4D.bellman.select_authoritative_bellman_response`, returning an `AuthoritativeBellmanAttackerResponse`. `attacker_nlp.py`/`candidate_filtering.py` remain only as an offline experimental comparison, never imported by `stackelberg_solver.py` or the notebook.

This specification defines the data exchanged by all modules. It preserves the 15-phase notebook architecture established in Phase 0–3. Names, meanings, units, provenance, and dependency direction are part of the public contract.

## 1. Interface principles

Every module is an independent component that:

1. receives all required information through explicit function arguments;
2. performs its single assigned responsibility;
3. returns one explicit result envelope;
4. does not depend on hidden notebook variables, implicit global state, or incidental cell execution history;
5. depends only on Configuration or declared outputs of earlier notebook phases.

Passing a returned object explicitly is mandatory. Reading a variable merely because a previous notebook cell created it is not a module interface.

## 2. Universal result envelope

Every public module returns a result envelope with exactly these top-level fields:

| Field | Meaning |
|---|---|
| `primary_result` | Typed authoritative output of the module's single responsibility |
| `validation` | Structured validation information and pass/fail summary |
| `metadata` | Identity, schema, units, dimensions, configuration, dependency, and provenance data |
| `status` | Explicit success/failure state and diagnostic information |

### 2.1 Status contract

`status` contains:

- `success`: Boolean success flag;
- `code`: stable machine-readable status code;
- `message`: informative human-readable diagnostic;
- `warnings`: ordered collection of non-fatal diagnostics;
- `failed_checks`: identifiers of failed required checks.

A failed result shall never masquerade as a successful result. Downstream computation shall reject an unsuccessful required dependency unless its interface explicitly defines a recovery path. No recovery path may silently change the mathematical problem.

### 2.2 Validation contract

`validation` contains:

- `passed`: aggregate Boolean;
- `checks`: named check results;
- `metrics`: named numerical or categorical validation metrics;
- `tolerances`: the configured tolerances used;
- `summary`: concise diagnostic summary.

Validation does not modify `primary_result`.

### 2.3 Metadata contract

`metadata` contains at least:

- `schema_name` and `schema_version`;
- `run_id` and `scenario_id`;
- `producer_phase` and `producer_module`;
- `config_id`;
- `dependency_ids`;
- `units` and `coordinate_convention`;
- array shapes, dimensions, and axis order where applicable;
- random seed or deterministic seed-stream identity where applicable;
- creation and software provenance needed to interpret the result.

Metadata describes a result; it does not carry undeclared computational inputs.

## 3. Canonical phase mapping

The Phase 0–5 requested outputs map to the fixed Phase 0–3 notebook organization as follows:

| Fixed notebook phase | Public result object |
|---:|---|
| 1 — Configuration | `ConfigurationBundle` |
| 2 — Terrain Model | `TerrainModelResult` |
| 3 — Sensor Geometry | `SensorGeometryResult` |
| 4 — CasADi Symbolic Detection Model | `SymbolicDetectionResult` |
| 5 — 4D Stage Cost Construction | `StageCost4DResult` |
| 6 — 2D Projection | `Projection2DResult` |
| 7 — Multi-start Bellman | `BellmanCandidateSet` |
| 8 — Bellman Candidate Filtering | `FilteredCandidateSet` |
| 9 — Bellman to NLP Interface | `BellmanWarmStartSet` |
| 10 — Attacker CasADi NLP | `RefinedNlpSolutionSet` |
| 11 — Attacker Best-found Response | `BestFoundAttackerResponse` |
| 12 — Continuous Defender Optimization | `DefenderOptimizationResult`, containing per-decision `DefenderEvaluation` records |
| 13 — Stackelberg Solver | `FinalStackelbergSolution` |
| 14 — Export | `ExportManifest` |
| 15 — Visualization | `VisualizationManifest` |

The geometry outputs grouped together in the Phase 0–5 request are split between fixed notebook Phases 2 and 3 to retain the established single-responsibility architecture. Together, `TerrainModelResult` and `SensorGeometryResult` expose the complete required geometry interface.

## 4. Phase 1 — ConfigurationBundle

### Inputs

- explicit user or scenario configuration source;
- schema versions;
- optional documented overrides.

### Primary result

`ConfigurationBundle` contains:

- `environment_config`;
- `terrain_config` and terrain-model specification;
- `grid_config`;
- `vehicle_config`;
- `sensor_config`;
- `goal_config`;
- `bellman_config`;
- `nlp_config`;
- `defender_config`;
- `plot_config`;
- `io_config`;
- fixed objective weights and normalization definitions;
- coordinate, axis-order, and unit conventions;
- deterministic seed definitions.

The Phase 0–5 terms **Terrain Model**, **Grid**, **Vehicle Parameters**, **Sensor Parameters**, and **Goal Parameters** are standard configuration inputs exposed here. Computed terrain geometry is not produced until Phase 2.

### Validation

Schema completeness, type correctness, units, finite bounds, admissible ranges, cross-field consistency, required weights, grid axis definitions, and seed validity.

### Consumers

Every later phase may receive `ConfigurationBundle`. No later phase may mutate it.

## 5. Phase 2 — TerrainModelResult

### Inputs

- `ConfigurationBundle`.

### Primary result

`TerrainModelResult` contains:

- terrain geometry representation;
- terrain domain;
- terrain elevation function contract;
- terrain derivative function contract;
- evaluated terrain samples on configured grids when required;
- terrain validity mask;
- coordinate and unit declarations.

The terrain function maps horizontal position to terrain elevation. The derivative contract maps horizontal position to the corresponding terrain slope under the declared differentiability convention.

### Validation

Domain coverage, finiteness, shape consistency, derivative consistency, grid alignment, and terrain-model constraints.

### Consumers

Sensor Geometry and all later modules that require terrain queries. This module contains no LOS optimization or plotting.

## 6. Phase 3 — SensorGeometryResult

### Inputs

- `ConfigurationBundle`;
- successful `TerrainModelResult`;
- continuous Defender decision `z_sensor`.

### Primary result

`SensorGeometryResult` contains:

- sensor position and height;
- LOS geometry;
- LOS tangent;
- LOS zone;
- occlusion zone;
- LOS coverage area;
- geometry masks and domains;
- geometry-query function contracts required downstream.

Sensor height follows the authoritative mathematical formulation. Coverage area means LOS coverage area.

### Validation

Sensor placement domain, terrain relationship, LOS tangent residual, mask consistency, mutually consistent LOS/occlusion classification, nonnegative finite coverage area, and coordinate alignment.

### Consumers

Symbolic detection, 4D cost, Attacker solvers, Defender evaluation, export, and visualization through explicit arguments or exported artifacts.

## 7. Phase 4 — SymbolicDetectionResult

### Inputs

- `ConfigurationBundle`;
- `TerrainModelResult`;
- `SensorGeometryResult`;
- declared symbolic state/control signatures.

### Primary result

`SymbolicDetectionResult` contains independent CasADi symbolic function contracts for:

- acoustic detection;
- radar detection;
- radial-velocity detection;
- RCS detection;
- powered-phase detection;
- glide-phase detection;
- fused mission detection.

Each symbolic function exposes ordered input names, shapes, units, parameter dependencies, output names, output bounds, and symbolic backend metadata.

These functions remain independent from Bellman, NLP, and Defender optimization. They evaluate detection quantities but make no strategy decision.

### Validation

Symbolic signature checks, output dimensions, probability bounds, phase-allocation consistency, finite numerical spot evaluations, and agreement with authoritative reference cases.

## 8. Phase 5 — StageCost4DResult

### Inputs

- `ConfigurationBundle`;
- successful terrain and sensor geometry results;
- `SymbolicDetectionResult`;
- authoritative 4D state/control axes and dynamics interface.

### Primary result

`StageCost4DResult` contains:

- `j4d`: authoritative 4D stage-cost values or representation;
- 4D axes and axis order;
- feasible-state/action mask;
- component cost maps;
- Attacker objective weights and normalization identifiers;
- state/control-domain metadata.

### Validation

Shape and axis agreement, feasible-mask consistency, finite costs on feasible entries, invalid-entry policy, component-to-total objective reconstruction, unit consistency, and identical objective definitions for Bellman and NLP.

### Consumers

The authoritative 4D result feeds Bellman and may feed NLP evaluation through the declared mathematical contract. It also feeds the Phase 6 projection. Bellman never consumes the projected Phase 6 output.

## 9. Phase 6 — Projection2DResult

### Inputs

- `ConfigurationBundle`;
- successful `StageCost4DResult`.

### Primary result

`Projection2DResult` contains:

- projected cost;
- projected local speed;
- projected local gamma;
- 2D axes and mask;
- projection rule;
- source `StageCost4DResult` identifier;
- projection metadata;
- mandatory `visualization_only = true` marker.

### Validation

Projection-axis consistency, source-shape compatibility, valid reduction indices, projected speed/gamma alignment, mask behavior, and traceability to the source 4D artifact.

### Prohibition

`Projection2DResult` shall never define, constrain, rank, initialize, or validate a Bellman policy. It is not an accepted input type for Multi-start Bellman, Candidate Filtering, Warm Start, CasADi NLP, response selection, or Defender evaluation.

## 10. Phase 7 — BellmanCandidateSet

### Inputs

- `ConfigurationBundle`;
- terrain and sensor geometry results;
- `SymbolicDetectionResult` as required by the chosen cost interface;
- authoritative `StageCost4DResult`;
- explicit multi-start specifications.

### Primary result

`BellmanCandidateSet` contains an ordered collection of candidates. Every candidate contains:

- candidate and start identifiers;
- switching point;
- position trajectory;
- speed profile;
- gamma profile;
- Attacker objective total;
- objective component breakdown;
- topology signature;
- candidate metadata;
- candidate validation.

The set also records the unchanged Attacker objective contract and all start provenance.

### Validation

Boundary conditions, terrain penetration, LOS/switching feasibility, state/control domains, trajectory/profile alignment, objective reconstruction, and candidate status.

### Prohibition

No candidate is a final Attacker solution.

## 11. Phase 8 — FilteredCandidateSet

### Inputs

- `ConfigurationBundle`;
- successful or individually classified `BellmanCandidateSet`;
- explicit deterministic filtering, duplicate, ranking, and Top-(K) rules.

### Primary result

`FilteredCandidateSet` contains:

- filtered candidate collection;
- ordered Top-(K) candidates;
- duplicate-removal report;
- ranking information;
- accepted and rejected candidate IDs;
- rejection and exclusion reasons;
- retained objective and topology provenance.

### Validation

Top-(K) bounds, deterministic ordering, candidate identity preservation, duplicate-group consistency, ranking recomputation, and proof that objective weights were not changed.

## 12. Phase 9 — BellmanWarmStartSet

### Inputs

- `ConfigurationBundle`;
- `FilteredCandidateSet`;
- explicit NLP transcription/interface definition.

### Primary result

`BellmanWarmStartSet` contains one warm start per selected candidate, with:

- warm-start ID and parent Bellman candidate ID;
- initial switching point;
- initial trajectory;
- initial speed profile;
- initial gamma profile;
- initial control representation when required by the NLP interface;
- interpolation/transcription metadata;
- validation report.

### Validation

NLP dimension and ordering compatibility, boundary preservation, profile/mesh alignment, finite initial values, configured bounds, and parent traceability.

The interface transforms representations only. It performs no continuous refinement.

## 13. Phase 10 — RefinedNlpSolutionSet

### Inputs

- `ConfigurationBundle`;
- terrain and sensor geometry results;
- symbolic detection functions;
- dynamics, constraints, and unchanged Attacker objective contracts;
- `BellmanWarmStartSet`.

### Primary result

`RefinedNlpSolutionSet` contains one NLP solution record per attempted warm start. Every record contains:

- solution and parent warm-start IDs;
- refined switching point;
- refined position trajectory;
- refined speed profile;
- refined gamma profile;
- refined controls;
- objective total and component breakdown;
- solver status;
- constraint and dynamic residuals;
- validation report;
- refinement metadata.

Failed attempts remain represented with explicit status and diagnostics; they are not silently dropped.

### Validation

Solver termination, feasibility, terrain clearance, goal convergence, LOS constraints, dynamic residuals, bounds, time validity, objective reconstruction, and parent provenance.

## 14. Phase 11 — BestFoundAttackerResponse

### Inputs

- `ConfigurationBundle`;
- `RefinedNlpSolutionSet`;
- fixed feasible-response selection rule under the unchanged (J_A).

### Primary result

`BestFoundAttackerResponse` contains:

- selected refined solution ID;
- refined switching point;
- refined trajectory;
- refined speed and gamma profiles;
- refined controls;
- Attacker objective and component breakdown;
- feasibility and validation evidence;
- complete Bellman-to-warm-start-to-NLP provenance.

### Validation

Selected-record existence, feasibility, selection-rule recomputation, objective ordering among feasible refined records, and provenance completeness.

### Exclusivity

This is the only Attacker solution type accepted by Defender evaluation. It is a best-found response, not a claimed global optimum.

## 15. Phase 12 — DefenderEvaluation and DefenderOptimizationResult

### DefenderEvaluation inputs

- `ConfigurationBundle`;
- continuous `z_sensor`;
- corresponding `SensorGeometryResult`;
- corresponding `BestFoundAttackerResponse`.

### DefenderEvaluation primary result

Each `DefenderEvaluation` contains:

- Defender decision;
- Best-found Attacker Response ID;
- (J_D);
- LOS coverage area and normalized coverage;
- mission PoD and normalized PoD;
- Defender objective breakdown;
- validation and provenance metadata.

It shall reject Bellman candidates, warm starts, and unselected NLP solutions.

### DefenderOptimizationResult primary result

`DefenderOptimizationResult` contains:

- all requested/effective continuous Defender evaluations;
- best-found Defender decision under the outer optimizer's termination contract;
- associated `BestFoundAttackerResponse`;
- outer optimization status and diagnostics;
- convergence/termination metadata;
- black-box Attacker-solver configuration identity.

### Validation

Continuous decision bounds, full nested-solve provenance for every evaluation, Defender objective reconstruction, LOS coverage definition, refined-response exclusivity, and termination-contract consistency.

The interface does not prescribe a Defender optimization algorithm.

## 16. Phase 13 — FinalStackelbergSolution

### Inputs

- `ConfigurationBundle`;
- successful `DefenderOptimizationResult`;
- complete nested-solver provenance.

### Primary result

`FinalStackelbergSolution` contains:

- final/best-found Defender position;
- associated Best-found Attacker strategy;
- refined switching point;
- refined trajectory;
- refined speed and gamma profiles;
- final Attacker and Defender metrics;
- both objective breakdowns;
- feasibility, validation, and termination summaries;
- complete artifact lineage.

The term **optimal** may describe the formal mathematical target. Numerical outputs retain best-found/returned terminology unless global optimality is independently established.

### Validation

Cross-object identity, objective reconstruction, selected-response consistency, full nested hierarchy, configuration consistency, and provenance completeness.

## 17. Phase 14 — ExportManifest

### Inputs

- any successful, validated result envelopes designated for persistence;
- `io_config`;
- target schema registry.

### Primary result

`ExportManifest` contains:

- exported artifact IDs;
- JSON and NPZ paths or logical locations;
- schema names and versions;
- array manifests, shapes, axes, and units;
- source result IDs and dependency lineage;
- checksums or integrity identifiers;
- export status and diagnostics.

Export performs serialization only. It does not recompute, repair, optimize, select, or plot.

## 18. Phase 15 — VisualizationManifest

### Inputs

- successful `ExportManifest`;
- exported JSON/NPZ artifacts;
- `plot_config`.

### Primary result

`VisualizationManifest` contains:

- figure identifiers and output locations;
- figure type;
- source artifact IDs;
- rendering metadata;
- terrain masking/z-order validation;
- visualization status and diagnostics.

Visualization accepts exported artifacts only and performs no authoritative numerical computation.

## 19. Compatibility and extension rules

Once published, an interface shall be reused without changing the meaning of existing fields.

Future extensions may:

- append optional fields with documented defaults or absence semantics;
- introduce new schema versions;
- add new result types that compose with existing envelopes;
- generalize scalar entities to typed collections while preserving single-entity compatibility.

Future extensions shall not:

- rename or remove required fields without an explicitly authorized migration;
- change units, axis order, normalization, or semantic meaning silently;
- change a success field into a different status convention;
- make a previously explicit dependency hidden;
- allow visualization-only data into optimization interfaces;
- replace a refined response with a Bellman candidate in Defender evaluation.

Multiple sensors, multiple attackers, moving sensors, moving targets, higher-order dynamics, and new sensor models shall be added through compatible fields or versioned types without reordering the fixed notebook phases.

## 20. Interface acceptance checklist

An interface is acceptable only if:

1. all required inputs are explicit;
2. the universal result envelope is complete;
3. status and validation failures are informative and non-silent;
4. shapes, axes, units, identifiers, and provenance are defined;
5. downstream consumers are named;
6. forbidden consumers and responsibilities are enforced;
7. the object is reusable without notebook-variable lookup;
8. optional extensions cannot change existing meaning;
9. the Bellman-to-NLP-to-refined-response hierarchy is preserved;
10. projected 2D data remains visualization-only.

This Phase 0–5 specification is the authoritative public-interface contract for all future implementation phases.
