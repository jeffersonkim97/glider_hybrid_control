# Hierarchical Stackelberg Security Problem

## Phase 0–4: Coding Standards and Implementation Rules

**Status:** mandatory implementation contract. This document contains no algorithm implementation or optimization code.

These standards apply to every future module, function, notebook execution cell, validation routine, exporter, and visualization component in the project. Earlier architecture, mathematics, public interfaces, output schemas, and solver hierarchy remain authoritative.

## 1. Priority order

All design and implementation decisions shall use this priority order:

1. mathematical correctness;
2. physical realism and consistency;
3. readability;
4. modularity;
5. maintainability;
6. computational efficiency.

Performance improvements are acceptable only when they preserve the mathematical formulation, physical constraints, numerical meaning, public interfaces, validation behavior, and reproducibility contract. A faster result that changes the defined problem is incorrect.

## 2. Implementation style

Future implementation shall be explicit, readable, documented, and deterministic.

- Prefer clear intermediate names and visible steps over compact expressions that obscure mathematical meaning.
- Avoid one-line implementations when they reduce readability or complicate validation.
- Introduce abstractions only when they remove genuine duplication, enforce a stable interface, or isolate a responsibility.
- Do not create abstraction layers that merely rename an operation or conceal important physical or numerical assumptions.
- Keep notebook execution cells short. Cells orchestrate reusable functions and display concise status information; they do not contain long algorithm bodies.
- Preserve the terminology defined by the mathematical formulation, including **Best-found Attacker Response**.

## 3. Single-responsibility functions

Every function shall perform one coherent task.

- A function should normally remain below approximately 100–150 lines, excluding its docstring.
- A longer function requires a clear justification based on cohesion and readability.
- Functions that validate, compute, select, export, or plot shall remain separate.
- Large workflows shall be decomposed into reusable functions connected through explicit return values.
- Decomposition shall not fragment a single readable mathematical operation into unnecessary wrappers.

Line count is a review signal, not permission to combine unrelated responsibilities below the threshold.

## 4. Function interfaces

Every function shall have:

- explicit, clearly named inputs;
- explicit return values on every successful path;
- documented input shapes, units, coordinate conventions, and accepted types;
- documented output shapes, units, meanings, and types;
- documented assumptions and admissible domains;
- input and output validation proportional to its responsibility;
- no dependence on hidden notebook state whenever avoidable.

Functions shall receive configuration and dependency artifacts explicitly. They shall not obtain essential inputs from execution order, ambient globals, previously displayed notebook values, or undeclared filesystem state.

When a function can fail in an expected way, its contract shall define how failure is represented. Unexpected invariant violations shall produce informative exceptions rather than plausible-looking fallback values.

## 5. Public-function documentation

Every public function docstring shall contain these sections:

1. **Purpose** — the single operation performed.
2. **Inputs** — names, types, shapes, units, coordinate conventions, and valid ranges.
3. **Outputs** — names, types, shapes, units, and semantic meaning.
4. **Assumptions** — mathematical, physical, numerical, and ordering assumptions.
5. **Notes** — side effects, determinism, solver caveats, failure behavior, and provenance requirements.

Documentation shall explain why non-obvious physical or numerical choices are valid. It shall not merely restate syntax.

## 6. Configuration policy

Hard-coded configurable numerical constants are forbidden in computational logic.

Parameters shall be organized by responsibility in validated configuration mappings:

- `environment_config`;
- `vehicle_config`;
- `sensor_config`;
- `bellman_config`;
- `nlp_config`;
- `defender_config`;
- `plot_config`;
- `io_config`.

Each configuration entry shall have:

- one canonical name;
- a documented type and unit;
- a documented valid range or admissible set;
- a defined default only when a scientifically justified default exists;
- one authoritative source during a run.

Physical constants may use documented `UPPER_CASE` names when they are genuinely invariant. If a value may differ by scenario, experiment, model choice, resolution, solver, tolerance, or presentation requirement, it belongs in configuration.

Configuration shall be validated before downstream computation and treated as immutable for the duration of a run. Derived values shall be distinguishable from user-specified values and included in provenance.

## 7. Global state

Mutable global variables shall be avoided.

If global state is unavoidable, its purpose, owner, lifetime, mutation points, reset behavior, and reproducibility implications shall be documented. A global shall not carry a result from one notebook phase into another when an explicit artifact or function argument can do so.

Caches shall be keyed by all inputs that affect their contents. Cache use shall not change numerical results or conceal stale configuration.

## 8. CasADi and NumPy boundary

CasADi symbolic construction and NumPy numerical evaluation shall remain isolated behind explicit interfaces.

- A symbolic function shall construct or transform CasADi expressions using CasADi-compatible operations.
- A numerical function shall evaluate concrete arrays using NumPy or another explicitly declared numerical backend.
- Conversion between symbolic and numerical representations shall occur only at named boundary functions.
- Symbolic expressions shall not be silently converted to NumPy arrays.
- NumPy operations shall not be inserted into symbolic graphs unless their values are intentionally fixed constants and that assumption is documented.
- Symbolic and numerical implementations of the same physical quantity shall share definitions, parameter names, units, and validation cases.

This separation shall be visible in function names, types, documentation, and tests.

## 9. NumPy implementation

Prefer vectorized NumPy operations when they are practical, readable, memory-safe, and mathematically equivalent.

- Avoid deeply nested Python loops when a clear vectorized formulation exists.
- Do not vectorize at the expense of excessive memory use, obscure broadcasting, changed reduction order with material numerical consequences, or loss of physical readability.
- Document array axis order and shapes at module boundaries.
- Validate broadcasting assumptions explicitly for important calculations.
- Performance-driven rewrites require equivalence validation against the readable reference behavior.

## 10. Error handling and diagnostics

Every important function shall validate:

- required inputs and configuration keys;
- types, dimensions, shapes, and axis order;
- finiteness and admissible numeric ranges;
- units or unit contracts where represented;
- physical state validity;
- dependency artifact schema and provenance compatibility.

Failures shall never be ignored silently.

Diagnostics shall identify:

- the function or phase that failed;
- the invalid quantity;
- the expected condition;
- the observed value, shape, or status when safe and useful;
- the affected candidate or run identifier;
- whether partial output exists and whether it is invalid.

Broad exception suppression is forbidden. A phase shall not mark itself successful when a required validation or solver step failed.

## 11. Mandatory optimization validation

Every optimization stage shall validate its primary result before it becomes eligible for export or downstream selection.

Applicable checks include:

- terrain penetration;
- initial and terminal boundary conditions;
- goal convergence at ((z_{\mathrm{goal}},0));
- LOS and switching-point feasibility;
- dynamic residuals;
- state and control bounds;
- time monotonicity and nonnegative phase durations;
- objective-component finiteness;
- recomputed objective consistency;
- consistency of Bellman and NLP weights and normalization;
- NLP feasibility and solver termination status;
- parent-candidate and warm-start provenance.

Validation shall return structured metrics and a clear pass/fail status. Tolerances shall come from validated configuration, carry units where applicable, and be exported. Validation shall not silently modify the candidate it evaluates.

## 12. Naming conventions

Use the following conventions consistently:

- functions: `snake_case`;
- variables and parameters: `snake_case`;
- classes and typed records: `PascalCase`;
- genuine constants: `UPPER_CASE`;
- configurable values: responsibility-specific configuration mappings.

Names shall express physical meaning and phase role. Include units in names only when ambiguity would otherwise remain or when an interface intentionally carries a fixed unit. Avoid unexplained abbreviations, generic temporary names in public interfaces, and multiple names for the same mathematical object.

## 13. Side effects

Computational functions shall return results and avoid printing, plotting, saving files, changing directories, mutating configuration, or modifying global state.

A side effect is permitted only when it is the function's primary documented responsibility:

- status/reporting functions may print or format diagnostics;
- Phase 14 export functions may create files;
- Phase 15 visualization functions may create figures.

Even dedicated side-effect functions shall receive explicit inputs and report their outputs, targets, or status.

## 14. Plotting boundary

No computational, geometry, validation, Bellman, NLP, selection, or Defender-optimization function shall generate a figure.

Plotting occurs only in the dedicated visualization phase and shall:

- load authoritative exported artifacts;
- avoid recomputing numerical quantities;
- avoid triggering Bellman, NLP, or Defender optimization;
- preserve the required terrain masking and z-order rules;
- identify source artifact and run IDs.

Display convenience shall never become a dependency of computational correctness.

## 15. Export boundary

Computational functions return typed results. Dedicated Phase 14 export functions serialize them.

- Optimization and file writing shall not occur in the same function.
- Exporters shall not recompute, refine, select, or repair results.
- Exporters shall validate target schema compatibility before writing.
- JSON shall contain metadata, identifiers, scalar results, validation metrics, and NPZ manifests.
- NPZ shall contain numerical arrays referenced by the JSON manifest.
- Failed or invalid results shall not be exported as successful authoritative results.

## 16. Determinism and reproducibility

Randomized and multi-start procedures shall accept explicit fixed seeds.

Identical explicit inputs, dependency artifacts, configuration, software contract, and seeds shall produce identical outputs. Implementations shall export:

- seed values and seed-stream identifiers;
- relevant solver settings and tolerances;
- configuration and schema versions;
- dependency artifact identifiers;
- software/backend provenance required to interpret reproducibility.

Iteration order, unordered containers, parallel scheduling, hidden notebook state, and cache history shall not silently determine scientific results.

## 17. Code reuse

Shared algorithms and transformations shall be implemented once and reused through stable interfaces.

- Do not copy detection, normalization, objective, dynamics, validation, or geometry logic between Bellman and NLP paths.
- Bellman and NLP shall call the same authoritative Attacker objective definition.
- Symbolic and numerical variants may have separate backend functions, but they shall share the same mathematical contract and validation suite.
- Duplication introduced temporarily during a verified migration shall be marked, bounded, and removed before the phase is accepted.

Reuse shall not merge responsibilities that the architecture requires to remain separate.

## 18. Backward compatibility

Future phases shall reuse established:

- public interfaces;
- field names and meanings;
- result schemas and exported outputs;
- module boundaries;
- notebook phase order;
- Stackelberg and Bellman-to-NLP hierarchy.

An explicitly authorized incompatible change requires:

1. a stated mathematical or architectural reason;
2. a schema or interface version change;
3. a migration path for existing artifacts or callers;
4. updated validation evidence;
5. documentation of affected phases.

Silent reinterpretation of an existing field or output is forbidden.

## 19. Review checklist

A future implementation is not acceptable unless reviewers can answer yes to all applicable questions:

1. Does it preserve the mathematical and physical formulation?
2. Does each function have one responsibility and an explicit interface?
3. Are configurable constants stored in the correct validated configuration?
4. Is hidden mutable state absent or fully documented?
5. Are CasADi symbolic and NumPy numerical operations separated?
6. Are shapes, axes, units, bounds, and invalid states validated?
7. Are optimization results internally validated before export?
8. Are failures explicit and informative?
9. Are naming and public docstring conventions satisfied?
10. Are computation, validation, export, and plotting separated?
11. Are randomized paths deterministic under fixed seeds?
12. Is shared logic reused rather than duplicated?
13. Are previous interfaces and outputs backward-compatible?
14. Has any optimization for performance been proven behaviorally equivalent?

## 20. Governing rule

When implementation trade-offs occur, the controlling order is:

**correctness → physical consistency → modularity → maintainability → performance**

Readability remains mandatory throughout that ordering. No future phase may waive these standards merely because an implementation is experimental, notebook-based, computationally expensive, or intended as an intermediate step.
