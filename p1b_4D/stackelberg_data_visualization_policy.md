# Hierarchical Stackelberg Security Problem

## Phase 0–6: Data Export, Import, and Visualization Policy

**Status:** mandatory data and visualization contract. This document contains no computational or plotting implementation.

This policy preserves the required pipeline:

**Computation → Validation → Export → Import → Visualization**

No stage may bypass, reverse, or collapse this sequence.

## 1. Separation of responsibilities

Numerical computation, validation, persistent storage, import, and visualization are independent responsibilities.

- Computational modules return numerical results and never generate figures.
- Validation modules evaluate completed results and never plot or silently repair them.
- Export modules serialize validated results and never recompute them.
- Import modules load and validate persisted data and never run scientific computation.
- Visualization modules consume imported data and never invoke geometry construction, cost-map generation, Bellman, CasADi NLP, response selection, or Defender optimization.

Plot convenience is not a valid reason to recompute missing data. If a required field is absent or invalid, visualization shall fail with an informative diagnostic.

## 2. Export ownership

Every computational phase owns the export contract for its result:

- it produces a complete in-memory export bundle;
- it identifies all JSON metadata/scalars and NPZ arrays required to preserve the result;
- it provides validation and provenance;
- later computational phases consume the returned result rather than regenerating it.

Fixed notebook Phase 14 is the sole serialization phase. It writes each phase-owned export bundle without changing its content or meaning. This preserves both requirements: every computational phase exports its own result contract, while file writing remains isolated from computation.

Once a validated earlier result has been exported, a later phase shall reference or import it. A later phase shall not regenerate that result merely for storage or visualization.

## 3. Bundle structure

Every persisted bundle consists of:

1. one authoritative JSON manifest;
2. zero or more referenced NPZ payloads when numerical arrays are present.

The JSON manifest and its NPZ payloads form one indivisible logical bundle. A bundle is incomplete if a required referenced payload is missing, mismatched, or invalid.

Each bundle shall expose:

- bundle ID and bundle type;
- schema name and schema version;
- run ID and scenario ID;
- producer phase and module;
- configuration identity;
- dependency bundle and artifact IDs;
- validation status and metrics;
- status and diagnostics;
- JSON/NPZ relative paths;
- payload references and integrity information;
- units, coordinate conventions, shapes, dimensions, and axis order;
- objective/normalization provenance where applicable;
- deterministic seed and solver provenance where applicable.

## 4. JSON policy

JSON stores descriptive, scalar, relational, and summary information. It shall contain, as applicable:

- configuration and configuration references;
- metadata and provenance;
- dimensions, shapes, axis names, and axis order;
- units and coordinate conventions;
- solver names, versions, options, statuses, termination reasons, and tolerances;
- validation results and pass/fail summaries;
- performance metrics;
- objective values and component breakdowns;
- summary statistics;
- candidate, solution, parent, and dependency identifiers;
- references to NPZ files and named arrays;
- relative paths and artifact locations;
- integrity identifiers or checksums;
- warnings and diagnostic messages.

Large multidimensional arrays shall never be embedded in JSON. Small scalar collections may remain in JSON only when they are genuinely metadata or summaries rather than numerical payloads.

JSON numbers shall not be used to conceal non-finite values. The representation of invalid, unavailable, or non-finite data shall be explicit and schema-defined.

## 5. NPZ policy

NPZ stores numerical payloads, including:

- large arrays;
- multidimensional cost maps;
- 4D stage costs;
- projected cost maps;
- cost-to-go maps;
- masks;
- grid axes when stored numerically;
- trajectories;
- speed and gamma profiles;
- control profiles;
- heatmap values;
- Bellman candidate arrays;
- NLP solution arrays;
- refined and final path data.

Every NPZ array shall have a stable key. Its JSON manifest entry shall declare:

- payload path;
- array key;
- dtype;
- shape;
- axis order;
- unit;
- semantic description;
- applicable mask and invalid-value convention;
- integrity identifier when required.

Object arrays and pickled payloads are forbidden unless explicitly authorized by a future schema change. Numerical arrays shall be loadable without executing arbitrary code.

## 6. Path and reference policy

Paths stored in JSON shall be portable relative paths rooted at the bundle or declared artifact root. Machine-specific absolute paths shall not be authoritative bundle references.

Import shall resolve a path only within the declared artifact root unless an explicit trusted external-artifact policy is later established. References shall identify both the target file and the expected artifact/bundle identity.

Renaming or moving a complete bundle is permitted only when its internal relative references remain valid or are migrated through an explicit manifest update.

## 7. Export eligibility and atomicity

A result is eligible for authoritative export only when:

- its computational status is successful;
- all mandatory internal validations pass;
- its schema is known and complete;
- dependency and configuration identities are available;
- required arrays and metadata agree in shape, axes, units, and meaning.

Failed attempts may be persisted as diagnostic records, but they shall be marked non-authoritative and unsuccessful. They shall not be exported under a successful result identity.

An exporter shall avoid leaving a manifest that claims missing or partially written payloads are complete. Bundle completion status is written only after all required payloads and integrity information are available.

## 8. Required export bundles

### 8.1 Geometry Bundle

Contains sufficient imported data to draw and audit:

- terrain coordinates and fill boundary;
- terrain outline;
- LOS zone;
- occlusion zone;
- LOS tangent;
- sensor position;
- goal position;
- coverage area and geometry metadata;
- coordinate bounds, units, masks, and validation.

It references the originating terrain and sensor-geometry results.

### 8.2 Projected Cost Bundle

Contains:

- projected local cost heatmap;
- projected local speed;
- projected local gamma;
- axes and masks;
- projection rule and metadata;
- source 4D stage-cost artifact ID;
- mandatory visualization-only marker;
- compatible Geometry Bundle reference.

It is never an input to Bellman or NLP.

### 8.3 Projected Cost-to-Go Bundle

Contains:

- projected cost-to-go heatmap;
- axes, masks, and projection metadata;
- source authoritative higher-dimensional result ID;
- compatible Geometry Bundle reference;
- objective and normalization identifiers;
- visualization-only marker for the projected data.

The projected map is a visualization artifact and does not define the Bellman policy.

### 8.4 Bellman Candidate Bundle

Contains every Bellman candidate, not only Top-(K):

- switching points;
- position trajectories;
- speed and gamma profiles;
- objective totals and components;
- topology and start identifiers;
- candidate validation and status;
- filtering/ranking references when available;
- axes, units, lengths, and array indexing metadata.

Rejected or invalid candidates remain distinguishable through explicit status. Paths shall not be silently removed from the all-candidate visualization payload.

### 8.5 NLP Solution Bundle

Contains every attempted refined NLP solution:

- refined switching points;
- refined trajectories;
- refined speed and gamma profiles;
- controls;
- objective totals and components;
- solver status and termination information;
- constraint and dynamic residuals;
- validation reports;
- Bellman candidate and warm-start parent IDs;
- explicit selected Best-found Attacker Response ID.

Failed NLP attempts remain identifiable and shall not be represented as feasible trajectories.

### 8.6 Stackelberg Solution Bundle

Contains:

- final/best-found continuous Defender position;
- corresponding Best-found Attacker Response;
- refined switching point and trajectory;
- refined speed, gamma, and control profiles;
- mission PoD and LOS coverage;
- normalized objective components and totals;
- final feasibility and validation summaries;
- outer and inner solver status/termination metadata;
- complete lineage to geometry, projected maps, Bellman candidates, and NLP solutions.

This bundle provides all data specific to the final Stackelberg figure without rerunning any optimizer.

## 9. Import contract

An importer accepts a JSON manifest path or bundle identity and returns imported data plus validation, metadata, and status. It shall:

1. parse the JSON without scientific recomputation;
2. validate schema name and supported version;
3. resolve declared relative NPZ paths;
4. verify required files, keys, shapes, dtypes, axes, units, IDs, and integrity information;
5. reject missing, incompatible, stale, or cross-run payloads;
6. load only the arrays required by the requesting visualization when practical;
7. return informative diagnostics on failure.

Import shall not:

- construct terrain or LOS geometry;
- project 4D data;
- generate a cost or cost-to-go map;
- run Bellman;
- run CasADi NLP;
- select an Attacker response;
- evaluate or optimize the Defender;
- alter persisted numerical results.

## 10. Visualization input boundary

Phase 15 visualization accepts successful imported bundles only. Every plotted coordinate, mask, heatmap, path, marker, scalar annotation, and selection identity shall originate from imported JSON/NPZ data.

Formatting transformations that do not create new scientific results are permitted, such as assigning colors, line styles, labels, display limits, and z-order. Visualization shall not interpolate, smooth, optimize, fuse, normalize, project, or otherwise derive authoritative numerical data.

If a figure requires a quantity that is not exported, the responsible computational phase and bundle schema must be extended first. The plotting phase shall not calculate the missing quantity.

## 11. Figure reproducibility

Every figure shall be reproducible from its imported bundle set and plotting configuration without:

- rerunning computation;
- rebuilding geometry;
- rerunning Bellman or NLP;
- reevaluating objectives;
- rerunning Defender optimization;
- relying on hidden notebook variables.

The visualization manifest shall record:

- figure ID and required figure type;
- source bundle IDs and schema versions;
- plot configuration identity;
- selected colormap, styles, markers, and display limits;
- output location and format;
- rendering status and diagnostics.

## 12. Common appearance contract

Every required figure shall contain:

- an informative title;
- labeled horizontal and vertical axes with units;
- a colorbar when a scalar field or heatmap is shown;
- a legend for plotted semantic elements;
- consistent colormap use across comparable heatmaps;
- consistent line styles across semantic trajectory classes;
- consistent marker styles for sensor, goal, switching point, and selected solutions.

Style meaning shall be defined in `plot_config` and applied consistently. Bellman and NLP trajectories shall use different colors, line styles, or both.

## 13. Terrain drawing contract

In every applicable figure, terrain shall:

1. have the highest z-order;
2. be drawn after all elements that it must occlude;
3. be filled solid white;
4. hide all heatmap and trajectory content beneath the terrain surface;
5. retain a visible terrain outline above the white fill.

The exported Geometry Bundle shall provide the terrain polygon/fill boundary and outline explicitly. Visualization shall not reconstruct them from a terrain function.

Terrain-policy validation shall be included in the visualization status.

## 14. Required Figure 1 — Geometry

Imported sources:

- Geometry Bundle.

Required visible elements:

- terrain fill and outline;
- LOS zone;
- occlusion zone;
- LOS tangent;
- sensor;
- goal.

A colorbar is required only if the imported geometry bundle includes a plotted scalar field. All other common appearance and terrain rules apply.

## 15. Required Figure 2 — Projected Cost

Imported sources:

- Projected Cost Bundle;
- referenced compatible Geometry Bundle.

Required visible elements:

- projected cost heatmap;
- terrain fill and outline;
- LOS zone;
- occlusion zone;
- LOS tangent;
- sensor;
- goal;
- heatmap colorbar.

Projected local speed and gamma may be shown only when explicitly requested by the figure specification and already present in the imported bundle.

## 16. Required Figure 3 — Projected Cost-to-Go

Imported sources:

- Projected Cost-to-Go Bundle;
- referenced compatible Geometry Bundle.

Required visible elements:

- projected cost-to-go heatmap;
- terrain fill and outline;
- LOS zone;
- occlusion zone;
- LOS tangent;
- sensor;
- goal;
- heatmap colorbar.

## 17. Required Figure 4 — All Bellman and NLP Paths

Imported sources:

- Projected Cost-to-Go Bundle;
- Bellman Candidate Bundle;
- NLP Solution Bundle;
- referenced compatible Geometry Bundle.

Required visible elements:

- projected cost-to-go heatmap;
- every exported Bellman candidate path;
- every exported Attacker NLP path, with status-aware styling where needed;
- terrain fill and outline;
- LOS zone;
- occlusion zone;
- LOS tangent;
- sensor;
- goal;
- colorbar and legend.

Bellman and NLP paths shall use visibly different colors, line styles, or both. The figure shall not silently plot only Top-(K), only feasible NLP paths, or only the selected response when the required source bundle contains additional paths. Any intentionally excluded diagnostic path must be governed by explicit plot configuration and disclosed in the figure metadata.

## 18. Required Figure 5 — Final Stackelberg Result

Imported sources:

- Stackelberg Solution Bundle;
- Projected Cost-to-Go Bundle;
- referenced compatible Geometry Bundle.

Required visible elements:

- projected cost-to-go heatmap;
- final/best-found Defender position;
- corresponding Best-found Attacker path;
- refined switching point;
- terrain fill and outline;
- LOS zone;
- occlusion zone;
- LOS tangent;
- sensor;
- goal;
- colorbar and legend.

Numerical results shall use best-found terminology unless global optimality has been independently established. A presentation label such as “optimal” shall not override the governing result semantics.

## 19. Cross-bundle compatibility

Bundles combined in one figure shall agree on:

- run and scenario identity;
- configuration and terrain identity;
- Defender position where the geometry is decision-dependent;
- coordinate system and units;
- axes and domain;
- objective and normalization version where applicable;
- source/dependency lineage.

An importer or visualization validator shall reject incompatible bundles rather than align, rescale, or reinterpret them silently.

## 20. Visualization acceptance checklist

A figure is acceptable only if:

1. all numerical content came from imported JSON/NPZ bundles;
2. source bundles passed schema and compatibility checks;
3. no geometry, cost, Bellman, NLP, response selection, or optimization was run;
4. required elements for the figure type are present;
5. terrain was drawn last, white-filled, highest-z-order, and outlined;
6. heatmaps do not appear beneath terrain;
7. title, axes, units, colorbar when applicable, and legend are present;
8. colors, lines, and markers follow shared plot configuration;
9. Bellman and NLP paths are visually distinct;
10. the figure can be recreated without computational phases.

This Phase 0–6 policy is the authoritative contract for persistent data exchange and all project visualizations.
