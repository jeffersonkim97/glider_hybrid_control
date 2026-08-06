# P1 Exact Straight-Segment Geometry Feasibility — 2026-07-30

## Status and purpose

P1 is complete. It replaces finite interior sampling as the authoritative
terrain/LOS admissibility test for every powered, virtual-switch, regular, and
goal-terminal physical edge used by the successor-grid follower.

The certificate is exact up to ordinary floating-point evaluation for the
implemented geometry representation:

- terrain: natural piecewise-cubic `CubicSpline`;
- LOS boundary: the stored piecewise-linear swept multi-hill boundary; and
- trajectory edge: a straight segment with exact stored endpoints.

It certifies continuous geometry feasibility of finite edges. It does not
certify a continuous-space optimum or exact continuous detection integration.

## Terrain certificate

For a forward segment from $(z_0,h_0)$ to $(z_1,h_1)$, define

$$
h_e(z)=h_0+s_e(z-z_0),
\qquad
s_e=\frac{h_1-h_0}{z_1-z_0}.
$$

On terrain spline interval $i$, let

$$
T_i(z)=a_i\xi^3+b_i\xi^2+c_i\xi+d_i,
\qquad
\xi=z-z_i.
$$

The signed terrain margin is

$$
m_T(z)=h_e(z)-T_i(z).
$$

Its interior stationary points satisfy

$$
m_T'(z)=s_e-\left(3a_i\xi^2+2b_i\xi+c_i\right)=0.
$$

Thus the global minimum over the segment is found by evaluating $m_T$ at:

1. both segment endpoints;
2. every crossed spline knot; and
3. every real root of the quadratic stationarity equation lying inside its
   spline interval and the segment.

The edge passes terrain clearance when

$$
\min_{z\in[z_0,z_1]}m_T(z)\ge-\epsilon_T,
$$

where the existing configured terrain tolerance is
$\epsilon_T=10^{-6}\ \mathrm{m}$ unless the configuration explicitly changes
it.

## LOS certificate

The stored swept LOS boundary $L(z)$ is linear between consecutive boundary
breakpoints. Since $h_e(z)$ is also linear, either signed LOS margin is linear
inside each such interval.

For glide visibility,

$$
m_V(z)=h_e(z)-L(z),
$$

and the active interval stops at the sensor downrange coordinate, matching the
finite model's post-sensor visibility rule. For powered occlusion,

$$
m_O(z)=L(z)+\epsilon_{LOS}-h_e(z).
$$

The exact minimum of either piecewise-linear margin is therefore attained at a
segment endpoint or crossed LOS breakpoint. Glide requires
$\min m_V\ge0$; powered flight requires $\min m_O\ge0$ and remains on the
pre-sensor side.

## Airspace, vertical, and terminal cases

- A rectangular airspace needs only endpoint extrema because every edge
  coordinate is linear in segment parameter.
- A zero-length powered segment is certified as one point.
- A vertical powered segment has constant terrain/LOS geometry in $z$ and
  uses its altitude endpoint extrema.
- A goal-terminal action is certified only to its actual circle-intersection
  endpoint. It does not inherit the feasibility of the unused remainder of the
  full grid action.

## Solver integration

`segment_feasibility.py` is the authoritative scalar certificate.

- `bellman.evaluate_powered_segment` uses the occluded certificate while
  retaining its configured samples only for hazard quadrature and stored path
  display.
- `successor_grid_solver._physical_edge_metrics` uses the visible certificate
  for virtual switching edges.
- Regular grid edges use one exact reference clearance per spatial offset and
  start-z row. Because changing the start altitude only adds the same constant
  to terrain and visible-LOS margins, the reference result is broadcast over
  the altitude grid. Results are cached across speed values sharing an offset.
- Sparse goal-terminal starts are recertified individually at their truncated
  endpoints.
- Selected candidate validation recomputes exact powered and glide margins and
  reports actual terrain, LOS, and airspace checks instead of placeholder
  booleans.

Detection hazard and edge cost remain the configured trapezoidal finite
objective. The P1 change only strengthens edge admissibility.

## Verification

The following regressions were added or strengthened:

- an interior cubic terrain peak missed by endpoint checks;
- a narrow LOS breakpoint missed by generic coarse fractions;
- visible-constraint cutoff at the sensor;
- combined terrain/LOS/airspace diagnostics;
- zero-length and vertical powered segments;
- deterministic multi-hill comparisons against 20,001-point dense references;
- solver metadata and selected-policy terrain/LOS/airspace checks; and
- the pre-existing independent NetworkX shortest-path and replay regressions.

The complete `p1b_4D` suite passed:

```text
Ran 101 tests in 409.047s
OK
```

## Direction-B preservation audit

`audit_p1_direction_b_geometry_certificates.py` reevaluated every stored B2/B3
selected policy without rerunning optimization:

| Quantity | Result |
|---|---:|
| Stored B2/B3 cases | 54 |
| Cases with selected policies | 42 |
| Previously infeasible/not applicable | 12 |
| Exact-certified selected policies | 42 |
| Failed selected policies | 0 |
| Minimum powered terrain margin | $-6.5874\times10^{-7}$ m |
| Minimum powered occlusion margin | $3.0$ m |
| Minimum glide terrain margin | $0.0382234$ m |
| Minimum glide LOS margin | $0$ m at the switching boundary |

The small negative powered terrain margin remains inside the predeclared
$10^{-6}$ m terrain tolerance. Because P1 only removes formerly admitted edges,
and every previously selected minimum-cost path remains admitted with unchanged
finite cost, the stored B2/B3 optimum values remain minima of the strengthened
finite graph.

The reproducible audit output is
`results/direction_b/p1_exact_geometry_audit.json`.

## Remaining scope boundary

P1 certifies geometry against the implemented spline and swept LOS boundary.
It does not claim that the spline equals an unknown real terrain surface, that
the swept boundary is free of upstream modeling error, that quadrature is an
exact continuous hazard integral, or that the finite graph contains every
continuous trajectory. P2 supplies the graph-level mathematical soundness,
discretization-relative completeness, and finite exactness proof package.

