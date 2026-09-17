# Stage 14.7 theoretical-versus-empirical interpretation

All exponents below are descriptive log-log fits over the measured finite range. They are not formal Big-O results and are not extrapolated.
Negative exponents on spacing variables mean runtime rises as the discretization spacing becomes finer.

- `spatial__local_sse__T_total_s_vs_spatial_resolution_m`: alpha=-1.44932, c=6947.14, R^2=0.965124, n=3, range=[25, 100].
- `heading__local_sse__T_total_s_vs_heading_resolution_deg`: alpha=-2.18573, c=1784.32, R^2=0.998075, n=4, range=[2.5, 15].
- `switching__local_sse__T_total_s_vs_switching_contour_spacing`: alpha=-1.78622, c=0.741942, R^2=0.952956, n=3, range=[0.0833333, 0.166667].
- `radius__local_sse__T_total_s_vs_r_neighbor`: alpha=0.426161, c=16.1731, R^2=0.746468, n=4, range=[1, 8].
- `radius__local_sse__T_total_s_vs_N_eval_unique`: alpha=0.60937, c=10.7356, R^2=0.806819, n=4, range=[2, 9].

The active-state and active-edge counts are retained separately. No artificial `N_S_active + N_E` plot variable is constructed.

Noncompleted repetitions: 9 total; 6 model-infeasible and 3 computational/instrumentation failures.

Reachability pruning, changing edge density, fixed overhead, and local-search path changes can cause departures from a simple power law.
