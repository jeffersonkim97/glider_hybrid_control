#!/usr/bin/env python3
"""
p1b_4d_dp.py — 4D Backward DP for Fixed-Wing Glider SSG.
State: (z, h, v, gamma), Action: gamma_cmd.  Objective: J = PoD + W_TIME*T.
"""

import os, math, time, json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── params (copied from notebook cell bebf78df) ──────────────────────────────
params = {
    'G': 9.81, 'RHO': 1.225, 'M': 1.5, 'S': 0.1, 'CD': 0.025,
    'V_STALL': 14.0, 'V_MAX': 100.0,
    'Z_GOAL': 2500.0, 'H_GOAL': 0.0,
    'DT_GLIDE': 1.0, 'DT_DP': 2.0, 'DT_POW': 1.0,
    'Z_RIDGE': 1250.0, 'H_RIDGE': 100.0, 'SIGMA_TERRAIN': 200.0,
    'Z_SENSOR': 2000.0, 'H_SENSOR': 0.0,
    'LAM_DOPPLER': 1.33e6, 'LAM_RCS': 5.2e8,
    'SIGMA0': 1.0, 'GAMMA_MIN': -90.0,
    'R_FLOOR': 10.0, 'LAM_FLOOR': 5e-4,
    'W_TIME': 1e-3, 'TAU_CONTROL': 1.0,
    'H_MAX_GRID': 300.0,
}
params['CL_MAX'] = params['M'] * params['G'] / (0.5 * params['RHO'] * params['V_STALL']**2 * params['S'])
params['V_LAUNCH'] = 1.5 * params['V_STALL']

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ─── terrain / LOS (copied from notebook cell ab1cea42) ──────────────────────
def h_terrain(z, params):
    return params['H_RIDGE'] * np.exp(-0.5 * ((z - params['Z_RIDGE']) / params['SIGMA_TERRAIN'])**2)

def compute_LOS(z_sensor, h_sensor, params):
    zg = np.linspace(0.0, z_sensor, 2000, endpoint=False)
    hg = h_terrain(zg, params)
    sl = (hg - h_sensor) / (zg - z_sensor)
    st = hg * (-(zg - params['Z_RIDGE']) / params['SIGMA_TERRAIN']**2)
    diff = sl - st
    sc = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    z_t = float((zg[sc[0]] + zg[sc[0]+1]) / 2) if len(sc) else float(zg[np.nanargmin(np.abs(diff))])
    h_t = float(h_terrain(z_t, params))
    slope = (h_t - h_sensor) / (z_t - z_sensor)
    return {'slope': float(slope), 'intercept': float(h_sensor - slope * z_sensor), 'z_tangent': z_t}

# ─── dynamics (copied from notebook cell cf192e90) ────────────────────────────
def drag_coefficient(CL, params): return params['CD']

def required_CL(v, gamma, gamma_cmd, params):
    vs = max(v, 1e-3)
    L_req = params['M'] * vs * (gamma_cmd - gamma) / params['TAU_CONTROL'] + \
            params['M'] * params['G'] * math.cos(gamma)
    return float(np.clip(L_req / (0.5 * params['RHO'] * vs**2 * params['S']), 0.0, params['CL_MAX']))

def glider_dynamics(state, gamma_cmd, params):
    def deriv(s):
        z, h, vz, vh = s
        v = math.hypot(vz, vh); vs = max(v, 1e-6)
        g = math.atan2(vh, vz)
        CL = required_CL(v, g, gamma_cmd, params)
        L = 0.5 * params['RHO'] * v**2 * CL * params['S']
        D = 0.5 * params['RHO'] * v**2 * params['CD'] * params['S']
        az = (-D * vz - L * vh) / vs / params['M']
        ah = (-D * vh + L * vz) / vs / params['M'] - params['G']
        return np.array([vz, vh, az, ah])
    dt = params['DT_GLIDE']
    z0, h0, v0, g0 = state
    s = np.array([z0, h0, v0 * math.cos(g0), v0 * math.sin(g0)])
    k1 = deriv(s); k2 = deriv(s+0.5*dt*k1); k3 = deriv(s+0.5*dt*k2); k4 = deriv(s+dt*k3)
    s2 = s + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    z2, h2, vz2, vh2 = s2
    return (float(z2), float(max(h2, 0.0)), float(math.hypot(vz2, vh2)), float(math.atan2(vh2, vz2)))

# ─── grid ─────────────────────────────────────────────────────────────────────
# State gamma extends into climb angles (up to GAMMA_MAX_STATE) to represent
# the powered phase's entry attitude: powered_trajectory is a straight-line
# dash from launch (0,0) to the chosen switching point, and for
# switching-candidates close to launch that line is steep -- up to ~90 deg
# for the earliest LOS-line candidates. The action grid (gamma_cmd) stays
# glide-only [GAMMA_MIN, 0]: glider_dynamics has no thrust, so commanding a
# sustained climb only bleeds energy -- the optimizer should never choose to
# hold a climb, only fly the transient (autopilot lag) out of one it was
# handed at entry. NG grows to keep the same ~5 deg/step resolution over the
# now-180-deg state range (vs. 90 deg before).
GAMMA_MAX_STATE = 90.0
NZ, NH, NV, NG, NA = 100, 100, 4, 37, 19
z_grid = np.linspace(0.0, params['Z_GOAL'] * 1.1, NZ)
h_grid = np.linspace(0.0, params['H_MAX_GRID'], NH)
v_grid = np.array([14.0, 30.0, 55.0, 100.0])
g_grid = np.linspace(params['GAMMA_MIN'], GAMMA_MAX_STATE, NG)   # degrees, state
a_grid = np.linspace(params['GAMMA_MIN'], 0.0, NA)               # degrees, action (gamma_cmd)
n_sub = int(round(params['DT_DP'] / params['DT_GLIDE']))   # 2

# ─── precompute transitions ────────────────────────────────────────────────────
print('Precomputing transitions (iv x ig x ia = 4x19x19) ...')
t0 = time.time()
H_PRE = 150.0   # safe start altitude avoids h-floor clipping in dynamics
dz_sub = np.zeros((NV, NG, NA, n_sub))   # displacement at each substep
dh_sub = np.zeros((NV, NG, NA, n_sub))
v_next_pre = np.full((NV, NG, NA), np.inf)
g_next_pre = np.full((NV, NG, NA), np.inf)   # degrees
stall_inv   = np.ones((NV, NG, NA), dtype=bool)   # start as all-invalid, clear when valid

for iv, v0 in enumerate(v_grid):
    for ig, g0d in enumerate(g_grid):
        g0r = math.radians(g0d)
        for ia, gad in enumerate(a_grid):
            gar = math.radians(gad)
            state = (0.0, H_PRE, v0, g0r)
            z_p, h_p = 0.0, H_PRE
            valid = True
            for k in range(n_sub):
                state = glider_dynamics(state, gar, params)
                zk, hk, vk, gk = state
                dz_sub[iv, ig, ia, k] = zk - z_p
                dh_sub[iv, ig, ia, k] = hk - h_p
                z_p, h_p = zk, hk
                if vk < 0.5 * params['V_STALL']:
                    valid = False
                    break
            if valid:
                v_next_pre[iv, ig, ia] = vk
                g_next_pre[iv, ig, ia] = math.degrees(gk)
                stall_inv[iv, ig, ia] = False

dz_cs = np.cumsum(dz_sub, axis=3)   # (NV, NG, NA, n_sub) cumulative
dh_cs = np.cumsum(dh_sub, axis=3)
dz_fin = dz_cs[:, :, :, -1]         # (NV, NG, NA) final displacement
dh_fin = dh_cs[:, :, :, -1]
print(f'  {time.time()-t0:.1f}s  stall_invalid: {stall_inv.sum()}/{NV*NG*NA} combos')

# ─── precompute stage cost (vectorized over z, h) ─────────────────────────────
print('Computing stage costs (NZ x NH x NV x NG) ...')
t0 = time.time()
LOS = compute_LOS(params['Z_SENSOR'], params['H_SENSOR'], params)

ZZ = z_grid[:, np.newaxis]    # (NZ, 1)
HH = h_grid[np.newaxis, :]    # (1, NH)
r2d = np.sqrt((ZZ - params['Z_SENSOR'])**2 + (HH - params['H_SENSOR'])**2)
r2d_eff = np.maximum(r2d, params['R_FLOOR'])
h_los_2d = LOS['slope'] * ZZ + LOS['intercept']
occluded = (ZZ < LOS['z_tangent']) & (HH < h_los_2d)   # (NZ, NH)
terrain_2d = h_terrain(ZZ, params)                       # (NZ, 1)
# Gaussian terrain is asymptotically > 0 everywhere (never exactly 0 far from
# the ridge), so an un-tolerant `HH < terrain_2d` misclassifies the entire
# h=0 row as "on terrain" at every z. Epsilon matches the point-mass sweep's
# `h < h_terrain(z, params) - 1e-6` convention.
on_terrain = (HH < terrain_2d - 1e-6)                    # (NZ, NH)

# The glide-phase DP domain is the LOS-visible zone only, exactly like the
# point-mass baseline's `exclude_mask` (Step 6-1). The occlusion pocket
# (behind the ridge, hidden from the sensor) is the powered phase's
# territory -- launch flies a straight line from (0,0) to a switching point
# ON the LOS boundary, then the glide-phase cost-to-go map takes over from
# there. Letting cost-to-go propagate INTO the occlusion pocket lets the
# glide-phase DP silently re-solve territory that already belongs to (and is
# handled differently by) the powered phase, which is not what this map is
# for.
exclude_mask_2d = on_terrain | occluded                  # (NZ, NH)

stage_cost = np.zeros((NZ, NH, NV, NG))
los_z_unit = (params['Z_SENSOR'] - ZZ) / r2d_eff
los_h_unit = (params['H_SENSOR'] - HH) / r2d_eff

for iv, v0 in enumerate(v_grid):
    for ig, g0d in enumerate(g_grid):
        g0r = math.radians(g0d)
        vz = v0 * math.cos(g0r); vh = v0 * math.sin(g0r)
        v_r = vz * los_z_unit + vh * los_h_unit
        lam_dop = params['LAM_DOPPLER'] * v_r**2 / r2d_eff**4
        cos2 = math.cos(math.radians(g0d - params['GAMMA_MIN']))**2
        lam_rcs = params['LAM_RCS'] * params['SIGMA0'] * cos2 / r2d_eff**4
        lam = np.maximum(lam_dop + lam_rcs, params['LAM_FLOOR'])
        lam[occluded] = 0.0
        stage_cost[:, :, iv, ig] = 1.0 - np.exp(-lam * params['DT_DP']) + params['W_TIME'] * params['DT_DP']

stage_cost[exclude_mask_2d] = np.inf
print(f'  {time.time()-t0:.1f}s')

# ─── DP initialisation ────────────────────────────────────────────────────────
print('Initialising J ...')
J = np.full((NZ, NH, NV, NG), np.inf)
policy = np.full((NZ, NH, NV, NG), -1, dtype=np.int16)

# Goal = actually on the ground (h ~ H_GOAL) at or past the target range
# (z >= Z_GOAL) -- NOT merely crossing Z_GOAL at any altitude. The action
# set here is forward-only (gamma in [GAMMA_MIN, 0], no backward flight like
# the point-mass baseline's GAMMA_MIN_POINTMASS extension), so z increases
# monotonically and can never be "corrected" after an overshoot; the only
# thing that determines mission success once z >= Z_GOAL is whether the
# glider is actually down (h -> 0, which glider_dynamics enforces as a hard
# floor) rather than still airborne. Grounding short of Z_GOAL is a failure,
# not a goal.
iz_goal = int(np.searchsorted(z_grid, params['Z_GOAL']))
ih_goal = int(np.argmin(np.abs(h_grid - params['H_GOAL'])))
J[iz_goal:, ih_goal, :, :] = 0.0
J[exclude_mask_2d] = np.inf   # terrain + occlusion pocket override goal

n_finite_init = int(np.isfinite(J).sum())
print(f'  iz_goal={iz_goal}, z[iz_goal]={z_grid[iz_goal]:.1f}m, ih_goal={ih_goal}, h[ih_goal]={h_grid[ih_goal]:.1f}m, finite J cells at init: {n_finite_init}')

# ─── value iteration ──────────────────────────────────────────────────────────
print('Starting value iteration (live-update sweep, max 200 iters, eps=1e-6) ...')
max_iters = 20
eps = 1e-6
delta_history = []
t_total = time.time()

# Precompute position-independent validity parts (stall + v/g bounds)
valid_base = ~stall_inv   # (NV, NG, NA)
valid_base &= np.isfinite(v_next_pre) & (v_next_pre >= v_grid[0]) & (v_next_pre <= v_grid[-1])
valid_base &= np.isfinite(g_next_pre) & (g_next_pre >= g_grid[0]) & (g_next_pre <= g_grid[-1])
# h-displacement bounds (conservative: only need h_next in grid)
# (checked per-slice below since it depends on h_grid[ih])

for it in range(max_iters):
    t_iter = time.time()
    J_before = J.copy()

    # Live-update sweep: rebuild interpolator each z-slice so downstream
    # slices benefit from updates made earlier in the same sweep.
    for iz in range(NZ - 1, -1, -1):
        if iz >= iz_goal:
            continue
        z_cur = float(z_grid[iz])

        # Rebuild reach-weighted interpolators on CURRENT J.
        # Using J_fill=1e12 for inf causes "bleed": a 1% weight on an inf
        # cell contaminates the result with 1e10+ even when 99% of the
        # interpolation weight is on a valid finite cell. Fix: build two
        # interpolators (reachability mask + J-value where finite, 0 elsewhere)
        # and recover the conditional mean J over reachable corners only.
        J_reach = np.isfinite(J).astype(np.float64)
        J_val   = np.where(np.isfinite(J), J, 0.0)
        interp_r = RegularGridInterpolator(
            (z_grid, h_grid, v_grid, g_grid), J_reach,
            method='linear', bounds_error=False, fill_value=0.0)
        interp_v = RegularGridInterpolator(
            (z_grid, h_grid, v_grid, g_grid), J_val,
            method='linear', bounds_error=False, fill_value=0.0)

        # ── terrain/ground validity mask (NH, NV, NG, NA) ─────────────────
        valid = np.broadcast_to(valid_base[None, :, :, :], (NH, NV, NG, NA)).copy()

        for k in range(n_sub):
            z_sub_k = z_cur + dz_cs[np.newaxis, :, :, :, k]
            h_sub_k = h_grid[:, None, None, None] + dh_cs[None, :, :, :, k]
            ht_sub  = h_terrain(z_sub_k, params)
            valid &= (h_sub_k >= ht_sub - 1e-6)   # same epsilon as on_terrain -- see note above
            valid &= ~((h_sub_k < 0.0) & (z_sub_k < params['Z_GOAL']))

        # ── next-state coords ──────────────────────────────────────────────
        z_next_3 = z_cur + dz_fin                                          # (NV,NG,NA)
        h_next_4 = h_grid[:, None, None, None] + dh_fin[None, :, :, :]   # (NH,NV,NG,NA)

        valid &= (h_next_4 >= h_grid[0]) & (h_next_4 <= h_grid[-1])
        valid &= (z_next_3[None] >= z_grid[0]) & (z_next_3[None] <= z_grid[-1])

        # ── batch interpolation (reach-weighted) ───────────────────────────
        shape4 = (NH, NV, NG, NA)
        pts = np.empty((NH * NV * NG * NA, 4))
        pts[:, 0] = np.broadcast_to(z_next_3[None, :, :, :], shape4).ravel()
        pts[:, 1] = h_next_4.ravel()
        pts[:, 2] = np.broadcast_to(v_next_pre[None, :, :, :], shape4).ravel()
        pts[:, 3] = np.broadcast_to(g_next_pre[None, :, :, :], shape4).ravel()

        r_flat = interp_r(pts)
        v_flat = interp_v(pts)
        # Recover E[J | reachable corners]: divide by reachability weight.
        # Cells where r < 0.5 have majority-inf interpolation neighbourhood
        # and are considered unreachable.
        r_safe = np.where(r_flat > 0.5, r_flat, 1.0)
        J_fut_flat = np.where(r_flat > 0.5, v_flat / r_safe, np.inf)
        J_fut = J_fut_flat.reshape(shape4)
        J_fut[~valid] = np.inf

        # ── total cost, min over actions ───────────────────────────────────
        total = stage_cost[iz, :, :, :, None] + J_fut   # (NH,NV,NG,NA)
        J_best   = np.nanmin(total, axis=-1)             # (NH,NV,NG)
        pol_best = np.argmin(total, axis=-1)

        improve = J_best < J[iz, :, :, :]
        J[iz, improve]      = J_best[improve]
        policy[iz, improve] = pol_best[improve].astype(np.int16)

    # Convergence: track both value change and newly-reachable cells
    n_new = int((~np.isfinite(J_before) & np.isfinite(J)).sum())
    fin = np.isfinite(J_before) & np.isfinite(J)
    max_delta = float(np.max(np.abs(J[fin] - J_before[fin]))) if fin.any() else 1.0
    delta_history.append(max_delta)
    n_fin = int(np.isfinite(J).sum())
    print(f'  iter {it+1:03d}: max_delta={max_delta:.3e}  new_cells={n_new}  finite={n_fin}  t={time.time()-t_iter:.1f}s')
    if max_delta < eps and n_new == 0:
        print('  converged!')
        break

print(f'Total DP time: {time.time()-t_total:.1f}s')

# ─── save results ─────────────────────────────────────────────────────────────
npz_path = os.path.join(DATA_DIR, 'p1b_4d_dp_results.npz')
np.savez_compressed(npz_path,
    J_4D=J, policy_4D=policy,
    z_grid=z_grid, h_grid=h_grid, v_grid=v_grid,
    gamma_grid=g_grid, gamma_cmd_grid=a_grid)
print(f'Results saved to {npz_path}')

# ─── validation ───────────────────────────────────────────────────────────────
print('\n--- Validation ---')
goal_J = J[iz_goal:, ih_goal, :, :]
print(f'  J at goal (should be 0): max={np.nanmax(goal_J[np.isfinite(goal_J)]):.4f}')
print(f'  J at terrain (should be inf): all inf? {np.all(J[on_terrain] == np.inf)}')
print(f'  J at occlusion pocket (should be inf, glide-phase excludes it): all inf? {np.all(J[occluded] == np.inf)}')
J_2D = np.nanmin(J[:, :, :, :].reshape(NZ, NH, NV*NG), axis=-1)
J_2D[J_2D >= 1e10] = np.inf
fin_2d = J_2D[np.isfinite(J_2D)]
print(f'  J_2D = min_{{v,g}} J_4D: range [{fin_2d.min():.4f}, {fin_2d.max():.4f}]')
print(f'  finite J_2D cells: {np.isfinite(J_2D).sum()} / {NZ*NH}')

# ─── plots ────────────────────────────────────────────────────────────────────
print('\nGenerating plots ...')

# J_2D projection (min over v, gamma)
J_4D_min = J.reshape(NZ, NH, NV * NG)
J_2D = np.nanmin(J_4D_min, axis=-1)
J_2D[J_2D >= 1e10] = np.nan
pol_4D_min_v = np.full((NZ, NH), np.nan)
for iz in range(NZ):
    for ih in range(NH):
        slab = J[iz, ih, :, :]
        if np.any(np.isfinite(slab)):
            idx = np.unravel_index(np.nanargmin(slab), slab.shape)
            best_pol = policy[iz, ih, idx[0], idx[1]]
            pol_4D_min_v[iz, ih] = float(a_grid[best_pol]) if best_pol >= 0 else np.nan

# terrain / LOS overlay
zt = np.linspace(0, z_grid[-1], 400)
ht = h_terrain(zt, params)
h_los_line = LOS['slope'] * zt + LOS['intercept']

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(z_grid, h_grid, J_2D.T, cmap='magma', shading='auto')
fig.colorbar(im, ax=ax, label='J_2D = min_{v,γ} J_4D  (PoD + W_TIME·T)')
ax.fill_between(zt, 0, ht, color='gray', alpha=0.7, label='terrain')
ax.plot(zt, h_los_line, 'c--', lw=1.2, label='LOS')
ax.scatter([params['Z_SENSOR']], [params['H_SENSOR']], c='red', marker='^', s=100, zorder=5, label='sensor')
ax.scatter([params['Z_GOAL']], [params['H_GOAL']], c='gold', marker='*', s=200, zorder=5, label='goal')
ax.set_xlim(z_grid[0], z_grid[-1]); ax.set_ylim(h_grid[0], h_grid[-1])
ax.set_xlabel('z (m)'); ax.set_ylabel('h (m)')
ax.set_title('4D DP: J_2D(z,h) = min_{v,γ} J_4D(z,h,v,γ)')
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'J_2D_projection.png'), dpi=130)
plt.close()
print('  J_2D_projection.png saved')

# Policy 2D projection
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(z_grid, h_grid, pol_4D_min_v.T, cmap='RdYlGn', shading='auto',
                   vmin=params['GAMMA_MIN'], vmax=0.0)
fig.colorbar(im, ax=ax, label='optimal γ_cmd at argmin_{v,γ} J  (degrees)')
ax.fill_between(zt, 0, ht, color='gray', alpha=0.7)
ax.plot(zt, h_los_line, 'c--', lw=1.2, label='LOS')
ax.scatter([params['Z_SENSOR']], [params['H_SENSOR']], c='red', marker='^', s=100, zorder=5)
ax.scatter([params['Z_GOAL']], [params['H_GOAL']], c='gold', marker='*', s=200, zorder=5)
ax.set_xlim(z_grid[0], z_grid[-1]); ax.set_ylim(h_grid[0], h_grid[-1])
ax.set_xlabel('z (m)'); ax.set_ylabel('h (m)')
ax.set_title('4D DP: Optimal γ_cmd projected to (z,h)')
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'policy_2D_projection.png'), dpi=130)
plt.close()
print('  policy_2D_projection.png saved')

# J slices at fixed v (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
v_labels = ['14 m/s (stall)', '30 m/s (shallow glide)', '55 m/s (moderate dive)', '100 m/s (terminal)']
for k, (ax, vlbl) in enumerate(zip(axes.ravel(), v_labels)):
    Jv = np.nanmin(J[:, :, k, :], axis=-1)   # min over gamma: (NZ, NH)
    Jv[Jv >= 1e10] = np.nan
    im = ax.pcolormesh(z_grid, h_grid, Jv.T, cmap='magma', shading='auto')
    fig.colorbar(im, ax=ax, label='J')
    ax.fill_between(zt, 0, ht, color='gray', alpha=0.7)
    ax.plot(zt, h_los_line, 'c--', lw=1.0)
    ax.scatter([params['Z_SENSOR']], [params['H_SENSOR']], c='red', marker='^', s=60, zorder=5)
    ax.scatter([params['Z_GOAL']], [params['H_GOAL']], c='gold', marker='*', s=120, zorder=5)
    ax.set_xlim(z_grid[0], z_grid[-1]); ax.set_ylim(h_grid[0], h_grid[-1])
    ax.set_title(f'v = {vlbl}'); ax.set_xlabel('z (m)'); ax.set_ylabel('h (m)')
fig.suptitle('4D DP: J slice at each v (min over γ)')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'J_slices.png'), dpi=130)
plt.close()
print('  J_slices.png saved')

# Convergence
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(range(1, len(delta_history)+1), delta_history, 'o-')
ax.axhline(eps, color='red', ls='--', label=f'eps={eps}')
ax.set_xlabel('iteration'); ax.set_ylabel('max |ΔJ|'); ax.set_title('Value iteration convergence')
ax.legend(); ax.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'convergence.png'), dpi=130)
plt.close()
print('  convergence.png saved')

# ─── switching-point search (powered phase -> glide phase handoff) ────────────
print('\nSearching switching-point candidates ...')

# Reuse the same LOS-line switching candidates as the 2D point-mass baseline
# (Step 2 / terrain.json) -- pure geometry, independent of vehicle model.
terrain_path = os.path.join(DATA_DIR, 'terrain.json')
with open(terrain_path, 'r') as f:
    terrain_data = json.load(f)
switching_candidates = terrain_data['switching_candidates']


def powered_trajectory(z_sw, h_sw, params):
    """Straight-line kinematics from (0,0) to (z_sw,h_sw) at constant speed
    V_LAUNCH -- same model as the notebook's Step 4 `powered_trajectory` (no
    thrust/drag here by design; that physics lives in the glide phase only).
    """
    v = params['V_LAUNCH']
    dt = params['DT_POW']
    distance = math.sqrt(z_sw**2 + h_sw**2)
    if distance <= 0.0:
        return [(0.0, 0.0, float(v), 0.0)]
    n_steps = max(1, int(math.ceil(distance / (v * dt))))
    z_values = np.linspace(0.0, z_sw, n_steps + 1)
    h_values = np.linspace(0.0, h_sw, n_steps + 1)
    gamma = math.atan2(h_sw, z_sw)
    return [(float(z), float(h), float(v), float(gamma)) for z, h in zip(z_values, h_values)]


# Reach-weighted interpolator over J_4D (same trick as the value-iteration
# loop above): switching candidates sit exactly ON the LOS boundary line, so
# a naive linear interpolator blends finite (visible-side) and inf
# (occluded-side) grid corners -- inf contaminates the blend and every
# candidate looks unreachable. Recover E[J | reachable corners] instead.
J_reach_glob = np.isfinite(J).astype(np.float64)
J_val_glob = np.where(np.isfinite(J), J, 0.0)
interp_r_glob = RegularGridInterpolator(
    (z_grid, h_grid, v_grid, g_grid), J_reach_glob,
    method='linear', bounds_error=False, fill_value=0.0)
interp_v_glob = RegularGridInterpolator(
    (z_grid, h_grid, v_grid, g_grid), J_val_glob,
    method='linear', bounds_error=False, fill_value=0.0)


def J_lookup(pt):
    r = float(interp_r_glob(pt)[0])
    v = float(interp_v_glob(pt)[0])
    return v / r if r > 0.5 else np.inf


candidate_results = []
for z_sw, h_sw in switching_candidates:
    if z_sw <= 0.0 or h_sw > params['H_MAX_GRID']:
        candidate_results.append({'z_sw': float(z_sw), 'h_sw': float(h_sw), 'J': float('inf'), 'reachable': False})
        continue
    v_entry = params['V_LAUNCH']
    gamma_entry_deg = math.degrees(math.atan2(h_sw, z_sw))
    J_entry = J_lookup([[z_sw, h_sw, v_entry, gamma_entry_deg]])
    candidate_results.append({
        'z_sw': float(z_sw), 'h_sw': float(h_sw), 'v_entry': v_entry, 'gamma_entry_deg': gamma_entry_deg,
        'J': J_entry, 'reachable': bool(np.isfinite(J_entry)),
    })

reachable = [c for c in candidate_results if c['reachable']]
print(f'{len(reachable)} / {len(candidate_results)} switching candidates reachable')
if not reachable:
    raise RuntimeError('No switching candidate reached the goal -- check DP domain / grid.')
best = min(reachable, key=lambda c: c['J'])
print(f"optimal switching point: (z={best['z_sw']:.1f}, h={best['h_sw']:.1f}), "
      f"entry (v={best['v_entry']:.1f}, gamma={best['gamma_entry_deg']:.1f} deg), J={best['J']:.4f}")

# ─── forward simulation: powered phase (straight line) + glide phase (closed-loop) ──
print('\nRunning forward simulation (powered + glide) ...')

# Policy lookup via nearest-neighbour (policy is integer-valued)
pol_interp = RegularGridInterpolator(
    (z_grid, h_grid, v_grid, g_grid),
    policy.astype(float),
    method='nearest', bounds_error=False, fill_value=-1.0,
)

powered_states = powered_trajectory(best['z_sw'], best['h_sw'], params)
t_pow = [i * params['DT_POW'] for i in range(len(powered_states))]
z_pow = [s[0] for s in powered_states]
h_pow = [s[1] for s in powered_states]
v_pow = [s[2] for s in powered_states]
g_pow = [math.degrees(s[3]) for s in powered_states]
print(f'  powered phase: {len(t_pow)} steps, t={t_pow[-1]:.1f}s (zero detection cost -- inside occlusion pocket)')

state = powered_states[-1]
traj_z, traj_h, traj_v, traj_g, traj_pod, traj_J = [], [], [], [], [], []
lam_sum = 0.0
dt_sub = params['DT_GLIDE']

for step in range(2000):
    z, h, v, g = state
    if h <= 0.0:
        if z >= params['Z_GOAL']:
            print(f'  reached goal at step {step}: z={z:.1f}, h={h:.1f}')
        else:
            print(f'  crashed short of goal at step {step}: z={z:.1f} (Z_GOAL={params["Z_GOAL"]:.1f})')
        break
    if v < params['V_STALL']:
        print(f'  stall at step {step}: v={v:.1f}')
        break

    # Query policy (nearest-neighbor in 4D) from the REAL current state
    g_deg = math.degrees(g)
    pt = np.array([[z, h, v, g_deg]])
    ia = int(round(pol_interp(pt)[0]))
    ia = int(np.clip(ia, 0, NA - 1))
    g_cmd_rad = math.radians(a_grid[ia])

    # Propagate n_sub substeps
    stage_lam = 0.0
    for _ in range(n_sub):
        zs, hs, vs, gs = state
        h_los_cur = LOS['slope'] * zs + LOS['intercept']
        if (zs < LOS['z_tangent']) and (hs < h_los_cur):
            lam_here = 0.0
        else:
            gs_deg = math.degrees(gs)
            r = math.sqrt((zs - params['Z_SENSOR'])**2 + (hs - params['H_SENSOR'])**2)
            r_eff = max(r, params['R_FLOOR'])
            vz = vs * math.cos(gs); vh = vs * math.sin(gs)
            los_z = (params['Z_SENSOR'] - zs) / r_eff
            los_h = (params['H_SENSOR'] - hs) / r_eff
            v_r = vz * los_z + vh * los_h
            lam_dop = params['LAM_DOPPLER'] * v_r**2 / r_eff**4
            cos2 = math.cos(math.radians(gs_deg - params['GAMMA_MIN']))**2
            lam_rcs = params['LAM_RCS'] * params['SIGMA0'] * cos2 / r_eff**4
            lam_here = max(lam_dop + lam_rcs, params['LAM_FLOOR'])
        stage_lam += lam_here * dt_sub
        state = glider_dynamics(state, g_cmd_rad, params)

    # Record the state AFTER this stage's motion (so the final landed point
    # is always included -- the old version recorded pre-motion state and
    # dropped the last, landed point).
    lam_sum += stage_lam
    z2, h2, v2, g2 = state
    cum_pod = 1.0 - math.exp(-lam_sum)
    cum_T = t_pow[-1] + (step + 1) * params['DT_DP']
    cum_J = cum_pod + params['W_TIME'] * cum_T
    traj_z.append(z2); traj_h.append(h2); traj_v.append(v2); traj_g.append(math.degrees(g2))
    traj_pod.append(cum_pod); traj_J.append(cum_J)
else:
    print('  max steps reached without goal')

final_pod = traj_pod[-1] if traj_pod else 0.0
T_total = t_pow[-1] + len(traj_z) * params['DT_DP']
J_total = final_pod + params['W_TIME'] * T_total
print(f'  final PoD={final_pod:.4f}, T={T_total:.1f}s, W_TIME*T={params["W_TIME"]*T_total:.4f}, J={J_total:.4f}')

# ─── save trajectory (powered + glide, same shape convention as step6_2.json) ──
t_full = t_pow + [t_pow[-1] + (i + 1) * params['DT_DP'] for i in range(len(traj_z))]
z_full = z_pow + traj_z
h_full = h_pow + traj_h
v_full = v_pow + traj_v
g_full = g_pow + traj_g
pod_full = [0.0] * len(t_pow) + traj_pod
J_full = [0.0] * len(t_pow) + traj_J

trajectory_4d_data = {
    'candidates': candidate_results,
    'switching_point': [best['z_sw'], best['h_sw']],
    'powered_end_index': len(t_pow) - 1,
    't_traj': t_full, 'z_traj': z_full, 'h_traj': h_full, 'v_traj': v_full, 'gamma_traj': g_full,
    'pod_traj': pod_full, 'cost_traj': J_full,
    'final_pod': final_pod, 'T_total': T_total, 'J_total': J_total,
}
with open(os.path.join(DATA_DIR, 'trajectory_4d.json'), 'w') as f:
    json.dump(trajectory_4d_data, f, indent=2)
print('  trajectory_4d.json saved')

# Trajectory plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.fill_between(zt, 0, ht, color='gray', alpha=0.7, label='terrain')
ax.plot(zt, h_los_line, 'c--', lw=1.2, label='LOS')
ax.plot(z_pow, h_pow, color='deepskyblue', lw=1.5, label='powered phase')
im = ax.scatter(traj_z, traj_h, c=traj_J, cmap='magma', s=6, zorder=4, label='glide phase (J-colored)')
fig.colorbar(im, ax=ax, label='cumulative J')
ax.scatter([params['Z_SENSOR']], [params['H_SENSOR']], c='red', marker='^', s=100, zorder=5, label='sensor')
ax.scatter([params['Z_GOAL']], [params['H_GOAL']], c='gold', marker='*', s=200, zorder=5, label='goal')
ax.scatter([0.0], [0.0], c='lime', marker='o', s=80, zorder=5, label='launch')
ax.scatter([best['z_sw']], [best['h_sw']], c='white', edgecolor='black', marker='D', s=70, zorder=6, label='switch')
ax.set_xlim(z_grid[0], z_grid[-1]); ax.set_ylim(h_grid[0], h_grid[-1])
ax.set_xlabel('z (m)'); ax.set_ylabel('h (m)')
ax.set_title('4D DP: Optimal trajectory (powered + glide)')
ax.legend(fontsize=7)

ax2 = axes[1]
t_arr = np.arange(len(traj_v)) * params['DT_DP'] + t_pow[-1]
ax2.plot(t_arr, traj_v, label='v (m/s)', color='dodgerblue')
ax2b = ax2.twinx()
ax2b.plot(t_arr, traj_g, label='gamma (deg)', color='orange', ls='--')
ax2b.set_ylabel('gamma (deg)', color='orange')
ax2.axvline(t_pow[-1], color='gray', ls=':', lw=1, label='switch')
ax2.set_xlabel('t (s)'); ax2.set_ylabel('v (m/s)', color='dodgerblue')
ax2.set_title('State history: v and gamma (glide phase)')
ax2.legend(loc='upper left', fontsize=8); ax2b.legend(loc='upper right', fontsize=8)

fig.suptitle(f'4D DP trajectory: switch=(z={best["z_sw"]:.0f},h={best["h_sw"]:.0f}), '
             f'PoD={final_pod:.4f}, T={T_total:.0f}s, J={J_total:.4f}')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'trajectory_4d.png'), dpi=130)
plt.close()
print('  trajectory_4d.png saved')
print('\nDone.')
