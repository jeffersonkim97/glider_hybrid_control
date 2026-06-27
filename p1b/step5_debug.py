import json, os, numpy as np

DATA_DIR = os.path.join(os.getcwd(), 'data')
params = json.load(open(os.path.join(DATA_DIR, 'params.json')))
terrain_data = json.load(open(os.path.join(DATA_DIR, 'terrain.json')))
sensor_data = json.load(open(os.path.join(DATA_DIR, 'sensors.json')))
LOS = terrain_data['LOS']
sensor_list = [(sensor_data['params']['Z_SENSOR'], sensor_data['params']['H_SENSOR'])]

def build_state_grid(params):
    z_grid = np.linspace(0.0, params['Z_GOAL'], params['N_Z'])
    h_max = max(params['H_RIDGE'] * 2.0, params['H_GOAL'] + 50.0)
    h_grid = np.linspace(0.0, h_max, params['N_H'])
    if params['N_V'] == 1:
        v_grid = np.array([params['V_STALL']], dtype=float)
    else:
        v_grid = np.linspace(params['V_STALL'], params['V_MAX'], params['N_V'])
    n_gamma = int(np.round((params['GAMMA_MAX'] - params['GAMMA_MIN']) / params['D_GAMMA'])) + 1
    gamma_grid = np.linspace(params['GAMMA_MIN'], params['GAMMA_MAX'], n_gamma)
    return z_grid, h_grid, v_grid, gamma_grid

def h_terrain(z, params):
    z_ridge = params['Z_RIDGE']
    h_ridge = params['H_RIDGE']
    sigma = params['SIGMA_TERRAIN']
    return h_ridge * np.exp(-0.5 * ((z - z_ridge) / sigma) ** 2)

def glider_dynamics_deg(state, gamma_cmd_deg, params):
    import numpy as np
    def derivatives(s, gamma_rad):
        z, h, v, gamma = s
        dz = v * np.cos(gamma_rad)
        dh = v * np.sin(gamma_rad)
        D = 0.5 * params['RHO'] * v**2 * params['CD'] * params['S']
        dv = -D / params['M']
        return np.array([dz, dh, dv, 0.0], dtype=float)
    dt = params['DT_GLIDE']
    gamma_rad = np.radians(gamma_cmd_deg)
    s = np.array(state, dtype=float)
    k1 = derivatives(s, gamma_rad)
    k2 = derivatives(s + 0.5 * dt * k1, gamma_rad)
    k3 = derivatives(s + 0.5 * dt * k2, gamma_rad)
    k4 = derivatives(s + dt * k3, gamma_rad)
    s_next = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    s_next[3] = float(gamma_cmd_deg)
    s_next[2] = max(s_next[2], 0.0)
    s_next[1] = max(s_next[1], 0.0)
    return (float(s_next[0]), float(s_next[1]), float(s_next[2]), float(s_next[3]))

def find_grid_index(value, grid):
    return int(np.argmin(np.abs(grid - value)))

def terrain_valid(z, h, params):
    return h > h_terrain(z, params)

def terminal_state(z, h, params, tol=1e-1):
    return (z >= params['Z_GOAL']) and (h <= params['H_GOAL'] + tol)

def lambda_acoustic(z, h, v, z_sensor, h_sensor, params):
    r = np.sqrt((z - z_sensor)**2 + (h - h_sensor)**2)
    r_eff = np.maximum(r, params['R_FLOOR'])
    return float(params['LAM_ACOUSTIC'] * (v ** params['N_ACOUSTIC']) / (r_eff ** 2))

def lambda_doppler(z, h, v, gamma, z_sensor, h_sensor, params):
    dz = z_sensor - z
    dh = h_sensor - h
    r = np.sqrt(dz**2 + dh**2)
    if r < 1e-6:
        return 0.0
    gamma_rad = np.radians(gamma)
    v_z = v * np.cos(gamma_rad)
    v_h = v * np.sin(gamma_rad)
    los_z = dz / r
    los_h = dh / r
    v_r = v_z * los_z + v_h * los_h
    r_eff = np.maximum(r, params['R_FLOOR'])
    return float(params['LAM_DOPPLER'] * (v_r ** 2) / (r_eff ** 4))

def lambda_rcs(z, h, gamma, z_sensor, h_sensor, params):
    r = np.sqrt((z - z_sensor)**2 + (h - h_sensor)**2)
    r_eff = np.maximum(r, params['R_FLOOR'])
    gamma_min = params['GAMMA_MIN']
    cos_term = np.cos(np.radians(gamma - gamma_min))
    return float(params['LAM_RCS'] * params['SIGMA0'] * (cos_term ** 2) / (r_eff ** 4))

def lambda_total(z, h, v, gamma, sensor_list, phase, LOS, params):
    if z < LOS['z_tangent']:
        return 0.0
    lam = 0.0
    for z_s, h_s in sensor_list:
        if phase == 'powered':
            lam += lambda_acoustic(z, h, v, z_s, h_s, params)
        lam += lambda_doppler(z, h, v, gamma, z_s, h_s, params)
        lam += lambda_rcs(z, h, gamma, z_s, h_s, params)
    return float(lam)

def dp_backward(sensor_list, LOS, params):
    z_grid, h_grid, v_grid, gamma_grid = build_state_grid(params)
    N_Z, N_H, N_V, N_G = map(int, (len(z_grid), len(h_grid), len(v_grid), len(gamma_grid)))
    J = np.full((N_Z, N_H, N_V, N_G), np.inf, dtype=float)
    policy = np.full((N_Z, N_H, N_V, N_G), -1, dtype=int)
    dt = params['DT_GLIDE']
    inf = np.inf
    for iz, z in enumerate(z_grid):
        for ih, h in enumerate(h_grid):
            if not terrain_valid(z, h, params):
                continue
            for iv, v in enumerate(v_grid):
                for ig, gamma in enumerate(gamma_grid):
                    if terminal_state(z, h, params):
                        J[iz, ih, iv, ig] = 0.0
                        policy[iz, ih, iv, ig] = ig
    convergence = []
    max_iters = 150
    for _ in range(max_iters):
        J_old = J.copy()
        for iz, z in enumerate(z_grid):
            for ih, h in enumerate(h_grid):
                if not terrain_valid(z, h, params):
                    continue
                for iv, v in enumerate(v_grid):
                    for ig, gamma in enumerate(gamma_grid):
                        if terminal_state(z, h, params):
                            continue
                        best_cost = inf
                        best_action = -1
                        for action_idx, gamma_cmd in enumerate(gamma_grid):
                            next_state = glider_dynamics_deg((z, h, v, gamma), gamma_cmd, params)
                            z_n, h_n, v_n, gamma_n = next_state
                            if not terrain_valid(z_n, h_n, params):
                                continue
                            if z_n < 0.0 or h_n < 0.0 or v_n <= 0.0:
                                continue
                            iz_n = find_grid_index(z_n, z_grid)
                            ih_n = find_grid_index(h_n, h_grid)
                            iv_n = find_grid_index(v_n, v_grid)
                            ig_n = find_grid_index(gamma_n, gamma_grid)
                            stage_cost = 1.0 - np.exp(-lambda_total(z, h, v, gamma, sensor_list, 'glide', LOS, params) * dt)
                            cost = stage_cost + J[iz_n, ih_n, iv_n, ig_n]
                            if cost < best_cost:
                                best_cost = cost
                                best_action = action_idx
                        if best_action >= 0:
                            J[iz, ih, iv, ig] = best_cost
                            policy[iz, ih, iv, ig] = best_action
                        else:
                            J[iz, ih, iv, ig] = inf
                            policy[iz, ih, iv, ig] = -1
        delta = np.nanmax(np.abs(J - J_old))
        convergence.append(float(delta))
        if delta < 1e-3:
            break
    return z_grid, h_grid, v_grid, gamma_grid, J, policy, convergence

try:
    z_grid, h_grid, v_grid, gamma_grid, J, policy, convergence = dp_backward(sensor_list, LOS, params)
    print('dp computed', z_grid.shape, h_grid.shape, v_grid.shape, gamma_grid.shape)
except Exception as e:
    import traceback; traceback.print_exc()
    print('error', type(e).__name__, e)
