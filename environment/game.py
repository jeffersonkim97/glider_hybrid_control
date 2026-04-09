import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from environment.vehicle import (Vehicle, VehicleParams, TrajectoryPoint)
from environment.sensors import (SensorSuite, AcousticParams, DopplerParams, RCSParams, CameraParams)

"""
Evaluate attacker and defender cost functions for the non-zero-sum Stackelberg game (Assume SSG)

Game Structure:
    Defender (leader): commits to a sensor placement z_d first
    Attacker (follower): observes z_d, chooses optimal strategy

Cost functions:
    J_A(a, d) = alpha_1 * J_PD(a, d) + alpha_2 * T_norm(a)
    J_D(a, d) = beta_1 * J_PD(a, d) - beta_2 * C(d)

where: 
    J_PD = probability of detection (depends on sensor placement and attack strategy)
    T_norm = normalized attack time (e.g., time to reach target)
    C(d) = coverage score for defender placement d

Solution via backward induction:
    Step 1: a*(d) = argmin_a J_A(a, d)
    Step 2: d* = argmax_d J_D(a*(d), d)
"""

@dataclass
class AttackerWeights:
    alpha_1: float = 0.7  # Weight for probability of detection
    alpha_2: float = 0.3  # Weight for attack time

@dataclass
class DefenderWeights:
    beta_1: float = 0.8  # Weight for probability of detection
    beta_2: float = 0.2  # Weight for coverage score

class Game:
    def __init__(self,
                 vehicle_params: VehicleParams = None,
                 attacker_weights: AttackerWeights = None,
                 defender_weights: DefenderWeights = None,
                 acoustic_params: AcousticParams = None,
                 doppler_params: DopplerParams = None,
                 rcs_params: RCSParams = None,
                 camera_params: CameraParams = None):
        
        self.vp = vehicle_params or VehicleParams()
        self.aw = attacker_weights or AttackerWeights()
        self.dw = defender_weights or DefenderWeights()
        self.vehicle = Vehicle(self.vp)

        self.acoustic_params = acoustic_params
        self.doppler_params = doppler_params
        self.rcs_params = rcs_params
        self.camera_params = camera_params

    def evaluate(self, h_dot, h_sw, gamma, z_d):
        trajectory = self.vehicle.simulate(h_dot, h_sw, gamma)
        arrays = self.vehicle.to_arrays(trajectory)

        # Find switch point
        switch_idx = next(i for i, p in enumerate(trajectory) if p.phase == 2)

        z_sw = trajectory[switch_idx].z
        delta = self.vehicle.compute_delta(z_sw, h_sw)

        suit = SensorSuite(
            z_d = z_d,
            acoustic_params = self.acoustic_params,
            doppler_params = self.doppler_params,
            rcs_params = self.rcs_params,
            camera_params = self.camera_params
        )

        pod_integral = self._integrate_pod(trajectory, suit)
        J_PD = 1.0 - np.exp(-pod_integral)  # Example: J_PD = 1 - exp(-integral)

        T = self.vehicle.total_time(trajectory)
        T_max = (self.vp.z_goal - self.vp.z_start) / self.vehicle.v_star
        T_norm = np.clip(T/T_max, 0, 1)

        Coverage = self._coverage_score(z_d)

        J_A = self.aw.alpha_1 * J_PD + self.aw.alpha_2 * T_norm
        J_D = self.dw.beta_1 * J_PD + self.dw.beta_2 * Coverage
        
        return {
            'trajectory': trajectory,
            'arrays': arrays,
            'J_PD': J_PD,
            'T': T,
            'T_norm': T_norm,
            'J_A': J_A,
            'J_D': J_D,
            'Coverage': Coverage,
            'pod_integral': pod_integral,
            'delta': delta
        }
    
    def _integrate_pod(self, trajectory, suite):
        if len(trajectory) < 2:
            return 0.0

        integral = 0.0
        prev_Rate = suite.total_pod_rate(
            trajectory[0].z, trajectory[0].h, trajectory[0].v, trajectory[0].gamma, trajectory[0].phase
        )

        for i in range(1, len(trajectory)):
            p = trajectory[i]
            curr_rate = suite.total_pod_rate(p.z, p.h, p.v, p.gamma, p.phase)
            integral += 0.5*(prev_Rate + curr_rate) * (p.t - trajectory[i-1].t)
            prev_Rate = curr_rate

        return integral

    def _coverage_score(self, z_d):
        z_mid = (self.vp.z_start + self.vp.z_goal) / 2
        z_range = (self.vp.z_goal - self.vp.z_start) / 2
        return float(np.clip(1.0 - abs(z_d - z_mid) / z_range, 0, 1))

    def get_individual_pod_arrays(self, trajectory, z_d):
        suite = SensorSuite(
            z_d = z_d,
            acoustic_params = self.acoustic_params,
            doppler_params = self.doppler_params,
            rcs_params = self.rcs_params,
            camera_params = self.camera_params
        )

        n = len(trajectory)
        rates = {
            'acoustic': np.zeros(n),
            'doppler': np.zeros(n),
            'rcs': np.zeros(n),
            'camera': np.zeros(n)
        }

        for i, p in enumerate(trajectory):
            individual = suite.individual_pod_rates(p.z, p.h, p.v, p.gamma, p.phase)
            for k in rates:
                rates[k][i] = individual[k]

        return rates

    def attacker_best_response(self, z_d, method='nelder-mead', verbose=False):
        from scipy.optimize import minimize

        # Derive bounds from vehicle param
        h_dot_min = 0.1
        h_dot_max = self.vp.h_dot_max
        h_sw_min = 50.0
        # h_sw_max = (self.vp.z_goal - self.vp.z_start) / self.vehicle.LD_max
        # h_sw_max = (self.vp.z_goal - self.vp.z_start) / (self.vehicle.LD_max + self.vp.v_climb / h_dot_min)
        h_sw_max = (self.vp.z_goal - self.vp.z_start) / (
                    self.vehicle.LD_max + self.vp.v_climb / h_dot_min
                    ) * 0.95
        gamma_min = self.vp.gamma_min
        gamma_max = self.vp.gamma_max

        def objective(x):
            h_dot  = np.clip(x[0], h_dot_min, h_dot_max)
            h_sw   = np.clip(x[1], h_sw_min,  h_sw_max)
            gamma  = np.clip(x[2], gamma_min,  gamma_max)

            # Recompute feasibility with clipped values
            t_climb = h_sw / h_dot
            z_sw    = self.vp.z_start + self.vp.v_climb * t_climb
            delta   = self.vehicle.compute_delta(z_sw, h_sw)
            if delta < 0:
                return 1.0

            try:
                result = self.evaluate(h_dot, h_sw, gamma, z_d)
                return result['J_A']
            except Exception:
                return 1.0
            
        # x0 = [
        #     (h_dot_min + h_dot_max) / 2,
        #     (h_sw_min + h_sw_max) / 2,
        #     (gamma_min + gamma_max) / 2
        # ]

        x0 = [
                (h_dot_min + h_dot_max) / 2.0,
                h_sw_max / 2.0,              # start at half of max feasible
                (gamma_min + gamma_max) / 2.0,
            ]

        res = minimize(
            objective, x0, method=method, options={'maxiter':20000, 'xatol':1e-4, 'fatol':1e-6, 'disp':verbose}
        )

        h_dot_opt = np.clip(res.x[0], h_dot_min, h_dot_max)
        h_sw_opt = np.clip(res.x[1], h_sw_min, h_sw_max)
        gamma_opt = np.clip(res.x[2], gamma_min, gamma_max)

        final = self.evaluate(h_dot_opt, h_sw_opt, gamma_opt, z_d)

        switch_idx = next(i for i, p in enumerate(final['trajectory']) if p.phase == 2)

        return {
            'h_dot': h_dot_opt,
            'h_sw': h_sw_opt,
            'gamma': gamma_opt,
            'z_sw': final['trajectory'][switch_idx].z,
            'J_A': final['J_A'],
            'J_PD': final['J_PD'],
            'T': final['T'],
            'delta': final['delta'],
            'success': res.success,
            'n_evals': res.nfev,
            'full_result': final
        }

    def defender_best_response(self, n_grid=50, corridor_margin=0.05, verbose=False):
        corridor = self.vp.z_goal - self.vp.z_start
        margin = corridor * corridor_margin
        z_d_min = self.vp.z_start + margin
        z_d_max = self.vp.z_goal - margin

        z_d_values = np.linspace(z_d_min, z_d_max, n_grid)

        results = []
        for i, z_d in enumerate(z_d_values):
            if verbose:
                print(f'[{i+1}/{n_grid}] z_d = {z_d:.0f}...', end=' ')
            
            abr = self.attacker_best_response(z_d, verbose=False)

            defender_eval = self.evaluate(
                abr['h_dot'], abr['h_sw'], abr['gamma'], z_d
            )
            J_D = defender_eval['J_D']

            if verbose:
                print(f'J_A = {abr["J_A"]:.4f}, J_D = {J_D:.4f}')

            results.append({
                'z_d': z_d,
                'J_D': J_D,
                'J_A': abr['J_A'],
                'J_PD': abr['J_PD'],
                'attacker_response': abr,
                'defender_eval': defender_eval
            })

        best = max(results, key=lambda r: r['J_D'])
        return {
            'z_d_star': best['z_d'],
            'J_D_star': best['J_D'],
            'all_results': results,
            'best': best
        }
    
    def attacker_best_response_piecewise(self, z_d: float,
                                      N: int = 5,
                                      method: str = 'nelder-mead',
                                      verbose: bool = False) -> dict:
        from scipy.optimize import minimize

        # Bounds — same as attacker_best_response
        h_dot_min = 0.1
        h_dot_max = self.vp.h_dot_max
        h_sw_min  = 50.0
        h_sw_max  = (self.vp.z_goal - self.vp.z_start) / self.vehicle.LD_max
        gamma_min = self.vp.gamma_min
        gamma_max = self.vp.gamma_max

        def make_policy(h_sw, gammas):
            """
            Returns a callable policy(z, h, gamma_prev, delta_rem, z_d)
            that divides the glide into N altitude segments and assigns
            one gamma per segment.

            Segment boundaries are evenly spaced in altitude from h_sw to 0.
            Segment i covers altitudes [h_sw*(N-i)/N, h_sw*(N-i+1)/N].
            """
            boundaries = np.linspace(h_sw, 0.0, N + 1)
            # boundaries[0] = h_sw (top), boundaries[N] = 0 (ground)
            # segment i: altitude in [boundaries[i+1], boundaries[i]]

            def policy(z, h, gamma_prev, delta_rem, z_d_ignored):
                # Find which segment we are in based on current altitude
                for i in range(N):
                    if h >= boundaries[i + 1]:
                        return gammas[i]
                return gammas[-1]    # below last boundary — use final gamma

            return policy

        def objective(x):
            h_dot  = np.clip(x[0], h_dot_min, h_dot_max)
            h_sw   = np.clip(x[1], h_sw_min,  h_sw_max)
            gammas = np.clip(x[2:], gamma_min, gamma_max)

            # Feasibility check
            t_climb = h_sw / h_dot
            z_sw    = self.vp.z_start + self.vp.v_climb * t_climb
            delta   = self.vehicle.compute_delta(z_sw, h_sw)
            if delta < 0:
                return 1.0

            try:
                policy    = make_policy(h_sw, gammas)
                trajectory = self.vehicle.simulate(h_dot, h_sw, policy)
                arrays    = self.vehicle.to_arrays(trajectory)

                suite = SensorSuite(
                    z_d=z_d,
                    acoustic_params=self.acoustic_params,
                    doppler_params=self.doppler_params,
                    rcs_params=self.rcs_params,
                    camera_params=self.camera_params,
                )

                pod_integral = self._integrate_pod(trajectory, suite)
                J_PD         = 1.0 - np.exp(-pod_integral)
                T            = self.vehicle.total_time(trajectory)
                T_max        = (self.vp.z_goal - self.vp.z_start) / self.vehicle.v_star
                T_norm       = np.clip(T / T_max, 0.0, 1.0)
                J_A          = self.aw.alpha_1 * J_PD + self.aw.alpha_2 * T_norm
                return J_A

            except Exception:
                return 1.0

        # Initial guess: same h_dot and h_sw as scalar BR,
        # all gammas initialized to midpoint
        gamma_mid = (gamma_min + gamma_max) / 2.0
        x0 = np.array([
            (h_dot_min + h_dot_max) / 2.0,
            (h_sw_min  + h_sw_max)  / 2.0,
            *([gamma_mid] * N)
        ])

        res = minimize(
            objective, x0, method=method,
            options={'maxiter': 5000, 'xatol': 1e-4,
                    'fatol': 1e-6, 'disp': verbose}
        )

        h_dot_opt  = np.clip(res.x[0], h_dot_min, h_dot_max)
        h_sw_opt   = np.clip(res.x[1], h_sw_min,  h_sw_max)
        gammas_opt = np.clip(res.x[2:], gamma_min, gamma_max)

        # Build final trajectory with optimal piecewise policy
        policy_opt = make_policy(h_sw_opt, gammas_opt)
        traj_opt   = self.vehicle.simulate(h_dot_opt, h_sw_opt, policy_opt)
        arrays_opt = self.vehicle.to_arrays(traj_opt)

        switch_idx = next(i for i, p in enumerate(traj_opt) if p.phase == 2)

        suite_final = SensorSuite(
            z_d=z_d,
            acoustic_params=self.acoustic_params,
            doppler_params=self.doppler_params,
            rcs_params=self.rcs_params,
            camera_params=self.camera_params,
        )
        pod_integral_final = self._integrate_pod(traj_opt, suite_final)
        J_PD_final = 1.0 - np.exp(-pod_integral_final)
        T_final    = self.vehicle.total_time(traj_opt)
        T_max      = (self.vp.z_goal - self.vp.z_start) / self.vehicle.v_star
        T_norm_final = np.clip(T_final / T_max, 0.0, 1.0)
        Coverage   = self._coverage_score(z_d)
        J_A_final  = self.aw.alpha_1 * J_PD_final + self.aw.alpha_2 * T_norm_final
        J_D_final  = self.dw.beta_1  * J_PD_final + self.dw.beta_2  * Coverage

        full_result = {
            'trajectory': traj_opt,
            'arrays':     arrays_opt,
            'J_PD':       J_PD_final,
            'T':          T_final,
            'T_norm':     T_norm_final,
            'J_A':        J_A_final,
            'J_D':        J_D_final,
            'Coverage':   Coverage,
            'pod_integral': pod_integral_final,
            'delta':      self.vehicle.compute_delta(traj_opt[switch_idx].z, h_sw_opt),
        }

        return {
            'h_dot':      h_dot_opt,
            'h_sw':       h_sw_opt,
            'gamma':     gammas_opt,      # array of N gamma values
            'z_sw':       traj_opt[switch_idx].z,
            'J_A':        J_A_final,
            'J_PD':       J_PD_final,
            'T':          T_final,
            'delta':      full_result['delta'],
            'N':          N,
            'success':    res.success,
            'n_evals':    res.nfev,
            'full_result': full_result,
        }