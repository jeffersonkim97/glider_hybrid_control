import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class AirframeParams:
    C_D0:   float = 0.025            # Zero-lift drag coefficient
    k:      float = 0.045             # Induced drag factor
    W:      float = 150.0           # Weight (N)
    rho:    float = 1.225            # Air density (kg/m^3)
    S:      float = 0.8              # Wing area (m^2)

@dataclass
class VehicleParams:
    v_climb: float = 25.0                # Climb speed (m/s)
    h_dot_max: float = 5.0                   # Max climb rate (m/s)
    gamma_min: float = -np.pi/4             # Minimum climb angle (radians)
    gamma_max: float = -0.035            # Maximum climb angle (r
    z_start: float = 0.0                   # Starting altitude (m)
    z_goal: float = 50000.0                    # Goal altitude (m)
    dt: float = 0.5                        # Time step (s)
    airframe: AirframeParams = field(default_factory=AirframeParams)

@dataclass
class TrajectoryPoint:
    t: float
    z: float
    h: float
    v: float
    gamma: float
    phase: int
    delta_remaining: float = 0

class Vehicle:
    def __init__(self, params: VehicleParams=None):
        self.p = params or VehicleParams()
        self._precompute_polar()

    def _precompute_polar(self):
        af = self.p.airframe

        self.CL_star = np.sqrt(af.C_D0 / af.k)
        self.CD_star = af.C_D0 + af.k * self.CL_star**2
        self.LD_max = self.CL_star / self.CD_star
        self.gamma_star = -np.arctan(1.0 / self.LD_max)
        self.v_star = np.sqrt(
            2.0*af.W * np.cos(abs(self.gamma_star)) / (af.rho * af.S * self.CL_star)
        )

    def cl_from_gamma(self, gamma):
        gamma = np.clip(gamma, self.p.gamma_min, self.p.gamma_max)
        af = self.p.airframe

        tan_g = np.tan(abs(gamma))
        discriminant = tan_g**2 - 4.0 * af.k * af.C_D0

        if discriminant < 0:
            return self.CL_star
        CL = (tan_g - np.sqrt(discriminant)) / (2.0 * af.k)
        return max(CL, 0.01)

    def glide_speed(self, gamma):
        gamma = np.clip(gamma, self.p.gamma_min, self.p.gamma_max)
        af = self.p.airframe
        CL = self.cl_from_gamma(gamma)
        return np.sqrt(2.0*af.W * np.cos(abs(gamma)) / (af.rho * af.S * CL))

    def compute_delta(self, z_sw, h_sw):
        return z_sw + h_sw * self.LD_max - self.p.z_goal

    def _delta_decrement(self, gamma, h_dot, dt):
        cot_star = self.LD_max
        cot_gamma = 1 / np.tan(abs(gamma))
        return (cot_star - cot_gamma) * abs(h_dot) * dt

    def simulate_powered(self, h_dot, h_sw):
        h_dot = np.clip(h_dot, 0.1, self.p.h_dot_max)
        
        # z_sw is determined by geometry of climb
        t_climb = h_sw / h_dot
        z_sw = self.p.z_start + self.p.v_climb * t_climb

        # Feasibility check
        delta = self.compute_delta(z_sw, h_sw)
        if delta < 0:
            raise ValueError(
                f'Infeasible: Delta={delta:.1f} m,'
                f'Cannot reach z_goal from (z_sw={z_sw:.1f} m, h_sw={h_sw:.1f} m)'
                f'Increase h_sw or reduce h_dot'
            )
        
        gamma_climb = np.arctan2(h_dot, self.p.v_climb)

        trajectory = []
        t, z, h = 0.0, self.p.z_start, 0.0

        while h < h_sw -1e-3 and z < self.p.z_goal:
            trajectory.append(
                TrajectoryPoint(t=t, z=z, h=h, v=self.p.v_climb, gamma=gamma_climb, phase=1)
            )
            z += self.p.v_climb * self.p.dt
            h += h_dot * self.p.dt
            t += self.p.dt

        # Append exact switch point
        trajectory.append(
            TrajectoryPoint(t=t, z=z_sw, h=h_sw, v=self.p.v_climb, gamma=gamma_climb, phase=1)
        )
        return trajectory

    # Fixed gamma for optimal
    # def simulate_glide(self, z_sw, h_sw, gamma, delta = None, t_start=0.0):
    #     if delta is None:
    #         delta = self.compute_delta(z_sw, h_sw)

    #     gamma = np.clip(gamma, self.p.gamma_min, self.p.gamma_max)
    #     v = self.glide_speed(gamma)
    #     z_dot = v * np.cos(abs(gamma))
    #     h_dot = v * np.sin(gamma)
        
    #     trajectory = []
    #     t = t_start
    #     z, h = z_sw, h_sw
    #     delta_rem = delta

    #     while h > 0 and z < self.p.z_goal:
    #         trajectory.append(
    #             TrajectoryPoint(t=t, z=z, h=h, v=v, gamma=gamma, phase=2, delta_remaining=delta_rem)
    #         )
            
    #         decrement = self._delta_decrement(gamma, h_dot, self.p.dt)
    #         delta_rem -= decrement

    #         if delta_rem <= 0:
    #             delta_rem = 0
    #             gamma = self.gamma_star
    #             v = self.v_star
    #             z_dot = v*np.cos(abs(gamma))
    #             h_dot = v*np.sin(gamma)

    #         z += z_dot * self.p.dt
    #         h += h_dot * self.p.dt
    #         t += self.p.dt

    #     trajectory.append(
    #         TrajectoryPoint(t=t, z=z, h=max(h, 0), v=v, gamma=gamma, phase=2, delta_remaining=max(delta_rem, 0))
    #     )
    #     return trajectory

    def simulate_glide(self, z_sw: float, h_sw: float,
                    policy,
                    t_start: float = 0.0,
                    delta: float = None) -> List[TrajectoryPoint]:
        if delta is None:
            delta = self.compute_delta(z_sw, h_sw)

        # Handle backward compatibility: if policy is a scalar, wrap it
        if not callable(policy):
            gamma_fixed = float(policy)
            policy = lambda z, h, gamma_prev, delta_rem, z_d: gamma_fixed

        trajectory = []
        t         = t_start
        z, h      = z_sw, h_sw
        delta_rem = delta
        gamma     = self.gamma_star    # initial gamma before first policy call

        while h > 0.0 and z < self.p.z_goal:

            # Policy chooses gamma at this timestep given current state
            gamma = np.clip(
                policy(z, h, gamma, delta_rem, getattr(self, '_z_d', None)),
                self.p.gamma_min,
                self.p.gamma_max
            )

            # Budget enforcement: force best glide if budget exhausted
            if delta_rem <= 0.0:
                delta_rem = 0.0
                gamma     = self.gamma_star

            v     = self.glide_speed(gamma)
            z_dot = v * np.cos(abs(gamma))
            h_dot = v * np.sin(gamma)          # negative

            trajectory.append(TrajectoryPoint(
                t=t, z=z, h=h, v=v,
                gamma=gamma, phase=2,
                delta_remaining=delta_rem
            ))

            decrement  = self._delta_decrement(gamma, h_dot, self.p.dt)
            delta_rem -= decrement

            z += z_dot * self.p.dt
            h += h_dot * self.p.dt
            t += self.p.dt

        # Final point
        trajectory.append(TrajectoryPoint(
            t=t, z=z, h=max(h, 0.0), v=self.glide_speed(gamma),
            gamma=gamma, phase=2,
            delta_remaining=max(delta_rem, 0.0)
        ))

        return trajectory

    def simulate(self, h_dot, h_sw, gamma):
        phase1 = self.simulate_powered(h_dot, h_sw)
        z_sw = phase1[-1].z
        t_sw = phase1[-1].t
        delta = self.compute_delta(z_sw, h_sw)
        # phase2 = self.simulate_glide(z_sw, h_sw, gamma, delta, t_start=t_sw)
        phase2 = self.simulate_glide(z_sw, h_sw, gamma, t_start=t_sw, delta=delta)

        return phase1 + phase2

    def total_time(self, trajectory):
        return trajectory[-1].t - trajectory[0].t

    def to_arrays(self, trajectory):
        return {
            't': np.array([p.t for p in trajectory]),
            'z': np.array([p.z for p in trajectory]),
            'h': np.array([p.h for p in trajectory]),
            'v': np.array([p.v for p in trajectory]),
            'gamma': np.array([p.gamma for p in trajectory]),
            'phase': np.array([p.phase for p in trajectory]),
            'delta_remaining': np.array([p.delta_remaining for p in trajectory])
        }