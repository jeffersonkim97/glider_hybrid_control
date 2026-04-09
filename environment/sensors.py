import numpy as np
from dataclasses import dataclass
from environment.geometry import Geometry

"""
Four sensor Probability of Detection models for A1b CUAS

Each sensor computes an instantaneous PoD rate lambda_k(z, h, v, gamma; z_d)

The combined detection probability over a trajectory is computed in game.py via Possion fusion (Zabarankin et al. 2006)
J_PD = 1 - exp(-integral(sum_k lambda_k(t) dt))

Sensors:
    - Acoustic: Engine noise, powered phase only, r^2 propagation
    - Doppler: Radial velocity, both phases, r^4 propagation
    - RCS: aspect-angle RCS, both phases, r^4 propagation
    - Camera: boresight angle, both phases, r^2 propagation
"""

@dataclass
class AcousticParams:
    kappa: float = 1e-4
    n: float = 6.0

@dataclass
class DopplerParams:
    kappa: float = 1e-6

@dataclass
class RCSParams:
    kappa: float = 1e-6
    sigma_psi_deg: float = 30.0

@dataclass
class CameraParams:
    kappa: float = 5e-5
    fov_half_deg: float = 45.0

class AcousticSensor:
    def __init__(self, params: AcousticParams=None):
        self.p = params or AcousticParams()

    def pod_rate(self, z, h, v, gamma, geom, phase):
        if phase != 1:
            return 0.0
        
        r = geom.range(z, h)
        if r < 1e-3:
            return 0.0
        
        return self.p.kappa * (v ** self.p.n) / (r ** 2)
    
class DopplerSensor:
    def __init__(self, params: DopplerParams=None):
        self.p = params or DopplerParams()

    def pod_rate(self, z, h, v, gamma, geom, phase):
        r = geom.range(z, h)
        if r < 1e-3:
            return 0.0
        
        v_r = geom.radial_velocity(z, h, v, gamma)
        return self.p.kappa * (v_r ** 2) / (r ** 4)
    
class RCSSensor:
    def __init__(self, params: RCSParams=None):
        self.p = params or RCSParams()
        self._sigma_psi_deg = np.radians(self.p.sigma_psi_deg)

    def rcs(self, psi_deg):
        psi_rad = np.radians(psi_deg)
        psi_ref = np.pi/2
        return np.exp(
            -((psi_rad - psi_ref) ** 2) / (2 * self._sigma_psi_deg ** 2)
        )

    def pod_rate(self, z, h, v, gamma, geom, phase):
        r = geom.range(z, h)
        if r < 1e-3:
            return 0.0
        
        psi_deg = geom.aspect_angle(z, h, gamma)
        sigma = self.rcs(psi_deg)
        return self.p.kappa * sigma / (r ** 4)
    
class CameraSensor:
    def __init__(self, params: CameraParams=None):
        self.p = params or CameraParams()
        self._fov_half_rad = np.radians(self.p.fov_half_deg)

    def pod_rate(self, z, h, v, gamma, geom, phase):
        r = geom.range(z, h)
        if r < 1e-3:
            return 0.0
        
        phi = geom.boresight_angle(z, h)
        
        if phi > self._fov_half_rad:
            return 0.0
        return self.p.kappa * (np.cos(phi)**2) / (r ** 2)
    
class SensorSuite:
    def __init__(self, z_d,
                 acoustic_params: AcousticParams=None,
                 doppler_params: DopplerParams=None,
                 rcs_params: RCSParams=None,
                 camera_params: CameraParams=None,
                 camera_elevation_deg: float=45.0):
        self.geom = Geometry(z_d, camera_elevation_deg)
        self.acoustic = AcousticSensor(acoustic_params)
        self.doppler = DopplerSensor(doppler_params)
        self.rcs = RCSSensor(rcs_params)
        self.camera = CameraSensor(camera_params)

    def individual_pod_rates(self, z, h, v, gamma, phase):
        return {
            'acoustic': self.acoustic.pod_rate(z, h, v, gamma, self.geom, phase),
            'doppler': self.doppler.pod_rate(z, h, v, gamma, self.geom, phase),
            'rcs': self.rcs.pod_rate(z, h, v, gamma, self.geom, phase),
            'camera': self.camera.pod_rate(z, h, v, gamma, self.geom, phase)
        }
    
    def total_pod_rate(self, z, h, v, gamma, phase):
        rates = self.individual_pod_rates(z, h, v, gamma, phase)
        return sum(rates.values())