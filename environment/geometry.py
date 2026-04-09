import numpy as np

"""
All spatial computations shared acrossed every sensor model done in this file.

Coordinate system:
    z: longitudinal axis, positive forward, from z_start to z_goal
    h: altitude, positive upward, h=0 at ground level

Sensor placement:
    All sensors are collocated at (z_d, 0) on the ground. z_d is the defender's decision variable

Sign convention for glide angle:
    gamma < 0 for descent

"""

class Geometry:
    def __init__(self, z_d, camera_elevation_deg=45.0):
        self.z_d = z_d
        self.camera_elevation = np.radians(camera_elevation_deg)
        self.b_hat = np.array([-np.cos(self.camera_elevation),
                                np.sin(self.camera_elevation)]) # unit vector from sensor to target
        
    def range(self, z, h):
        return np.sqrt((z - self.z_d)**2 + h**2)
    
    def los_angle(self, z, h):
        dz = abs(self.z_d - z)
        if dz < 1e-6:
            return np.pi/2
        return np.arctan2(h, dz)
    
    def radial_velocity(self, z, h, v, gamma):
        theta = self.los_angle(z, h)
        return v*np.cos(theta - abs(gamma))
    
    def aspect_angle(self, z, h, gamma):
        theta = self.los_angle(z, h)
        ventral_angle = np.pi/2 + gamma
        psi = abs(ventral_angle - theta)
        return np.degrees(psi)
    
    def boresight_angle(self, z, h):
        u = np.array([z - self.z_d, h]) # vector from sensor to target
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            return 0.0
        u_hat = u / u_norm
        dot = np.clip(np.dot(self.b_hat, u_hat), -1.0, 1.0)
        return np.arccos(dot)
    
    def all(self, z, h, v, gamma):
        return {
            'r': self.range(z, h),
            'theta': self.los_angle(z, h),
            'v_r': self.radial_velocity(z, h, v, gamma),
            'psi': self.aspect_angle(z, h, gamma),
            'phi': self.boresight_angle(z, h)
        }