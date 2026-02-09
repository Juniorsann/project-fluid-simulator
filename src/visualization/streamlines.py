"""
Streamline generation module for flow visualization.

This module provides classes and functions for generating streamlines, pathlines,
and streaklines from velocity fields. It also includes vorticity computation and
vortex identification methods.
"""

from typing import Optional, Tuple, List
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class StreamlineGenerator:
    """
    Generator for streamlines, pathlines, and streaklines.
    
    Streamlines: Instantaneous flow lines tangent to velocity field
    Pathlines: Trajectories of individual particles over time
    Streaklines: Locus of particles released from a point over time
    """
    
    def __init__(
        self,
        velocity_field: Tuple[np.ndarray, np.ndarray],
        grid: Tuple[np.ndarray, np.ndarray]
    ):
        """
        Initialize streamline generator with velocity field.
        
        Args:
            velocity_field: Tuple of (u_r, u_z) velocity components [m/s]
            grid: Tuple of (r, z) grid coordinates [m]
        """
        self.u_r, self.u_z = velocity_field
        self.r_grid, self.z_grid = grid
        
        # Domain bounds
        self.r_min, self.r_max = self.r_grid.min(), self.r_grid.max()
        self.z_min, self.z_max = self.z_grid.min(), self.z_grid.max()
        
        # Create interpolators
        self._setup_interpolators()
    
    def _setup_interpolators(self):
        """Set up interpolators for velocity field."""
        points = (self.r_grid, self.z_grid)
        
        self.u_r_interp = RegularGridInterpolator(
            points, self.u_r, bounds_error=False, fill_value=0.0
        )
        self.u_z_interp = RegularGridInterpolator(
            points, self.u_z, bounds_error=False, fill_value=0.0
        )
    
    def generate_streamlines(
        self,
        seed_points: np.ndarray,
        max_length: float = 100.0,
        step_size: float = 0.01,
        max_steps: int = 10000
    ) -> List[np.ndarray]:
        """
        Generate streamlines from seed points.
        
        Args:
            seed_points: Array of (r, z) seed positions, shape (n_seeds, 2)
            max_length: Maximum streamline length [m]
            step_size: Integration step size [m]
            max_steps: Maximum number of integration steps
            
        Returns:
            List of streamline arrays, each shape (n_points, 2)
        """
        streamlines = []
        
        for seed in seed_points:
            streamline = self._integrate_streamline(
                seed, max_length, step_size, max_steps
            )
            if len(streamline) > 1:
                streamlines.append(streamline)
        
        return streamlines
    
    def _integrate_streamline(
        self,
        seed: np.ndarray,
        max_length: float,
        step_size: float,
        max_steps: int
    ) -> np.ndarray:
        """
        Integrate a single streamline using RK4.
        
        Args:
            seed: Starting position (r, z)
            max_length: Maximum length
            step_size: Step size
            max_steps: Maximum steps
            
        Returns:
            Streamline points array, shape (n_points, 2)
        """
        points = [seed.copy()]
        current_pos = seed.copy()
        total_length = 0.0
        
        for _ in range(max_steps):
            # RK4 integration
            k1 = self._get_velocity(current_pos[0], current_pos[1])
            
            # Normalize velocity for constant step size
            k1_mag = np.sqrt(k1[0]**2 + k1[1]**2)
            if k1_mag < 1e-10:
                break
            k1 = np.array(k1) / k1_mag * step_size
            
            mid1 = current_pos + 0.5 * k1
            k2 = self._get_velocity(mid1[0], mid1[1])
            k2_mag = np.sqrt(k2[0]**2 + k2[1]**2)
            if k2_mag < 1e-10:
                break
            k2 = np.array(k2) / k2_mag * step_size
            
            mid2 = current_pos + 0.5 * k2
            k3 = self._get_velocity(mid2[0], mid2[1])
            k3_mag = np.sqrt(k3[0]**2 + k3[1]**2)
            if k3_mag < 1e-10:
                break
            k3 = np.array(k3) / k3_mag * step_size
            
            end = current_pos + k3
            k4 = self._get_velocity(end[0], end[1])
            k4_mag = np.sqrt(k4[0]**2 + k4[1]**2)
            if k4_mag < 1e-10:
                break
            k4 = np.array(k4) / k4_mag * step_size
            
            # Update position
            new_pos = current_pos + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            
            # Check bounds
            if not self._is_in_domain(new_pos[0], new_pos[1]):
                break
            
            # Update tracking
            step_length = np.linalg.norm(new_pos - current_pos)
            total_length += step_length
            
            points.append(new_pos.copy())
            current_pos = new_pos
            
            if total_length >= max_length:
                break
        
        return np.array(points)
    
    def generate_pathlines(
        self,
        seed_points: np.ndarray,
        time_steps: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate pathlines (particle trajectories in unsteady flow).
        
        For steady flow, pathlines are identical to streamlines.
        
        Args:
            seed_points: Array of (r, z) seed positions, shape (n_seeds, 2)
            time_steps: Array of time values [s]
            
        Returns:
            List of pathline arrays
        """
        # For steady flow, pathlines = streamlines
        # This is a simplified implementation
        pathlines = []
        
        for seed in seed_points:
            pathline = [seed.copy()]
            current_pos = seed.copy()
            
            for i in range(1, len(time_steps)):
                dt = time_steps[i] - time_steps[i-1]
                
                # Simple Euler integration (could use RK4)
                vel = self._get_velocity(current_pos[0], current_pos[1])
                new_pos = current_pos + np.array(vel) * dt
                
                if not self._is_in_domain(new_pos[0], new_pos[1]):
                    break
                
                pathline.append(new_pos.copy())
                current_pos = new_pos
            
            if len(pathline) > 1:
                pathlines.append(np.array(pathline))
        
        return pathlines
    
    def generate_streaklines(
        self,
        seed_point: np.ndarray,
        release_times: np.ndarray
    ) -> np.ndarray:
        """
        Generate streaklines (locus of particles released from a point).
        
        For steady flow, streaklines are identical to streamlines.
        
        Args:
            seed_point: Release point (r, z)
            release_times: Times at which particles are released [s]
            
        Returns:
            Streakline points array
        """
        # For steady flow, streakline = streamline
        # This is a simplified implementation
        streamline = self._integrate_streamline(
            seed_point, max_length=100.0, step_size=0.01, max_steps=10000
        )
        return streamline
    
    def compute_vorticity(self) -> np.ndarray:
        """
        Compute vorticity field (curl of velocity).
        
        For 2D axisymmetric flow in (r,z) coordinates:
        ω_θ = ∂u_z/∂r - ∂u_r/∂z
        
        Returns:
            Vorticity field (azimuthal component)
        """
        # Compute gradients
        dr = np.diff(self.r_grid)[0] if len(self.r_grid) > 1 else 1.0
        dz = np.diff(self.z_grid)[0] if len(self.z_grid) > 1 else 1.0
        
        # ∂u_z/∂r
        du_z_dr = np.gradient(self.u_z, dr, axis=0)
        
        # ∂u_r/∂z
        du_r_dz = np.gradient(self.u_r, dz, axis=1)
        
        # Vorticity (azimuthal component)
        vorticity = du_z_dr - du_r_dz
        
        return vorticity
    
    def identify_vortices(
        self,
        method: str = 'q_criterion',
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Identify vortex cores using Q-criterion or lambda2 method.
        
        Args:
            method: Method to use ('q_criterion' or 'lambda2')
            threshold: Threshold value for vortex identification
            
        Returns:
            Field indicating vortex regions
        """
        if method == 'q_criterion':
            # Q-criterion: Q = 0.5 * (Ω² - S²)
            # where Ω is rotation rate and S is strain rate
            vorticity = self.compute_vorticity()
            
            # For 2D flow, simplified Q-criterion
            Q = 0.5 * vorticity**2
            
            if threshold is None:
                threshold = 0.1 * np.max(np.abs(Q))
            
            vortex_mask = Q > threshold
            
        elif method == 'lambda2':
            # Lambda2 method (simplified for 2D)
            # This is a placeholder - full implementation requires
            # eigenvalue analysis of velocity gradient tensor
            vorticity = self.compute_vorticity()
            
            if threshold is None:
                threshold = 0.1 * np.max(np.abs(vorticity))
            
            vortex_mask = np.abs(vorticity) > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return vortex_mask.astype(float)
    
    def _get_velocity(self, r: float, z: float) -> Tuple[float, float]:
        """
        Get velocity at arbitrary position.
        
        Args:
            r: Radial position [m]
            z: Axial position [m]
            
        Returns:
            Tuple of (u_r, u_z) velocity components [m/s]
        """
        pos = np.array([[r, z]])
        u_r = float(self.u_r_interp(pos)[0])
        u_z = float(self.u_z_interp(pos)[0])
        return (u_r, u_z)
    
    def _is_in_domain(self, r: float, z: float) -> bool:
        """
        Check if position is within domain.
        
        Args:
            r: Radial position [m]
            z: Axial position [m]
            
        Returns:
            True if in domain
        """
        return (self.r_min <= r <= self.r_max and 
                self.z_min <= z <= self.z_max)
