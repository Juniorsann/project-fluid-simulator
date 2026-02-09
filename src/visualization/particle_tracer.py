"""
Particle Tracer module for Lagrangian particle tracking in CFD simulations.

This module implements massless particles that follow the velocity field
using 4th-order Runge-Kutta (RK4) integration.
"""

from typing import Optional, Tuple, Union, Literal
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ParticleTracer:
    """
    Simulates massless particles that follow the fluid velocity field.
    Uses Runge-Kutta 4th order (RK4) integration for accurate tracking.
    """
    
    def __init__(
        self,
        velocity_field: np.ndarray,
        grid_coordinates: Tuple[np.ndarray, np.ndarray],
        n_particles: int = 1000,
        domain_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    ):
        """
        Initialize particle tracer.
        
        Args:
            velocity_field: Velocity field as (u, v) tuple of 2D arrays or single 2D array
            grid_coordinates: Tuple of (z, r) coordinate arrays for the grid
            n_particles: Number of tracer particles
            domain_bounds: Optional domain bounds ((z_min, z_max), (r_min, r_max))
        """
        self.n_particles = n_particles
        self.grid_z, self.grid_r = grid_coordinates
        
        # Handle velocity field format
        if isinstance(velocity_field, tuple):
            self.u_field, self.v_field = velocity_field
        else:
            self.u_field = velocity_field
            self.v_field = np.zeros_like(velocity_field)
        
        # Set domain bounds
        if domain_bounds is None:
            self.z_bounds = (self.grid_z.min(), self.grid_z.max())
            self.r_bounds = (self.grid_r.min(), self.grid_r.max())
        else:
            self.z_bounds, self.r_bounds = domain_bounds
        
        # Initialize particle positions (will be set by inject_particles)
        self.positions = None
        self.active = None
        
        # Create interpolators for velocity field
        self._setup_interpolators()
    
    def _setup_interpolators(self):
        """Setup interpolators for velocity components."""
        # Create interpolator for axial velocity (u)
        self.u_interpolator = RegularGridInterpolator(
            (self.grid_r, self.grid_z),
            self.u_field,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Create interpolator for radial velocity (v)
        self.v_interpolator = RegularGridInterpolator(
            (self.grid_r, self.grid_z),
            self.v_field,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
    
    def inject_particles(
        self,
        region: Literal['inlet', 'uniform', 'line', 'circle'] = 'inlet',
        distribution: Literal['random', 'uniform', 'grid'] = 'random',
        custom_positions: Optional[np.ndarray] = None
    ):
        """
        Inject particles in the specified region.
        
        Args:
            region: Injection region type
                - 'inlet': At the inlet (z_min)
                - 'uniform': Uniformly throughout domain
                - 'line': Along a line
                - 'circle': In a circle
            distribution: Distribution type within region
                - 'random': Random positions
                - 'uniform': Evenly spaced
                - 'grid': Grid pattern
            custom_positions: Optional array of custom positions (n_particles, 2)
        """
        if custom_positions is not None:
            self.positions = custom_positions
            self.n_particles = len(custom_positions)
        else:
            if region == 'inlet':
                # Particles at inlet (z_min) with varying radial positions
                z_pos = np.full(self.n_particles, self.z_bounds[0])
                
                if distribution == 'random':
                    r_pos = np.random.uniform(
                        self.r_bounds[0],
                        self.r_bounds[1],
                        self.n_particles
                    )
                elif distribution == 'uniform' or distribution == 'grid':
                    r_pos = np.linspace(
                        self.r_bounds[0],
                        self.r_bounds[1],
                        self.n_particles
                    )
                
                self.positions = np.column_stack([z_pos, r_pos])
            
            elif region == 'uniform':
                # Particles uniformly distributed in domain
                if distribution == 'random':
                    z_pos = np.random.uniform(
                        self.z_bounds[0],
                        self.z_bounds[1],
                        self.n_particles
                    )
                    r_pos = np.random.uniform(
                        self.r_bounds[0],
                        self.r_bounds[1],
                        self.n_particles
                    )
                elif distribution == 'grid':
                    n_z = int(np.sqrt(self.n_particles))
                    n_r = self.n_particles // n_z
                    z_grid = np.linspace(self.z_bounds[0], self.z_bounds[1], n_z)
                    r_grid = np.linspace(self.r_bounds[0], self.r_bounds[1], n_r)
                    Z, R = np.meshgrid(z_grid, r_grid)
                    z_pos = Z.ravel()[:self.n_particles]
                    r_pos = R.ravel()[:self.n_particles]
                else:  # uniform
                    z_pos = np.random.uniform(
                        self.z_bounds[0],
                        self.z_bounds[1],
                        self.n_particles
                    )
                    r_pos = np.random.uniform(
                        self.r_bounds[0],
                        self.r_bounds[1],
                        self.n_particles
                    )
                
                self.positions = np.column_stack([z_pos, r_pos])
            
            elif region == 'line':
                # Particles along a line at inlet
                z_pos = np.linspace(self.z_bounds[0], self.z_bounds[1], self.n_particles)
                r_pos = np.full(self.n_particles, (self.r_bounds[0] + self.r_bounds[1]) / 2)
                self.positions = np.column_stack([z_pos, r_pos])
            
            elif region == 'circle':
                # Particles in a circle at inlet
                theta = np.linspace(0, 2*np.pi, self.n_particles, endpoint=False)
                r_circle = (self.r_bounds[1] - self.r_bounds[0]) / 2
                r_center = (self.r_bounds[0] + self.r_bounds[1]) / 2
                z_pos = np.full(self.n_particles, self.z_bounds[0])
                r_pos = r_center + r_circle * 0.8 * np.cos(theta)  # 80% of radius
                self.positions = np.column_stack([z_pos, r_pos])
        
        # All particles start active
        self.active = np.ones(self.n_particles, dtype=bool)
    
    def _get_velocity(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get velocity at particle positions using interpolation.
        
        Args:
            positions: Particle positions (n, 2) as (z, r)
        
        Returns:
            Tuple of (u, v) velocity components
        """
        # Swap to (r, z) for interpolator
        points = np.column_stack([positions[:, 1], positions[:, 0]])
        
        u = self.u_interpolator(points)
        v = self.v_interpolator(points)
        
        return u, v
    
    def advect(self, dt: float):
        """
        Advect particles using 4th-order Runge-Kutta (RK4) integration.
        
        Integration scheme:
        k1 = v(x_n)
        k2 = v(x_n + 0.5*dt*k1)
        k3 = v(x_n + 0.5*dt*k2)
        k4 = v(x_n + dt*k3)
        x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        Args:
            dt: Time step
        """
        if self.positions is None:
            raise ValueError("Particles not initialized. Call inject_particles() first.")
        
        # Only advect active particles
        active_positions = self.positions[self.active]
        
        if len(active_positions) == 0:
            return
        
        # RK4 integration
        # k1
        u1, v1 = self._get_velocity(active_positions)
        k1 = np.column_stack([u1, v1])
        
        # k2
        pos2 = active_positions + 0.5 * dt * k1
        u2, v2 = self._get_velocity(pos2)
        k2 = np.column_stack([u2, v2])
        
        # k3
        pos3 = active_positions + 0.5 * dt * k2
        u3, v3 = self._get_velocity(pos3)
        k3 = np.column_stack([u3, v3])
        
        # k4
        pos4 = active_positions + dt * k3
        u4, v4 = self._get_velocity(pos4)
        k4 = np.column_stack([u4, v4])
        
        # Update positions
        new_positions = active_positions + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Update active particle positions
        self.positions[self.active] = new_positions
        
        # Remove particles that left the domain
        self._remove_out_of_bounds()
    
    def _remove_out_of_bounds(self):
        """Remove particles that have left the computational domain."""
        z_pos = self.positions[self.active, 0]
        r_pos = self.positions[self.active, 1]
        
        # Check bounds
        in_bounds = (
            (z_pos >= self.z_bounds[0]) &
            (z_pos <= self.z_bounds[1]) &
            (r_pos >= self.r_bounds[0]) &
            (r_pos <= self.r_bounds[1])
        )
        
        # Update active status
        active_indices = np.where(self.active)[0]
        self.active[active_indices[~in_bounds]] = False
    
    def get_particle_properties(
        self,
        property_field: np.ndarray,
        property_name: str = 'property'
    ) -> np.ndarray:
        """
        Get interpolated property values at particle positions.
        
        Args:
            property_field: 2D field of property values on grid
            property_name: Name of the property (for reference)
        
        Returns:
            Array of property values at active particle positions
        """
        if self.positions is None:
            raise ValueError("Particles not initialized. Call inject_particles() first.")
        
        # Create interpolator for property
        interpolator = RegularGridInterpolator(
            (self.grid_r, self.grid_z),
            property_field,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Get property at active particle positions
        active_positions = self.positions[self.active]
        points = np.column_stack([active_positions[:, 1], active_positions[:, 0]])
        
        return interpolator(points)
    
    def get_active_positions(self) -> np.ndarray:
        """
        Get positions of active particles.
        
        Returns:
            Array of active particle positions (n_active, 2)
        """
        if self.positions is None:
            return np.array([])
        return self.positions[self.active]
    
    def get_particle_count(self) -> dict:
        """
        Get particle statistics.
        
        Returns:
            Dictionary with particle counts
        """
        if self.positions is None:
            return {'total': 0, 'active': 0, 'inactive': 0}
        
        n_active = np.sum(self.active)
        return {
            'total': self.n_particles,
            'active': n_active,
            'inactive': self.n_particles - n_active
        }
