"""
Particle tracer module for visualizing flow paths.

This module implements a particle tracing system that follows the velocity field
using Runge-Kutta 4th order integration. Particles can be colored by velocity,
temperature, or residence time.
"""

from typing import Optional, Tuple, List
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ParticleTracer:
    """
    Particle tracer that follows velocity field using RK4 integration.
    
    Particles are advected through the flow field and can be used to
    visualize flow patterns and fluid trajectories.
    """
    
    def __init__(
        self,
        velocity_field: Tuple[np.ndarray, np.ndarray],
        grid: Tuple[np.ndarray, np.ndarray],
        temperature_field: Optional[np.ndarray] = None,
        domain_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    ):
        """
        Initialize particle tracer with velocity and optional temperature fields.
        
        Args:
            velocity_field: Tuple of (u_r, u_z) velocity components [m/s]
            grid: Tuple of (r, z) grid coordinates [m]
            temperature_field: Optional temperature field [K]
            domain_bounds: Optional domain bounds ((r_min, r_max), (z_min, z_max))
        """
        self.u_r, self.u_z = velocity_field
        self.r_grid, self.z_grid = grid
        self.temperature_field = temperature_field
        
        # Set domain bounds
        if domain_bounds is None:
            self.r_min, self.r_max = self.r_grid.min(), self.r_grid.max()
            self.z_min, self.z_max = self.z_grid.min(), self.z_grid.max()
        else:
            (self.r_min, self.r_max), (self.z_min, self.z_max) = domain_bounds
        
        # Initialize particle storage
        self.positions = []  # List of (r, z) positions
        self.velocities = []  # List of velocity magnitudes
        self.temperatures = []  # List of temperatures (if available)
        self.ages = []  # List of particle ages
        self.active = []  # List of active flags
        
        # Create interpolators for velocity components
        self._setup_interpolators()
        
    def _setup_interpolators(self):
        """Set up interpolators for velocity and temperature fields."""
        # Create 2D interpolators
        points = (self.r_grid, self.z_grid)
        
        self.u_r_interp = RegularGridInterpolator(
            points, self.u_r, bounds_error=False, fill_value=0.0
        )
        self.u_z_interp = RegularGridInterpolator(
            points, self.u_z, bounds_error=False, fill_value=0.0
        )
        
        if self.temperature_field is not None:
            self.temp_interp = RegularGridInterpolator(
                points, self.temperature_field, bounds_error=False, fill_value=None
            )
    
    def add_particles(
        self,
        positions: np.ndarray,
        properties: Optional[dict] = None
    ) -> None:
        """
        Add particles at specified positions.
        
        Args:
            positions: Array of (r, z) positions, shape (n_particles, 2)
            properties: Optional dict with particle properties
        """
        n_new = len(positions)
        
        for i in range(n_new):
            r, z = positions[i]
            
            # Check if position is within domain
            if self._is_in_domain(r, z):
                self.positions.append([r, z])
                self.active.append(True)
                self.ages.append(0.0)
                
                # Get velocity at position
                vel = self._get_velocity(r, z)
                vel_mag = np.sqrt(vel[0]**2 + vel[1]**2)
                self.velocities.append(vel_mag)
                
                # Get temperature if available
                if self.temperature_field is not None:
                    temp = self._get_temperature(r, z)
                    self.temperatures.append(temp)
    
    def update(self, dt: float) -> None:
        """
        Update particle positions using RK4 integration.
        
        Args:
            dt: Time step [s]
        """
        for i in range(len(self.positions)):
            if not self.active[i]:
                continue
            
            r, z = self.positions[i]
            
            # RK4 integration
            k1 = self._get_velocity(r, z)
            k2 = self._get_velocity(r + 0.5*dt*k1[0], z + 0.5*dt*k1[1])
            k3 = self._get_velocity(r + 0.5*dt*k2[0], z + 0.5*dt*k2[1])
            k4 = self._get_velocity(r + dt*k3[0], z + dt*k3[1])
            
            # Update position
            dr = (dt/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            dz = (dt/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            
            r_new = r + dr
            z_new = z + dz
            
            # Check if particle is still in domain
            if self._is_in_domain(r_new, z_new):
                self.positions[i] = [r_new, z_new]
                
                # Update velocity magnitude
                vel = self._get_velocity(r_new, z_new)
                vel_mag = np.sqrt(vel[0]**2 + vel[1]**2)
                self.velocities[i] = vel_mag
                
                # Update temperature if available
                if self.temperature_field is not None:
                    self.temperatures[i] = self._get_temperature(r_new, z_new)
                
                # Update age
                self.ages[i] += dt
            else:
                # Deactivate particle that left domain
                self.active[i] = False
    
    def get_positions(self) -> np.ndarray:
        """
        Return current particle positions.
        
        Returns:
            Array of active particle positions, shape (n_active, 2)
        """
        active_positions = [
            self.positions[i] for i in range(len(self.positions))
            if self.active[i]
        ]
        return np.array(active_positions) if active_positions else np.array([]).reshape(0, 2)
    
    def get_colors(self, color_by: str = 'velocity') -> np.ndarray:
        """
        Get particle colors based on velocity, temperature, or age.
        
        Args:
            color_by: What to color by ('velocity', 'temperature', or 'age')
            
        Returns:
            Array of color values for active particles
        """
        if color_by == 'velocity':
            values = [self.velocities[i] for i in range(len(self.velocities)) 
                     if self.active[i]]
        elif color_by == 'temperature':
            if self.temperature_field is None:
                raise ValueError("Temperature field not available")
            values = [self.temperatures[i] for i in range(len(self.temperatures))
                     if self.active[i]]
        elif color_by == 'age':
            values = [self.ages[i] for i in range(len(self.ages))
                     if self.active[i]]
        else:
            raise ValueError(f"Unknown color_by option: {color_by}")
        
        return np.array(values) if values else np.array([])
    
    def remove_inactive_particles(self) -> None:
        """Remove particles that have left the domain."""
        # Create new lists with only active particles
        new_positions = []
        new_velocities = []
        new_temperatures = []
        new_ages = []
        new_active = []
        
        for i in range(len(self.positions)):
            if self.active[i]:
                new_positions.append(self.positions[i])
                new_velocities.append(self.velocities[i])
                new_ages.append(self.ages[i])
                new_active.append(True)
                if self.temperature_field is not None:
                    new_temperatures.append(self.temperatures[i])
        
        self.positions = new_positions
        self.velocities = new_velocities
        self.temperatures = new_temperatures
        self.ages = new_ages
        self.active = new_active
    
    def get_particle_count(self) -> int:
        """
        Get number of active particles.
        
        Returns:
            Number of active particles
        """
        return sum(self.active)
    
    def _get_velocity(self, r: float, z: float) -> Tuple[float, float]:
        """
        Get velocity at arbitrary position using interpolation.
        
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
    
    def _get_temperature(self, r: float, z: float) -> float:
        """
        Get temperature at arbitrary position using interpolation.
        
        Args:
            r: Radial position [m]
            z: Axial position [m]
            
        Returns:
            Temperature [K]
        """
        pos = np.array([[r, z]])
        return float(self.temp_interp(pos)[0])
    
    def _is_in_domain(self, r: float, z: float) -> bool:
        """
        Check if position is within domain bounds.
        
        Args:
            r: Radial position [m]
            z: Axial position [m]
            
        Returns:
            True if position is in domain
        """
        return (self.r_min <= r <= self.r_max and 
                self.z_min <= z <= self.z_max)
