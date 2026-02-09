"""
Tests for particle tracer module.

Tests particle advection, RK4 integration accuracy, and boundary conditions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.visualization.particle_tracer import ParticleTracer


def test_particle_initialization():
    """Test particle tracer initialization."""
    # Create simple velocity field
    r = np.linspace(0, 1, 10)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((10, 50))
    u_z = np.ones((10, 50))  # Constant axial velocity
    
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    assert tracer.r_min == 0.0
    assert tracer.r_max == 1.0
    assert tracer.z_min == 0.0
    assert tracer.z_max == 5.0
    assert tracer.get_particle_count() == 0


def test_particle_advection():
    """Test particle follows velocity field correctly."""
    # Create uniform velocity field: u_z = 1.0 m/s
    r = np.linspace(0, 1, 10)
    z = np.linspace(0, 10, 100)
    u_r = np.zeros((10, 100))
    u_z = np.ones((10, 100))
    
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Add particle at inlet
    initial_pos = np.array([[0.5, 0.1]])
    tracer.add_particles(initial_pos)
    
    assert tracer.get_particle_count() == 1
    
    # Update particle position (should move in +z direction)
    dt = 0.1  # seconds
    tracer.update(dt)
    
    positions = tracer.get_positions()
    assert len(positions) == 1
    
    # Check particle moved approximately dt * u_z in z-direction
    assert positions[0, 0] == pytest.approx(0.5, abs=0.01)  # r unchanged
    assert positions[0, 1] > 0.1  # z increased


def test_rk4_accuracy():
    """Test RK4 integration accuracy against analytical solution."""
    # Create linear velocity field: u_z = z (analytical: z(t) = z0 * exp(t))
    # For constant velocity, we can check accuracy more easily
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 10, 100)
    u_r = np.zeros((20, 100))
    u_z = np.ones((20, 100)) * 2.0  # Constant 2 m/s
    
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Initial position
    z0 = 1.0
    initial_pos = np.array([[0.5, z0]])
    tracer.add_particles(initial_pos)
    
    # Integrate for 1 second with small timesteps
    dt = 0.01
    n_steps = 100
    
    for _ in range(n_steps):
        tracer.update(dt)
    
    positions = tracer.get_positions()
    
    # Analytical solution: z = z0 + u_z * t
    expected_z = z0 + 2.0 * (n_steps * dt)
    
    # RK4 should be very accurate for constant velocity
    assert positions[0, 1] == pytest.approx(expected_z, rel=0.01)


def test_particle_boundary_conditions():
    """Test particles respect boundary conditions."""
    r = np.linspace(0, 1, 10)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((10, 50))
    u_z = np.ones((10, 50)) * 10.0  # Fast velocity
    
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Add particle near outlet
    initial_pos = np.array([[0.5, 4.5]])
    tracer.add_particles(initial_pos)
    
    # Update - particle should leave domain
    dt = 1.0
    tracer.update(dt)
    
    # Particle should be marked inactive (outside domain)
    assert tracer.get_particle_count() == 0  # No active particles


def test_particle_colors_velocity():
    """Test particle coloring by velocity."""
    r = np.linspace(0, 1, 10)
    z = np.linspace(0, 5, 50)
    
    # Create parabolic velocity profile
    R, Z = np.meshgrid(r, z, indexing='ij')
    u_r = np.zeros_like(R)
    u_z = 1.0 * (1 - (R / 1.0)**2)  # Parabolic
    
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Add particles at different radii
    positions = np.array([
        [0.0, 1.0],  # Center - max velocity
        [0.5, 1.0],  # Mid
        [0.9, 1.0],  # Near wall - low velocity
    ])
    tracer.add_particles(positions)
    
    # Get colors by velocity
    colors = tracer.get_colors(color_by='velocity')
    
    assert len(colors) == 3
    # Center should have highest velocity
    assert colors[0] > colors[1] > colors[2]


def test_particle_colors_temperature():
    """Test particle coloring by temperature."""
    r = np.linspace(0, 1, 10)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((10, 50))
    u_z = np.ones((10, 50))
    
    # Temperature field (increasing with z)
    R, Z = np.meshgrid(r, z, indexing='ij')
    T = 300 + 10 * Z  # Linear temperature increase
    
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z),
        temperature_field=T
    )
    
    # Add particles at different z
    positions = np.array([
        [0.5, 1.0],
        [0.5, 3.0],
    ])
    tracer.add_particles(positions)
    
    # Get colors by temperature
    colors = tracer.get_colors(color_by='temperature')
    
    assert len(colors) == 2
    # Particle at higher z should have higher temperature
    assert colors[1] > colors[0]


def test_particle_age():
    """Test particle age tracking."""
    r = np.linspace(0, 1, 10)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((10, 50))
    u_z = np.ones((10, 50))
    
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Add particle
    initial_pos = np.array([[0.5, 1.0]])
    tracer.add_particles(initial_pos)
    
    # Age should start at 0
    ages = tracer.get_colors(color_by='age')
    assert ages[0] == 0.0
    
    # Update and check age increased
    dt = 0.5
    tracer.update(dt)
    tracer.update(dt)
    
    ages = tracer.get_colors(color_by='age')
    assert ages[0] == pytest.approx(1.0, abs=0.01)


def test_remove_inactive_particles():
    """Test removal of inactive particles."""
    r = np.linspace(0, 1, 10)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((10, 50))
    u_z = np.ones((10, 50)) * 10.0
    
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Add particles
    positions = np.array([
        [0.5, 1.0],  # Will stay
        [0.5, 4.9],  # Will leave
    ])
    tracer.add_particles(positions)
    
    assert len(tracer.positions) == 2
    
    # Update - one particle leaves
    tracer.update(1.0)
    
    # Remove inactive
    tracer.remove_inactive_particles()
    
    # Should have fewer particles now
    assert len(tracer.positions) < 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
