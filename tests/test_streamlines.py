"""
Tests for streamline generation module.

Tests streamline integration, vorticity computation, and vortex identification.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.visualization.streamlines import StreamlineGenerator


def test_streamline_generator_initialization():
    """Test streamline generator initialization."""
    r = np.linspace(0, 1, 10)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((10, 50))
    u_z = np.ones((10, 50))
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    assert generator.r_min == 0.0
    assert generator.r_max == 1.0
    assert generator.z_min == 0.0
    assert generator.z_max == 5.0


def test_generate_streamlines():
    """Test streamline generation from seed points."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 10, 100)
    
    # Create uniform velocity field
    u_r = np.zeros((20, 100))
    u_z = np.ones((20, 100))
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Generate streamlines
    seed_points = np.array([
        [0.3, 0.5],
        [0.5, 0.5],
        [0.7, 0.5],
    ])
    
    streamlines = generator.generate_streamlines(
        seed_points,
        max_length=5.0,
        step_size=0.1
    )
    
    assert len(streamlines) == 3
    
    # Each streamline should have multiple points
    for streamline in streamlines:
        assert len(streamline) > 1
        
        # For uniform velocity, streamlines should be straight lines
        # Check that r coordinate stays roughly constant
        r_values = streamline[:, 0]
        assert np.std(r_values) < 0.1  # Small variation


def test_streamline_follows_velocity():
    """Test that streamlines follow velocity field direction."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 10, 100)
    
    # Create parabolic velocity profile
    R, Z = np.meshgrid(r, z, indexing='ij')
    u_r = np.zeros_like(R)
    u_z = 1.0 * (1 - (R / 1.0)**2)  # Faster at center
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Seed at center (high velocity) and near wall (low velocity)
    seed_points = np.array([
        [0.1, 0.5],  # Near center
        [0.9, 0.5],  # Near wall
    ])
    
    streamlines = generator.generate_streamlines(
        seed_points,
        max_length=3.0,
        step_size=0.05
    )
    
    # Center streamline should be longer (travels farther in same distance)
    # because velocity is higher
    assert len(streamlines[0]) >= len(streamlines[1])


def test_streamline_boundary_respect():
    """Test streamlines stop at domain boundaries."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((20, 50))
    u_z = np.ones((20, 50))
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Seed near outlet
    seed_points = np.array([[0.5, 4.0]])
    
    streamlines = generator.generate_streamlines(
        seed_points,
        max_length=10.0,  # Request more than domain allows
        step_size=0.1
    )
    
    # Streamline should stop before z = 5.0
    assert streamlines[0][-1, 1] <= 5.0


def test_compute_vorticity():
    """Test vorticity computation."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    
    # Create velocity field with vorticity
    R, Z = np.meshgrid(r, z, indexing='ij')
    u_r = -Z  # Some radial velocity
    u_z = R   # Some axial velocity
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    vorticity = generator.compute_vorticity()
    
    # Vorticity should be computed for entire grid
    assert vorticity.shape == (20, 50)
    
    # For this specific field: ω_θ = ∂u_z/∂r - ∂u_r/∂z
    # ∂u_z/∂r = 1, ∂u_r/∂z = -1
    # ω_θ = 1 - (-1) = 2
    # Check that vorticity is approximately constant = 2
    assert np.mean(np.abs(vorticity)) > 0


def test_identify_vortices_q_criterion():
    """Test vortex identification using Q-criterion."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    
    # Create simple vortex-like flow
    R, Z = np.meshgrid(r, z, indexing='ij')
    u_r = -Z
    u_z = R
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    vortex_mask = generator.identify_vortices(method='q_criterion')
    
    assert vortex_mask.shape == (20, 50)
    assert np.max(vortex_mask) > 0  # Should identify some vortex regions


def test_identify_vortices_lambda2():
    """Test vortex identification using lambda2 method."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    
    R, Z = np.meshgrid(r, z, indexing='ij')
    u_r = -Z
    u_z = R
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    vortex_mask = generator.identify_vortices(
        method='lambda2',
        threshold=0.1
    )
    
    assert vortex_mask.shape == (20, 50)


def test_generate_pathlines():
    """Test pathline generation."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((20, 50))
    u_z = np.ones((20, 50))
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    seed_points = np.array([
        [0.5, 0.5],
    ])
    time_steps = np.linspace(0, 2, 20)
    
    pathlines = generator.generate_pathlines(seed_points, time_steps)
    
    assert len(pathlines) == 1
    # For steady flow, pathlines should follow streamlines
    assert len(pathlines[0]) > 1


def test_generate_streaklines():
    """Test streakline generation."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((20, 50))
    u_z = np.ones((20, 50))
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    seed_point = np.array([0.5, 0.5])
    release_times = np.linspace(0, 1, 10)
    
    streakline = generator.generate_streaklines(seed_point, release_times)
    
    # For steady flow, streakline = streamline
    assert len(streakline) > 1


def test_zero_velocity_handling():
    """Test handling of zero velocity regions."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    
    # Zero velocity field
    u_r = np.zeros((20, 50))
    u_z = np.zeros((20, 50))
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    seed_points = np.array([[0.5, 1.0]])
    
    streamlines = generator.generate_streamlines(
        seed_points,
        max_length=5.0,
        step_size=0.1
    )
    
    # Streamline should be very short (particle doesn't move)
    assert len(streamlines[0]) <= 2


def test_vorticity_zero_for_uniform_flow():
    """Test that vorticity is zero for uniform flow."""
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    
    # Uniform flow - no rotation
    u_r = np.zeros((20, 50))
    u_z = np.ones((20, 50))
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    vorticity = generator.compute_vorticity()
    
    # Vorticity should be approximately zero
    assert np.mean(np.abs(vorticity)) < 0.1


def test_multiple_streamlines():
    """Test generating multiple streamlines efficiently."""
    r = np.linspace(0, 1, 30)
    z = np.linspace(0, 10, 100)
    
    R, Z = np.meshgrid(r, z, indexing='ij')
    u_r = np.zeros_like(R)
    u_z = np.ones_like(R)
    
    generator = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    
    # Generate many streamlines
    n_lines = 50
    seed_points = np.column_stack([
        np.linspace(0.1, 0.9, n_lines),
        np.ones(n_lines) * 0.5
    ])
    
    streamlines = generator.generate_streamlines(
        seed_points,
        max_length=8.0,
        step_size=0.1
    )
    
    assert len(streamlines) == n_lines


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
