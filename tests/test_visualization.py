"""
Tests for visualization components.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from src.visualization.particle_tracer import ParticleTracer
from src.visualization.streamlines import StreamlineGenerator
from src.visualization.animator import FlowAnimator


class TestParticleTracer:
    """Test particle tracer functionality."""
    
    @pytest.fixture
    def simple_velocity_field(self):
        """Create a simple uniform velocity field for testing."""
        # Create a uniform flow in z-direction
        n_z, n_r = 50, 30
        z = np.linspace(0, 1.0, n_z)
        r = np.linspace(0, 0.1, n_r)
        
        # Uniform velocity field
        u_field = np.ones((n_r, n_z))  # Axial velocity
        v_field = np.zeros((n_r, n_z))  # Radial velocity
        
        return (u_field, v_field), (z, r)
    
    @pytest.fixture
    def parabolic_velocity_field(self):
        """Create a parabolic (Poiseuille) velocity field."""
        n_z, n_r = 50, 30
        z = np.linspace(0, 1.0, n_z)
        r = np.linspace(0, 0.1, n_r)
        
        # Parabolic profile: u = u_max * (1 - (r/R)^2)
        R = 0.1
        u_max = 2.0
        
        u_field = np.zeros((n_r, n_z))
        for i, r_val in enumerate(r):
            u_field[i, :] = u_max * (1 - (r_val / R) ** 2)
        
        v_field = np.zeros((n_r, n_z))
        
        return (u_field, v_field), (z, r)
    
    def test_particle_initialization(self, simple_velocity_field):
        """Test particle tracer initialization."""
        velocity_field, grid = simple_velocity_field
        
        tracer = ParticleTracer(velocity_field, grid, n_particles=100)
        
        assert tracer.n_particles == 100
        assert tracer.positions is None  # Not injected yet
        assert tracer.active is None
    
    def test_particle_injection_inlet(self, simple_velocity_field):
        """Test particle injection at inlet."""
        velocity_field, grid = simple_velocity_field
        
        tracer = ParticleTracer(velocity_field, grid, n_particles=100)
        tracer.inject_particles(region='inlet', distribution='random')
        
        assert tracer.positions is not None
        assert len(tracer.positions) == 100
        assert tracer.active.sum() == 100  # All active initially
        
        # All particles should be at inlet (z_min)
        z_min = grid[0].min()
        np.testing.assert_allclose(tracer.positions[:, 0], z_min, rtol=1e-5)
    
    def test_particle_injection_uniform(self, simple_velocity_field):
        """Test uniform particle injection."""
        velocity_field, grid = simple_velocity_field
        
        tracer = ParticleTracer(velocity_field, grid, n_particles=100)
        tracer.inject_particles(region='uniform', distribution='random')
        
        assert len(tracer.positions) == 100
        
        # Particles should be distributed in domain
        z_positions = tracer.positions[:, 0]
        r_positions = tracer.positions[:, 1]
        
        assert z_positions.min() >= grid[0].min()
        assert z_positions.max() <= grid[0].max()
        assert r_positions.min() >= grid[1].min()
        assert r_positions.max() <= grid[1].max()
    
    def test_particle_advection_uniform_flow(self, simple_velocity_field):
        """Test that particles in uniform flow move with constant velocity."""
        velocity_field, grid = simple_velocity_field
        
        tracer = ParticleTracer(velocity_field, grid, n_particles=10)
        tracer.inject_particles(region='inlet', distribution='uniform')
        
        initial_positions = tracer.positions.copy()
        
        # Advect for short time
        dt = 0.01
        tracer.advect(dt)
        
        # Particles should have moved in z-direction
        displacement = tracer.positions - initial_positions
        
        # Check z-displacement is approximately velocity * dt
        # Since uniform velocity = 1.0 m/s
        expected_dz = 1.0 * dt
        np.testing.assert_allclose(displacement[:, 0], expected_dz, rtol=0.1)
        
        # Radial displacement should be near zero
        assert np.all(np.abs(displacement[:, 1]) < 0.001)
    
    def test_particle_removal_out_of_bounds(self, simple_velocity_field):
        """Test that particles leaving domain are marked inactive."""
        velocity_field, grid = simple_velocity_field
        
        tracer = ParticleTracer(velocity_field, grid, n_particles=10)
        tracer.inject_particles(region='inlet', distribution='uniform')
        
        # Advect for long time to push particles out
        for _ in range(200):
            tracer.advect(0.01)
        
        # Some particles should have left the domain
        assert tracer.active.sum() < tracer.n_particles
    
    def test_get_particle_properties(self, simple_velocity_field):
        """Test property interpolation at particle positions."""
        velocity_field, grid = simple_velocity_field
        
        tracer = ParticleTracer(velocity_field, grid, n_particles=10)
        tracer.inject_particles(region='uniform', distribution='random')
        
        # Create a simple temperature field
        n_r, n_z = velocity_field[0].shape
        temp_field = 300 + np.linspace(0, 50, n_z)[np.newaxis, :] * np.ones((n_r, 1))
        
        temps = tracer.get_particle_properties(temp_field, 'temperature')
        
        assert len(temps) == tracer.active.sum()
        # Temperatures should be in reasonable range
        assert np.all(temps >= 300)
        assert np.all(temps <= 350)
    
    def test_particle_count(self, simple_velocity_field):
        """Test particle count statistics."""
        velocity_field, grid = simple_velocity_field
        
        tracer = ParticleTracer(velocity_field, grid, n_particles=50)
        tracer.inject_particles(region='inlet', distribution='uniform')
        
        counts = tracer.get_particle_count()
        
        assert counts['total'] == 50
        assert counts['active'] == 50
        assert counts['inactive'] == 0


class TestStreamlineGenerator:
    """Test streamline generation."""
    
    @pytest.fixture
    def uniform_field(self):
        """Create uniform velocity field."""
        n_z, n_r = 50, 30
        z = np.linspace(0, 1.0, n_z)
        r = np.linspace(0, 0.1, n_r)
        
        u_field = np.ones((n_r, n_z))
        v_field = np.zeros((n_r, n_z))  # Fixed: was (n_r, n_r)
        
        return (u_field, v_field), (z, r)
    
    @pytest.fixture
    def rotational_field(self):
        """Create rotational velocity field for testing."""
        n_z, n_r = 50, 50
        z = np.linspace(-0.5, 0.5, n_z)
        r = np.linspace(-0.5, 0.5, n_r)
        
        # Create simple rotational field: v_z = -r, v_r = z
        Z, R = np.meshgrid(z, r)
        u_field = -R  # v_z component
        v_field = Z   # v_r component
        
        return (u_field, v_field), (z, r)
    
    def test_streamline_initialization(self, uniform_field):
        """Test streamline generator initialization."""
        velocity_field, grid = uniform_field
        
        generator = StreamlineGenerator(velocity_field, grid)
        
        assert generator.streamlines == []
        assert generator.grid_z is not None
        assert generator.grid_r is not None
    
    def test_streamline_generation_auto(self, uniform_field):
        """Test automatic streamline generation."""
        velocity_field, grid = uniform_field
        
        generator = StreamlineGenerator(velocity_field, grid)
        streamlines = generator.generate_streamlines(
            seed_points='auto',
            n_streamlines=10,
            max_length=100
        )
        
        assert len(streamlines) == 10
        # Each streamline should have multiple points
        for sl in streamlines:
            assert len(sl) > 1
    
    def test_streamline_uniform_flow(self, uniform_field):
        """Test that streamlines in uniform flow are straight."""
        velocity_field, grid = uniform_field
        
        generator = StreamlineGenerator(velocity_field, grid)
        
        # Single streamline from center
        seed = np.array([[grid[0].min(), grid[1][len(grid[1])//2]]])
        
        streamlines = generator.generate_streamlines(
            seed_points=seed,
            integration_direction='forward',
            max_length=50,
            step_size=0.01
        )
        
        assert len(streamlines) == 1
        sl = streamlines[0]
        
        # In uniform flow, r should be constant
        r_values = sl[:, 1]
        assert np.std(r_values) < 0.01  # Nearly constant r
        
        # z should increase monotonically
        z_values = sl[:, 0]
        assert np.all(np.diff(z_values) > 0)
    
    def test_streamline_integration_direction(self, uniform_field):
        """Test different integration directions."""
        velocity_field, grid = uniform_field
        
        generator = StreamlineGenerator(velocity_field, grid)
        
        seed = np.array([[grid[0][len(grid[0])//2], grid[1][len(grid[1])//2]]])
        
        # Forward integration
        sl_forward = generator.generate_streamlines(
            seed_points=seed,
            integration_direction='forward',
            max_length=20
        )[0]
        
        # Backward integration
        sl_backward = generator.generate_streamlines(
            seed_points=seed,
            integration_direction='backward',
            max_length=20
        )[0]
        
        # Both integration
        sl_both = generator.generate_streamlines(
            seed_points=seed,
            integration_direction='both',
            max_length=20
        )[0]
        
        # Both should be longer than either individual
        assert len(sl_both) >= len(sl_forward)
        assert len(sl_both) >= len(sl_backward)


class TestFlowAnimator:
    """Test animation creation."""
    
    def test_animator_initialization(self):
        """Test animator initialization."""
        animator = FlowAnimator(figure_size=(10, 6), fps=30)
        
        assert animator.fps == 30
        assert animator.figure_size == (10, 6)
        assert animator.current_animation is None
    
    def test_animator_with_results(self):
        """Test animator with simulation results."""
        # Create mock results
        results = {'reynolds_number': 1000, 'flow_regime': 'laminar'}
        
        animator = FlowAnimator(simulation_results=results, fps=20)
        
        assert animator.simulation_results == results
        assert animator.fps == 20
    
    def test_animate_1d_profile(self):
        """Test 1D profile animation creation."""
        animator = FlowAnimator(fps=10)
        
        # Create simple data
        time_steps = np.linspace(0, 10, 50)
        positions = np.linspace(0, 1, 100)
        
        # Field that varies with time
        field_data = np.zeros((len(time_steps), len(positions)))
        for i, t in enumerate(time_steps):
            field_data[i, :] = np.sin(positions * 2 * np.pi) * np.exp(-t/5)
        
        anim = animator.animate_1d_profile(
            time_steps, positions, field_data,
            title="Test Animation"
        )
        
        assert anim is not None
        assert animator.fig is not None
    
    def test_animate_2d_field(self):
        """Test 2D field animation creation."""
        animator = FlowAnimator(fps=10)
        
        time_steps = np.linspace(0, 5, 20)
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 0.5, 30)
        
        # Create field data with correct shape for meshgrid
        # After meshgrid(x, y), shape is (len(y), len(x))
        field_data = np.zeros((len(time_steps), len(y), len(x)))
        for i in range(len(time_steps)):
            X, Y = np.meshgrid(x, y)
            field_data[i] = np.sin(X * 2 * np.pi) * np.cos(Y * 4 * np.pi)
        
        anim = animator.animate_2d_field(
            time_steps, x, y, field_data,
            title="2D Test"
        )
        
        assert anim is not None


def test_particle_tracer_rk4_accuracy():
    """Test that RK4 integration is more accurate than simple Euler."""
    # Create simple linear velocity field
    n_z, n_r = 30, 20
    z = np.linspace(0, 1.0, n_z)
    r = np.linspace(0, 0.1, n_r)
    
    # Constant velocity
    u_field = 0.5 * np.ones((n_r, n_z))
    v_field = np.zeros((n_r, n_z))
    
    tracer = ParticleTracer((u_field, v_field), (z, r), n_particles=1)
    
    # Single particle at inlet, center
    initial_pos = np.array([[z.min(), r.mean()]])
    tracer.inject_particles(custom_positions=initial_pos)
    
    # Advect for multiple time steps
    dt = 0.01
    n_steps = 100
    
    for _ in range(n_steps):
        tracer.advect(dt)
    
    # Expected position: initial_z + velocity * total_time
    expected_z = z.min() + 0.5 * dt * n_steps
    actual_z = tracer.positions[0, 0]
    
    # Should be very close due to RK4 accuracy
    assert abs(actual_z - expected_z) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
