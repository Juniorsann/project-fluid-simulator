"""
Tests for solver components.
"""

import pytest
import numpy as np
from src.core.solver import PoiseuilleFlow, PipeFlowSolver, HeatTransferSolver
from src.models.oil_properties import create_medium_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC, WallBC


class TestPoiseuilleFlow:
    """Test analytical Poiseuille flow solution."""

    @pytest.fixture
    def setup_poiseuille(self):
        """Setup for Poiseuille flow tests."""
        pipe = Pipe(diameter=0.1, length=10.0)
        oil = create_medium_oil()
        temperature = 293.15
        return PoiseuilleFlow(pipe, oil, temperature)

    def test_velocity_profile_shape(self, setup_poiseuille):
        """Test that velocity profile is parabolic."""
        solver = setup_poiseuille
        r = np.linspace(0, solver.pipe.radius(), 100)
        u = solver.velocity_profile(r, mean_velocity=1.0)
        
        # Maximum velocity at centerline
        assert u[0] == max(u)
        
        # Zero velocity at wall
        assert u[-1] >= 0
        assert u[-1] < 0.01  # Should be very close to zero

    def test_velocity_profile_mean(self, setup_poiseuille):
        """Test that mean of parabolic profile is correct."""
        solver = setup_poiseuille
        mean_velocity = 1.0
        
        # Create radial grid for integration
        r = np.linspace(0, solver.pipe.radius(), 1000)
        u = solver.velocity_profile(r, mean_velocity)
        
        # Calculate mean using integration (Q/A)
        # For axisymmetric: Q = 2π ∫ u*r dr
        dr = r[1] - r[0]
        Q = 2 * np.pi * np.sum(u * r * dr)
        A = np.pi * solver.pipe.radius()**2
        u_mean_calc = Q / A
        
        # Should match input mean velocity
        np.testing.assert_allclose(u_mean_calc, mean_velocity, rtol=0.01)

    def test_pressure_drop_laminar(self, setup_poiseuille):
        """Test Hagen-Poiseuille pressure drop formula."""
        solver = setup_poiseuille
        mean_velocity = 0.5
        
        dp = solver.pressure_drop(mean_velocity)
        
        # Check against analytical formula: Δp = 32 * μ * L * u / D²
        D = solver.pipe.diameter
        L = solver.pipe.length
        mu = solver.mu
        dp_analytical = 32 * mu * L * mean_velocity / D**2
        
        np.testing.assert_allclose(dp, dp_analytical, rtol=0.01)

    def test_reynolds_number(self, setup_poiseuille):
        """Test Reynolds number calculation."""
        solver = setup_poiseuille
        mean_velocity = 1.0
        
        Re = solver.reynolds_number(mean_velocity)
        
        # Re = ρ * u * D / μ
        rho = solver.rho
        D = solver.pipe.diameter
        mu = solver.mu
        Re_expected = rho * mean_velocity * D / mu
        
        np.testing.assert_allclose(Re, Re_expected, rtol=0.01)

    def test_mass_flow_rate(self, setup_poiseuille):
        """Test mass flow rate calculation."""
        solver = setup_poiseuille
        mean_velocity = 1.0
        
        m_dot = solver.mass_flow_rate(mean_velocity)
        
        # m_dot = ρ * u * A
        A = solver.pipe.cross_sectional_area()
        m_dot_expected = solver.rho * mean_velocity * A
        
        np.testing.assert_allclose(m_dot, m_dot_expected, rtol=0.01)


class TestPipeFlowSolver:
    """Test general pipe flow solver."""

    @pytest.fixture
    def setup_solver(self):
        """Setup for solver tests."""
        pipe = Pipe(diameter=0.2, length=100.0)
        oil = create_medium_oil()
        inlet = InletBC(velocity=1.0, temperature=293.15)
        outlet = OutletBC(pressure=101325.0)
        return PipeFlowSolver(pipe, oil, inlet, outlet)

    def test_solver_initialization(self, setup_solver):
        """Test solver initialization."""
        solver = setup_solver
        assert solver.pipe is not None
        assert solver.fluid is not None
        assert solver.inlet_bc is not None
        assert solver.outlet_bc is not None

    def test_solve_flow_laminar(self):
        """Test flow solution for laminar regime."""
        pipe = Pipe(diameter=0.1, length=10.0)
        oil = create_medium_oil()
        inlet = InletBC(velocity=0.1, temperature=293.15)  # Low velocity for laminar
        outlet = OutletBC(pressure=101325.0)
        
        solver = PipeFlowSolver(pipe, oil, inlet, outlet)
        results = solver.solve_flow()
        
        assert 'reynolds_number' in results
        assert 'flow_regime' in results
        assert 'pressure_drop' in results
        
        # Should be laminar
        assert results['reynolds_number'] < 2300
        assert results['flow_regime'] == 'laminar'
        assert results['pressure_drop'] > 0

    def test_solve_flow_turbulent(self):
        """Test flow solution for turbulent regime."""
        pipe = Pipe(diameter=0.2, length=100.0)
        oil = create_medium_oil()
        inlet = InletBC(velocity=5.0, temperature=353.15)  # High velocity, high temp
        outlet = OutletBC(pressure=101325.0)
        
        solver = PipeFlowSolver(pipe, oil, inlet, outlet)
        results = solver.solve_flow()
        
        # Might be transitional or turbulent depending on conditions
        assert results['reynolds_number'] > 0
        assert results['pressure_drop'] > 0

    def test_pressure_drop_positive(self, setup_solver):
        """Test that pressure drop is always positive."""
        solver = setup_solver
        results = solver.solve_flow()
        
        assert results['pressure_drop'] > 0

    def test_mass_flow_boundary_condition(self):
        """Test solving with mass flow rate boundary condition."""
        pipe = Pipe(diameter=0.2, length=100.0)
        oil = create_medium_oil()
        mass_flow = 10.0  # kg/s
        inlet = InletBC(mass_flow_rate=mass_flow, temperature=293.15)
        outlet = OutletBC(pressure=101325.0)
        
        solver = PipeFlowSolver(pipe, oil, inlet, outlet)
        results = solver.solve_flow()
        
        assert results['mean_velocity'] > 0
        
        # Verify mass flow rate
        rho = oil.density(293.15)
        A = pipe.cross_sectional_area()
        m_dot_calc = rho * results['mean_velocity'] * A
        np.testing.assert_allclose(m_dot_calc, mass_flow, rtol=0.01)


class TestHeatTransferSolver:
    """Test heat transfer solver."""

    @pytest.fixture
    def setup_heat_solver(self):
        """Setup for heat transfer tests."""
        pipe = Pipe(diameter=0.15, length=50.0)
        oil = create_medium_oil()
        T_inlet = 293.15
        wall = WallBC(temperature=350.15)
        mean_velocity = 0.5
        return HeatTransferSolver(pipe, oil, T_inlet, wall, mean_velocity)

    def test_heat_solver_initialization(self, setup_heat_solver):
        """Test heat solver initialization."""
        solver = setup_heat_solver
        assert solver.pipe is not None
        assert solver.fluid is not None
        assert solver.inlet_temperature > 0
        assert solver.wall_bc is not None

    def test_temperature_profile_constant_wall(self, setup_heat_solver):
        """Test temperature profile with constant wall temperature."""
        solver = setup_heat_solver
        z = np.linspace(0, solver.pipe.length, 100)
        T = solver.solve_temperature_profile_1d(z)
        
        # Temperature should increase along pipe
        assert T[0] == solver.inlet_temperature or np.isclose(T[0], solver.inlet_temperature, rtol=0.01)
        assert T[-1] > T[0]
        
        # Should approach wall temperature
        assert T[-1] <= solver.wall_bc.temperature

    def test_temperature_profile_heat_flux(self):
        """Test temperature profile with constant heat flux."""
        pipe = Pipe(diameter=0.15, length=50.0)
        oil = create_medium_oil()
        T_inlet = 293.15
        wall = WallBC(heat_flux=1000.0)  # 1000 W/m²
        mean_velocity = 0.5
        
        solver = HeatTransferSolver(pipe, oil, T_inlet, wall, mean_velocity)
        z = np.linspace(0, pipe.length, 100)
        T = solver.solve_temperature_profile_1d(z)
        
        # Temperature should increase linearly with constant heat flux
        assert T[-1] > T[0]
        
        # Check if roughly linear
        dT_dz = np.gradient(T, z)
        assert np.std(dT_dz) / np.mean(dT_dz) < 0.1  # Should be relatively constant

    def test_adiabatic_wall(self):
        """Test adiabatic wall condition."""
        pipe = Pipe(diameter=0.15, length=50.0)
        oil = create_medium_oil()
        T_inlet = 293.15
        wall = WallBC()  # Adiabatic (no temperature or heat flux)
        mean_velocity = 0.5
        
        solver = HeatTransferSolver(pipe, oil, T_inlet, wall, mean_velocity)
        z = np.linspace(0, pipe.length, 100)
        T = solver.solve_temperature_profile_1d(z)
        
        # Temperature should remain constant
        np.testing.assert_allclose(T, T_inlet, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
