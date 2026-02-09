"""
Solver module for Navier-Stokes equations.

This module implements solvers for incompressible flow with heat transfer.
Includes analytical solutions and numerical methods.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from numba import jit

from ..models.oil_properties import OilFluid
from ..models.turbulence import friction_factor
from ..geometry.pipe import Pipe
from ..core.boundary import InletBC, OutletBC, WallBC
from ..core.grid import Grid2DAxisymmetric


class PoiseuilleFlow:
    """Analytical solution for laminar pipe flow (Hagen-Poiseuille)."""

    def __init__(
        self,
        pipe: Pipe,
        fluid: OilFluid,
        temperature: float = 293.15
    ):
        """
        Initialize Poiseuille flow solver.

        Args:
            pipe: Pipe geometry
            fluid: Fluid properties
            temperature: Temperature [K]
        """
        self.pipe = pipe
        self.fluid = fluid
        self.temperature = temperature
        self.mu = fluid.dynamic_viscosity(temperature)
        self.rho = fluid.density(temperature)

    def velocity_profile(self, r: np.ndarray, mean_velocity: float) -> np.ndarray:
        """
        Calculate parabolic velocity profile.

        Args:
            r: Radial positions [m]
            mean_velocity: Mean velocity [m/s]

        Returns:
            Velocity at each radial position [m/s]

        Note:
            u(r) = u_max * (1 - (r/R)²)
            where u_max = 2 * u_mean
        """
        R = self.pipe.radius()
        u_max = 2 * mean_velocity
        return u_max * (1 - (r / R) ** 2)

    def pressure_drop(self, mean_velocity: float, length: Optional[float] = None) -> float:
        """
        Calculate pressure drop using Hagen-Poiseuille equation.

        Args:
            mean_velocity: Mean velocity [m/s]
            length: Length over which to calculate drop (default: pipe length) [m]

        Returns:
            Pressure drop [Pa]

        Note:
            Δp = 32 * μ * L * u_mean / D²
        """
        L = length if length is not None else self.pipe.length
        D = self.pipe.diameter
        return 32 * self.mu * L * mean_velocity / D ** 2

    def mass_flow_rate(self, mean_velocity: float) -> float:
        """
        Calculate mass flow rate.

        Args:
            mean_velocity: Mean velocity [m/s]

        Returns:
            Mass flow rate [kg/s]
        """
        return self.rho * mean_velocity * self.pipe.cross_sectional_area()

    def reynolds_number(self, mean_velocity: float) -> float:
        """
        Calculate Reynolds number.

        Args:
            mean_velocity: Mean velocity [m/s]

        Returns:
            Reynolds number [-]
        """
        return self.fluid.reynolds_number(mean_velocity, self.pipe.diameter, self.temperature)


class PipeFlowSolver:
    """General pipe flow solver with support for laminar and turbulent flow."""

    def __init__(
        self,
        pipe: Pipe,
        fluid: OilFluid,
        inlet_bc: InletBC,
        outlet_bc: OutletBC,
        wall_bc: Optional[WallBC] = None
    ):
        """
        Initialize pipe flow solver.

        Args:
            pipe: Pipe geometry
            fluid: Fluid properties
            inlet_bc: Inlet boundary condition
            outlet_bc: Outlet boundary condition
            wall_bc: Wall boundary condition (for thermal analysis)
        """
        self.pipe = pipe
        self.fluid = fluid
        self.inlet_bc = inlet_bc
        self.outlet_bc = outlet_bc
        self.wall_bc = wall_bc

        # Results storage
        self.mean_velocity = None
        self.pressure_drop = None
        self.reynolds_number = None
        self.flow_regime = None

    def solve_flow(self) -> dict:
        """
        Solve flow field.

        Returns:
            Dictionary with flow results
        """
        # Get inlet temperature
        T = self.inlet_bc.temperature if self.inlet_bc.temperature else 293.15

        # Calculate mean velocity
        if self.inlet_bc.velocity is not None:
            self.mean_velocity = self.inlet_bc.velocity
        elif self.inlet_bc.mass_flow_rate is not None:
            rho = self.fluid.density(T)
            A = self.pipe.cross_sectional_area()
            self.mean_velocity = self.inlet_bc.mass_flow_rate / (rho * A)

        # Calculate Reynolds number
        self.reynolds_number = self.fluid.reynolds_number(
            self.mean_velocity,
            self.pipe.diameter,
            T
        )

        # Determine flow regime
        if self.reynolds_number < 2300:
            self.flow_regime = "laminar"
        elif self.reynolds_number < 4000:
            self.flow_regime = "transitional"
        else:
            self.flow_regime = "turbulent"

        # Calculate pressure drop
        self.pressure_drop = self._calculate_pressure_drop(T)

        return {
            "mean_velocity": self.mean_velocity,
            "reynolds_number": self.reynolds_number,
            "flow_regime": self.flow_regime,
            "pressure_drop": self.pressure_drop,
            "temperature": T,
        }

    def _calculate_pressure_drop(self, temperature: float) -> float:
        """
        Calculate pressure drop using Darcy-Weisbach equation.

        Args:
            temperature: Temperature [K]

        Returns:
            Pressure drop [Pa]

        Note:
            Δp = f * (L/D) * (ρ*u²/2)
        """
        rho = self.fluid.density(temperature)
        f = friction_factor(self.reynolds_number, self.pipe.relative_roughness())

        # Darcy-Weisbach equation
        dp = f * (self.pipe.length / self.pipe.diameter) * (rho * self.mean_velocity ** 2 / 2)

        return dp

    def solve_velocity_profile_2d(
        self,
        grid: Grid2DAxisymmetric
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for 2D velocity profile.

        Args:
            grid: 2D axisymmetric grid

        Returns:
            Tuple of (radial velocity, axial velocity) arrays

        Note:
            For fully developed laminar flow, uses analytical Poiseuille profile.
            For turbulent flow, uses power law approximation.
        """
        if self.flow_regime == "laminar":
            # Poiseuille profile
            u_max = 2 * self.mean_velocity
            R = self.pipe.radius()
            u_z = u_max * (1 - (grid.r / R) ** 2)
            u_r = np.zeros_like(u_z)

        else:
            # Power law profile for turbulent flow: u/u_max = (1 - r/R)^(1/n)
            n = 7  # Common for turbulent flow
            u_max = (n + 1) * (n + 2) / (2 * n ** 2) * self.mean_velocity
            R = self.pipe.radius()
            u_z = u_max * (1 - grid.r / R) ** (1 / n)
            u_r = np.zeros_like(u_z)

        # Expand to full grid
        u_z_2d = np.tile(u_z[:, np.newaxis], (1, grid.n_axial))
        u_r_2d = np.zeros_like(u_z_2d)

        return u_r_2d, u_z_2d


class HeatTransferSolver:
    """Solver for heat transfer in pipe flow."""

    def __init__(
        self,
        pipe: Pipe,
        fluid: OilFluid,
        inlet_temperature: float,
        wall_bc: WallBC,
        mean_velocity: float
    ):
        """
        Initialize heat transfer solver.

        Args:
            pipe: Pipe geometry
            fluid: Fluid properties
            inlet_temperature: Inlet temperature [K]
            wall_bc: Wall boundary condition
            mean_velocity: Mean flow velocity [m/s]
        """
        self.pipe = pipe
        self.fluid = fluid
        self.inlet_temperature = inlet_temperature
        self.wall_bc = wall_bc
        self.mean_velocity = mean_velocity

    def solve_temperature_profile_1d(
        self,
        z: np.ndarray
    ) -> np.ndarray:
        """
        Solve 1D temperature profile along pipe axis.

        Args:
            z: Axial positions [m]

        Returns:
            Temperature at each position [K]

        Note:
            Assumes constant wall temperature or heat flux.
            Uses exponential temperature profile for constant wall temperature.
        """
        if self.wall_bc.temperature is not None:
            # Constant wall temperature case
            T_wall = self.wall_bc.temperature
            T_in = self.inlet_temperature

            # Calculate heat transfer coefficient (simplified)
            Re = self.fluid.reynolds_number(
                self.mean_velocity,
                self.pipe.diameter,
                self.inlet_temperature
            )
            Pr = self.fluid.prandtl_number(self.inlet_temperature)

            # Nusselt number correlation
            if Re < 2300:
                Nu = 3.66  # Laminar, constant wall temperature
            else:
                # Dittus-Boelter correlation
                Nu = 0.023 * Re ** 0.8 * Pr ** 0.4

            h = Nu * self.fluid.thermal_conductivity() / self.pipe.diameter

            # Exponential profile
            rho = self.fluid.density(self.inlet_temperature)
            cp = self.fluid.specific_heat()
            A = self.pipe.cross_sectional_area()
            P = self.pipe.wetted_perimeter()

            tau = rho * cp * A * self.mean_velocity / (h * P)
            T = T_wall - (T_wall - T_in) * np.exp(-z / tau)

        elif self.wall_bc.heat_flux is not None:
            # Constant heat flux case
            q = self.wall_bc.heat_flux
            P = self.pipe.wetted_perimeter()
            rho = self.fluid.density(self.inlet_temperature)
            cp = self.fluid.specific_heat()
            A = self.pipe.cross_sectional_area()

            # Linear temperature rise
            T = self.inlet_temperature + (q * P * z) / (rho * cp * A * self.mean_velocity)

        else:
            # Adiabatic wall
            T = np.full_like(z, self.inlet_temperature)

        return T
