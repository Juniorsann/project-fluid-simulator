"""
Boundary conditions module.

This module defines boundary conditions for CFD simulations including
inlet, outlet, and wall conditions.
"""

from typing import Optional, Union
from enum import Enum
import numpy as np


class BoundaryType(Enum):
    """Types of boundary conditions."""
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"


class BoundaryCondition:
    """Base class for boundary conditions."""

    def __init__(self, bc_type: BoundaryType, name: str = ""):
        """
        Initialize boundary condition.

        Args:
            bc_type: Type of boundary condition
            name: Name for identification
        """
        self.bc_type = bc_type
        self.name = name


class InletBC(BoundaryCondition):
    """Inlet boundary condition."""

    def __init__(
        self,
        velocity: Optional[float] = None,
        mass_flow_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        name: str = "inlet"
    ):
        """
        Initialize inlet boundary condition.

        Args:
            velocity: Prescribed velocity [m/s]
            mass_flow_rate: Mass flow rate [kg/s]
            temperature: Temperature [K]
            name: Name for identification

        Raises:
            ValueError: If neither velocity nor mass_flow_rate is specified
        """
        super().__init__(BoundaryType.INLET, name)

        if velocity is None and mass_flow_rate is None:
            raise ValueError("Either velocity or mass_flow_rate must be specified")
        if velocity is not None and mass_flow_rate is not None:
            raise ValueError("Cannot specify both velocity and mass_flow_rate")

        self.velocity = velocity
        self.mass_flow_rate = mass_flow_rate
        self.temperature = temperature

    def __repr__(self) -> str:
        if self.velocity is not None:
            return f"InletBC(velocity={self.velocity} m/s, T={self.temperature} K)"
        else:
            return f"InletBC(mass_flow={self.mass_flow_rate} kg/s, T={self.temperature} K)"


class OutletBC(BoundaryCondition):
    """Outlet boundary condition."""

    def __init__(
        self,
        pressure: float = 101325.0,
        name: str = "outlet"
    ):
        """
        Initialize outlet boundary condition.

        Args:
            pressure: Outlet pressure [Pa]
            name: Name for identification
        """
        super().__init__(BoundaryType.OUTLET, name)
        self.pressure = pressure

    def __repr__(self) -> str:
        return f"OutletBC(pressure={self.pressure} Pa)"


class WallBC(BoundaryCondition):
    """Wall boundary condition with no-slip condition."""

    def __init__(
        self,
        temperature: Optional[float] = None,
        heat_flux: Optional[float] = None,
        name: str = "wall"
    ):
        """
        Initialize wall boundary condition.

        Args:
            temperature: Wall temperature [K] (Dirichlet condition)
            heat_flux: Heat flux at wall [W/m²] (Neumann condition)
            name: Name for identification

        Note:
            Velocity is always zero at wall (no-slip condition).
            Specify either temperature or heat_flux, not both.
        """
        super().__init__(BoundaryType.WALL, name)

        if temperature is not None and heat_flux is not None:
            raise ValueError("Cannot specify both temperature and heat_flux")

        self.temperature = temperature
        self.heat_flux = heat_flux

    def __repr__(self) -> str:
        if self.temperature is not None:
            return f"WallBC(T={self.temperature} K, no-slip)"
        elif self.heat_flux is not None:
            return f"WallBC(q={self.heat_flux} W/m², no-slip)"
        else:
            return "WallBC(adiabatic, no-slip)"


class SymmetryBC(BoundaryCondition):
    """Symmetry boundary condition."""

    def __init__(self, name: str = "symmetry"):
        """
        Initialize symmetry boundary condition.

        Args:
            name: Name for identification

        Note:
            At symmetry: normal velocity = 0, normal gradients = 0
        """
        super().__init__(BoundaryType.SYMMETRY, name)

    def __repr__(self) -> str:
        return "SymmetryBC(normal_velocity=0, normal_gradients=0)"
