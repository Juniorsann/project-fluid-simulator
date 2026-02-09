"""
Fluid properties module.

This module defines base classes for fluid properties including density,
viscosity, thermal conductivity, and specific heat.
"""

from typing import Optional
import numpy as np


class FluidProperties:
    """Base class for fluid properties."""

    def __init__(
        self,
        density: float,
        viscosity: float,
        thermal_conductivity: Optional[float] = None,
        specific_heat: Optional[float] = None,
        name: str = "fluid"
    ):
        """
        Initialize fluid properties.

        Args:
            density: Fluid density [kg/m³]
            viscosity: Dynamic viscosity [Pa·s]
            thermal_conductivity: Thermal conductivity [W/(m·K)]
            specific_heat: Specific heat capacity [J/(kg·K)]
            name: Name of the fluid
        """
        self.density = density
        self.viscosity = viscosity
        self.thermal_conductivity = thermal_conductivity
        self.specific_heat = specific_heat
        self.name = name

    def kinematic_viscosity(self) -> float:
        """
        Calculate kinematic viscosity.

        Returns:
            Kinematic viscosity [m²/s]
        """
        return self.viscosity / self.density

    def reynolds_number(self, velocity: float, length_scale: float) -> float:
        """
        Calculate Reynolds number.

        Args:
            velocity: Characteristic velocity [m/s]
            length_scale: Characteristic length [m]

        Returns:
            Reynolds number [-]
        """
        return self.density * velocity * length_scale / self.viscosity

    def prandtl_number(self) -> float:
        """
        Calculate Prandtl number.

        Returns:
            Prandtl number [-]

        Raises:
            ValueError: If thermal properties are not defined
        """
        if self.thermal_conductivity is None or self.specific_heat is None:
            raise ValueError("Thermal properties not defined")
        return self.viscosity * self.specific_heat / self.thermal_conductivity

    def __repr__(self) -> str:
        return (
            f"FluidProperties(name='{self.name}', "
            f"density={self.density:.2f} kg/m³, "
            f"viscosity={self.viscosity:.6f} Pa·s)"
        )
