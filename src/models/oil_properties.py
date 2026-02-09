"""
Oil properties module.

This module provides oil characterization based on API gravity and implements
temperature-dependent properties for different oil types.
"""

from typing import Optional
import numpy as np
from .viscosity import BeggsRobinsonViscosity, ViscosityModel


class OilType:
    """Oil classification based on API gravity."""
    EXTRA_HEAVY = "extra_heavy"  # API < 10
    HEAVY = "heavy"  # 10 < API < 22.3
    MEDIUM = "medium"  # 22.3 < API < 31.1
    LIGHT = "light"  # API > 31.1


class OilFluid:
    """Oil fluid with temperature-dependent properties."""

    def __init__(
        self,
        api_gravity: float,
        temperature_ref: float = 288.15,  # 15°C
        viscosity_model: Optional[ViscosityModel] = None,
        name: str = "crude_oil"
    ):
        """
        Initialize oil fluid based on API gravity.

        Args:
            api_gravity: API gravity [degrees API]
            temperature_ref: Reference temperature for density [K]
            viscosity_model: Custom viscosity model (if None, uses Beggs-Robinson)
            name: Name of the oil

        Raises:
            ValueError: If API gravity is not positive
        """
        if api_gravity <= 0:
            raise ValueError("API gravity must be positive")

        self.api_gravity = api_gravity
        self.temperature_ref = temperature_ref
        self.name = name

        # Classify oil type
        self.oil_type = self._classify_oil(api_gravity)

        # Set viscosity model
        if viscosity_model is None:
            self.viscosity_model = BeggsRobinsonViscosity(api_gravity)
        else:
            self.viscosity_model = viscosity_model

        # Calculate reference density from API gravity
        # ρ(60°F) = 141.5 / (131.5 + API) * 999 kg/m³
        self._density_ref = 141.5 / (131.5 + api_gravity) * 999.0

        # Thermal properties (typical values for crude oil)
        self._thermal_conductivity = 0.13  # W/(m·K)
        self._specific_heat = 2000.0  # J/(kg·K)

        # Pour point estimation (approximate correlation)
        self.pour_point = self._estimate_pour_point(api_gravity)

    @staticmethod
    def _classify_oil(api_gravity: float) -> str:
        """Classify oil based on API gravity."""
        if api_gravity < 10:
            return OilType.EXTRA_HEAVY
        elif api_gravity < 22.3:
            return OilType.HEAVY
        elif api_gravity < 31.1:
            return OilType.MEDIUM
        else:
            return OilType.LIGHT

    @staticmethod
    def _estimate_pour_point(api_gravity: float) -> float:
        """
        Estimate pour point from API gravity (rough correlation).

        Args:
            api_gravity: API gravity

        Returns:
            Pour point temperature [K]
        """
        # Heavier oils have higher pour points
        # This is a simplified correlation
        pour_point_C = 50 - 1.5 * api_gravity
        return pour_point_C + 273.15

    def density(self, temperature: float = None) -> float:
        """
        Calculate density at given temperature.

        Args:
            temperature: Temperature [K] (if None, uses reference temperature)

        Returns:
            Density [kg/m³]

        Note:
            Uses simplified thermal expansion: ρ(T) = ρ_ref / (1 + β*(T - T_ref))
            where β ≈ 7e-4 K⁻¹ for crude oil
        """
        if temperature is None:
            return self._density_ref

        beta = 7e-4  # Thermal expansion coefficient [K⁻¹]
        return self._density_ref / (1 + beta * (temperature - self.temperature_ref))

    def dynamic_viscosity(self, temperature: float) -> float:
        """
        Calculate dynamic viscosity at given temperature.

        Args:
            temperature: Temperature [K]

        Returns:
            Dynamic viscosity [Pa·s]
        """
        return self.viscosity_model.calculate(temperature)

    def kinematic_viscosity(self, temperature: float) -> float:
        """
        Calculate kinematic viscosity at given temperature.

        Args:
            temperature: Temperature [K]

        Returns:
            Kinematic viscosity [m²/s]
        """
        return self.dynamic_viscosity(temperature) / self.density(temperature)

    def thermal_conductivity(self, temperature: float = None) -> float:
        """
        Get thermal conductivity.

        Args:
            temperature: Temperature [K] (currently not used)

        Returns:
            Thermal conductivity [W/(m·K)]
        """
        return self._thermal_conductivity

    def specific_heat(self, temperature: float = None) -> float:
        """
        Get specific heat capacity.

        Args:
            temperature: Temperature [K] (currently not used)

        Returns:
            Specific heat capacity [J/(kg·K)]
        """
        return self._specific_heat

    def reynolds_number(
        self,
        velocity: float,
        diameter: float,
        temperature: float
    ) -> float:
        """
        Calculate Reynolds number.

        Args:
            velocity: Velocity [m/s]
            diameter: Characteristic length (pipe diameter) [m]
            temperature: Temperature [K]

        Returns:
            Reynolds number [-]
        """
        rho = self.density(temperature)
        mu = self.dynamic_viscosity(temperature)
        return rho * velocity * diameter / mu

    def prandtl_number(self, temperature: float = None) -> float:
        """
        Calculate Prandtl number.

        Args:
            temperature: Temperature [K]

        Returns:
            Prandtl number [-]
        """
        mu = self.dynamic_viscosity(temperature) if temperature else 0.01
        cp = self.specific_heat(temperature)
        k = self.thermal_conductivity(temperature)
        return mu * cp / k

    def __repr__(self) -> str:
        return (
            f"OilFluid(name='{self.name}', API={self.api_gravity}°, "
            f"type={self.oil_type}, "
            f"ρ={self.density():.1f} kg/m³)"
        )


# Pre-configured oil types
def create_light_oil(name: str = "light_crude") -> OilFluid:
    """Create light crude oil (API ≈ 35°)."""
    return OilFluid(api_gravity=35.0, name=name)


def create_medium_oil(name: str = "medium_crude") -> OilFluid:
    """Create medium crude oil (API ≈ 27°)."""
    return OilFluid(api_gravity=27.0, name=name)


def create_heavy_oil(name: str = "heavy_crude") -> OilFluid:
    """Create heavy crude oil (API ≈ 15°)."""
    return OilFluid(api_gravity=15.0, name=name)


def create_extra_heavy_oil(name: str = "extra_heavy_crude") -> OilFluid:
    """Create extra heavy crude oil (API ≈ 8°)."""
    return OilFluid(api_gravity=8.0, name=name)
