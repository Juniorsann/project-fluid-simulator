"""
Turbulence models module.

This module implements turbulence models for CFD simulations.
Currently includes basic mixing length model and constants for k-epsilon.
"""

import numpy as np
from typing import Optional


class TurbulenceModel:
    """Base class for turbulence models."""

    def __init__(self, model_type: str = "laminar"):
        """
        Initialize turbulence model.

        Args:
            model_type: Type of model ('laminar', 'mixing_length', 'k_epsilon')
        """
        self.model_type = model_type

    def turbulent_viscosity(
        self,
        velocity_gradient: float,
        wall_distance: float
    ) -> float:
        """
        Calculate turbulent viscosity.

        Args:
            velocity_gradient: Velocity gradient [1/s]
            wall_distance: Distance from wall [m]

        Returns:
            Turbulent viscosity [Pa·s]
        """
        return 0.0  # Laminar default


class MixingLengthModel(TurbulenceModel):
    """Prandtl mixing length turbulence model."""

    def __init__(self, kappa: float = 0.41):
        """
        Initialize mixing length model.

        Args:
            kappa: Von Karman constant (default: 0.41)
        """
        super().__init__("mixing_length")
        self.kappa = kappa

    def mixing_length(self, wall_distance: float) -> float:
        """
        Calculate mixing length.

        Args:
            wall_distance: Distance from wall [m]

        Returns:
            Mixing length [m]
        """
        return self.kappa * wall_distance

    def turbulent_viscosity(
        self,
        density: float,
        velocity_gradient: float,
        wall_distance: float
    ) -> float:
        """
        Calculate turbulent viscosity using mixing length model.

        Args:
            density: Fluid density [kg/m³]
            velocity_gradient: Velocity gradient [1/s]
            wall_distance: Distance from wall [m]

        Returns:
            Turbulent viscosity [Pa·s]
        """
        l_m = self.mixing_length(wall_distance)
        return density * l_m ** 2 * abs(velocity_gradient)


class KEpsilonConstants:
    """Constants for standard k-epsilon turbulence model."""

    C_mu = 0.09  # Model constant
    C_1 = 1.44  # Model constant for epsilon equation
    C_2 = 1.92  # Model constant for epsilon equation
    sigma_k = 1.0  # Turbulent Prandtl number for k
    sigma_epsilon = 1.3  # Turbulent Prandtl number for epsilon

    @staticmethod
    def turbulent_viscosity(density: float, k: float, epsilon: float) -> float:
        """
        Calculate turbulent viscosity from k and epsilon.

        Args:
            density: Fluid density [kg/m³]
            k: Turbulent kinetic energy [m²/s²]
            epsilon: Turbulent dissipation rate [m²/s³]

        Returns:
            Turbulent viscosity [Pa·s]
        """
        if epsilon <= 0:
            return 0.0
        mu_t = KEpsilonConstants.C_mu * density * k ** 2 / epsilon
        return mu_t


def friction_factor_laminar(reynolds: float) -> float:
    """
    Calculate friction factor for laminar flow.

    Args:
        reynolds: Reynolds number

    Returns:
        Darcy friction factor

    Note:
        For laminar pipe flow: f = 64/Re
    """
    return 64.0 / reynolds


def friction_factor_turbulent(
    reynolds: float,
    relative_roughness: float = 0.0,
    max_iterations: int = 20
) -> float:
    """
    Calculate friction factor for turbulent flow using Colebrook-White equation.

    Solves iteratively: 1/√f = -2*log10(ε/(3.7*D) + 2.51/(Re*√f))

    Args:
        reynolds: Reynolds number
        relative_roughness: Relative roughness ε/D
        max_iterations: Maximum iterations for convergence

    Returns:
        Darcy friction factor
    """
    # Initial guess using Blasius correlation for smooth pipes
    f = 0.316 / reynolds ** 0.25

    # Iterative solution of Colebrook-White
    for _ in range(max_iterations):
        f_new = 1 / (
            -2 * np.log10(relative_roughness / 3.7 + 2.51 / (reynolds * np.sqrt(f)))
        ) ** 2

        if abs(f_new - f) < 1e-6:
            break
        f = f_new

    return f


def friction_factor(
    reynolds: float,
    relative_roughness: float = 0.0
) -> float:
    """
    Calculate Darcy friction factor for pipe flow.

    Args:
        reynolds: Reynolds number
        relative_roughness: Relative roughness ε/D

    Returns:
        Darcy friction factor

    Note:
        Automatically switches between laminar (Re < 2300) and turbulent
    """
    if reynolds < 2300:
        return friction_factor_laminar(reynolds)
    else:
        return friction_factor_turbulent(reynolds, relative_roughness)
