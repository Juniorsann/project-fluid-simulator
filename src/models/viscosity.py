"""
Viscosity models module.

This module implements various viscosity-temperature correlations used in
the oil and gas industry including Walther (ASTM D341), Andrade, and
Beggs-Robinson equations.
"""

from typing import Tuple
import numpy as np
from numba import jit


@jit(nopython=True)
def walther_equation(
    temperature: float,
    A: float,
    B: float
) -> float:
    """
    Calculate kinematic viscosity using Walther equation (ASTM D341).

    Equation: log10(log10(ν + 0.7)) = A - B*log10(T)

    Args:
        temperature: Temperature [K]
        A: Walther constant A
        B: Walther constant B

    Returns:
        Kinematic viscosity [m²/s]
    """
    log_log_nu = A - B * np.log10(temperature)
    log_nu = 10 ** log_log_nu
    nu = 10 ** log_nu - 0.7
    return nu * 1e-6  # Convert from cSt to m²/s


def fit_walther_parameters(
    temperature_1: float,
    viscosity_1: float,
    temperature_2: float,
    viscosity_2: float
) -> Tuple[float, float]:
    """
    Fit Walther equation parameters from two known viscosity-temperature points.

    Args:
        temperature_1: First temperature [K]
        viscosity_1: Kinematic viscosity at T1 [m²/s]
        temperature_2: Second temperature [K]
        viscosity_2: Kinematic viscosity at T2 [m²/s]

    Returns:
        Tuple of (A, B) parameters for Walther equation
    """
    # Convert to cSt
    nu1_cst = viscosity_1 * 1e6
    nu2_cst = viscosity_2 * 1e6

    # Calculate log-log values
    y1 = np.log10(np.log10(nu1_cst + 0.7))
    y2 = np.log10(np.log10(nu2_cst + 0.7))

    x1 = np.log10(temperature_1)
    x2 = np.log10(temperature_2)

    # Linear fit: y = A - B*x
    B = (y1 - y2) / (x2 - x1)
    A = y1 + B * x1

    return A, B


@jit(nopython=True)
def andrade_equation(
    temperature: float,
    A: float,
    B: float
) -> float:
    """
    Calculate dynamic viscosity using Andrade equation.

    Equation: μ = A * exp(B/T)

    Args:
        temperature: Temperature [K]
        A: Andrade constant A [Pa·s]
        B: Andrade constant B [K]

    Returns:
        Dynamic viscosity [Pa·s]
    """
    return A * np.exp(B / temperature)


def fit_andrade_parameters(
    temperature_1: float,
    viscosity_1: float,
    temperature_2: float,
    viscosity_2: float
) -> Tuple[float, float]:
    """
    Fit Andrade equation parameters from two known viscosity-temperature points.

    Args:
        temperature_1: First temperature [K]
        viscosity_1: Dynamic viscosity at T1 [Pa·s]
        temperature_2: Second temperature [K]
        viscosity_2: Dynamic viscosity at T2 [Pa·s]

    Returns:
        Tuple of (A, B) parameters for Andrade equation
    """
    # ln(μ) = ln(A) + B/T
    # Linear fit in ln(μ) vs 1/T
    y1 = np.log(viscosity_1)
    y2 = np.log(viscosity_2)

    x1 = 1 / temperature_1
    x2 = 1 / temperature_2

    B = (y1 - y2) / (x1 - x2)
    ln_A = y1 - B * x1
    A = np.exp(ln_A)

    return A, B


@jit(nopython=True)
def beggs_robinson_dead_oil(
    temperature: float,
    api_gravity: float
) -> float:
    """
    Calculate dead oil viscosity using Beggs-Robinson correlation.

    Reference: Beggs, H.D. and Robinson, J.R. (1975)
    "Estimating the Viscosity of Crude Oil Systems"
    Journal of Petroleum Technology, Sept. 1975, pp. 1140-1141

    Args:
        temperature: Temperature [K]
        api_gravity: API gravity [degrees API]

    Returns:
        Dynamic viscosity [Pa·s]

    Note:
        Valid for 70°F < T < 295°F and 16 < API < 58
    """
    # Convert to Fahrenheit
    temp_F = (temperature - 273.15) * 9 / 5 + 32

    # Beggs-Robinson correlation
    z = 3.0324 - 0.02023 * api_gravity
    y = 10 ** z
    X = y * temp_F ** (-1.163)
    mu_cp = 10 ** X - 1

    # Convert from cP to Pa·s
    return mu_cp * 0.001


class ViscosityModel:
    """Base class for viscosity models."""

    def __init__(self, model_type: str = "constant"):
        """
        Initialize viscosity model.

        Args:
            model_type: Type of model ('constant', 'walther', 'andrade', 'beggs-robinson')
        """
        self.model_type = model_type

    def calculate(self, temperature: float) -> float:
        """
        Calculate viscosity at given temperature.

        Args:
            temperature: Temperature [K]

        Returns:
            Viscosity [Pa·s or m²/s depending on model]
        """
        raise NotImplementedError("Subclass must implement calculate method")


class ConstantViscosity(ViscosityModel):
    """Constant viscosity model."""

    def __init__(self, viscosity: float):
        """
        Initialize constant viscosity model.

        Args:
            viscosity: Constant viscosity value [Pa·s]
        """
        super().__init__("constant")
        self.viscosity = viscosity

    def calculate(self, temperature: float) -> float:
        """Return constant viscosity."""
        return self.viscosity


class WaltherViscosity(ViscosityModel):
    """Walther viscosity model (ASTM D341)."""

    def __init__(self, A: float, B: float):
        """
        Initialize Walther viscosity model.

        Args:
            A: Walther parameter A
            B: Walther parameter B
        """
        super().__init__("walther")
        self.A = A
        self.B = B

    def calculate(self, temperature: float) -> float:
        """Calculate kinematic viscosity using Walther equation."""
        return walther_equation(temperature, self.A, self.B)


class AndradeViscosity(ViscosityModel):
    """Andrade viscosity model."""

    def __init__(self, A: float, B: float):
        """
        Initialize Andrade viscosity model.

        Args:
            A: Andrade parameter A [Pa·s]
            B: Andrade parameter B [K]
        """
        super().__init__("andrade")
        self.A = A
        self.B = B

    def calculate(self, temperature: float) -> float:
        """Calculate dynamic viscosity using Andrade equation."""
        return andrade_equation(temperature, self.A, self.B)


class BeggsRobinsonViscosity(ViscosityModel):
    """Beggs-Robinson viscosity model for crude oil."""

    def __init__(self, api_gravity: float):
        """
        Initialize Beggs-Robinson viscosity model.

        Args:
            api_gravity: API gravity [degrees API]
        """
        super().__init__("beggs-robinson")
        self.api_gravity = api_gravity

    def calculate(self, temperature: float) -> float:
        """Calculate dynamic viscosity using Beggs-Robinson correlation."""
        return beggs_robinson_dead_oil(temperature, self.api_gravity)
