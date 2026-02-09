"""
Validators module for input validation.

This module provides validation functions for simulation parameters.
"""

from typing import Union, Tuple


def validate_positive(value: float, name: str = "value") -> float:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate
        name: Name of the parameter

    Returns:
        The validated value

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative(value: float, name: str = "value") -> float:
    """
    Validate that a value is non-negative.

    Args:
        value: Value to validate
        name: Name of the parameter

    Returns:
        The validated value

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value"
) -> float:
    """
    Validate that a value is within a specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter

    Returns:
        The validated value

    Raises:
        ValueError: If value is outside the range
    """
    if value < min_val or value > max_val:
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )
    return value


def validate_temperature(temperature: float, min_temp: float = 0) -> float:
    """
    Validate temperature value.

    Args:
        temperature: Temperature in Kelvin
        min_temp: Minimum allowed temperature [K]

    Returns:
        The validated temperature

    Raises:
        ValueError: If temperature is invalid
    """
    if temperature < min_temp:
        raise ValueError(
            f"Temperature must be at least {min_temp} K, got {temperature} K"
        )
    if temperature > 1000:
        raise ValueError(
            f"Temperature {temperature} K seems unreasonably high for oil flow"
        )
    return temperature


def validate_api_gravity(api_gravity: float) -> float:
    """
    Validate API gravity value.

    Args:
        api_gravity: API gravity in degrees

    Returns:
        The validated API gravity

    Raises:
        ValueError: If API gravity is invalid
    """
    if api_gravity <= 0:
        raise ValueError("API gravity must be positive")
    if api_gravity > 100:
        raise ValueError(
            f"API gravity {api_gravity}° seems unreasonably high"
        )
    return api_gravity


def validate_reynolds_number(reynolds: float) -> Tuple[float, str]:
    """
    Validate Reynolds number and determine flow regime.

    Args:
        reynolds: Reynolds number

    Returns:
        Tuple of (validated Reynolds number, flow regime)

    Raises:
        ValueError: If Reynolds number is invalid
    """
    if reynolds < 0:
        raise ValueError("Reynolds number cannot be negative")

    if reynolds < 2300:
        regime = "laminar"
    elif reynolds < 4000:
        regime = "transitional"
    else:
        regime = "turbulent"

    return reynolds, regime


def validate_pressure(pressure: float, name: str = "pressure") -> float:
    """
    Validate pressure value.

    Args:
        pressure: Pressure [Pa]
        name: Name of the parameter

    Returns:
        The validated pressure

    Raises:
        ValueError: If pressure is invalid
    """
    if pressure < 0:
        raise ValueError(f"{name} cannot be negative")
    if pressure > 1e8:  # 1000 bar
        raise ValueError(
            f"{name} {pressure/1e5:.1f} bar seems unreasonably high"
        )
    return pressure


def validate_velocity(velocity: float) -> float:
    """
    Validate velocity value.

    Args:
        velocity: Velocity [m/s]

    Returns:
        The validated velocity

    Raises:
        ValueError: If velocity is invalid
    """
    if velocity < 0:
        raise ValueError("Velocity cannot be negative")
    if velocity > 50:
        raise ValueError(
            f"Velocity {velocity} m/s seems unreasonably high for pipe flow"
        )
    return velocity


def validate_grid_size(n_cells: int, name: str = "n_cells") -> int:
    """
    Validate grid size.

    Args:
        n_cells: Number of cells
        name: Name of the parameter

    Returns:
        The validated grid size

    Raises:
        ValueError: If grid size is invalid
    """
    if n_cells < 2:
        raise ValueError(f"{name} must be at least 2")
    if n_cells > 10000:
        raise ValueError(
            f"{name} = {n_cells} may be too large, consider using < 10000"
        )
    return n_cells
