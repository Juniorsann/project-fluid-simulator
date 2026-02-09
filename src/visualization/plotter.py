"""
Plotting module for visualization of CFD results.

This module provides functions to plot velocity profiles, temperature distributions,
pressure drops, and other flow field quantities.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_velocity_profile(
    r: np.ndarray,
    u: np.ndarray,
    radius: float,
    title: str = "Velocity Profile",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot radial velocity profile.

    Args:
        r: Radial positions [m]
        u: Velocity values [m/s]
        radius: Pipe radius [m]
        title: Plot title
        save_path: Path to save figure (if None, shows plot)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(r * 1000, u, 'b-', linewidth=2, label='Velocity')
    ax.axvline(x=radius * 1000, color='r', linestyle='--', alpha=0.5, label='Wall')
    ax.fill_betweenx([0, max(u)], 0, radius * 1000, alpha=0.1, color='gray')

    ax.set_xlabel('Radial Position [mm]', fontsize=12)
    ax.set_ylabel('Velocity [m/s]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_temperature_profile(
    z: np.ndarray,
    T: np.ndarray,
    length: float,
    title: str = "Temperature Profile",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot axial temperature profile.

    Args:
        z: Axial positions [m]
        T: Temperature values [K]
        length: Pipe length [m]
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to Celsius for display
    T_celsius = T - 273.15

    ax.plot(z, T_celsius, 'r-', linewidth=2, label='Temperature')
    ax.fill_between(z, 0, T_celsius, alpha=0.2, color='red')

    ax.set_xlabel('Axial Position [m]', fontsize=12)
    ax.set_ylabel('Temperature [°C]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, length])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_viscosity_temperature(
    temperatures: np.ndarray,
    viscosities: np.ndarray,
    oil_names: Optional[list] = None,
    title: str = "Viscosity vs Temperature",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot viscosity as function of temperature.

    Args:
        temperatures: Temperature values [K] or [°C]
        viscosities: Viscosity values [Pa·s]
        oil_names: Names for each oil (if multiple)
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if temperatures are 2D (multiple oils)
    if len(viscosities.shape) == 1:
        viscosities = viscosities.reshape(1, -1)
        temperatures = temperatures.reshape(1, -1) if len(temperatures.shape) == 2 else temperatures

    if len(temperatures.shape) == 1:
        temperatures = np.tile(temperatures, (viscosities.shape[0], 1))

    # Plot each oil
    for i in range(viscosities.shape[0]):
        label = oil_names[i] if oil_names and i < len(oil_names) else f'Oil {i+1}'
        # Convert viscosity to cP for better readability
        visc_cp = viscosities[i] * 1000
        ax.semilogy(temperatures[i] - 273.15, visc_cp, linewidth=2, marker='o', label=label)

    ax.set_xlabel('Temperature [°C]', fontsize=12)
    ax.set_ylabel('Dynamic Viscosity [cP]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_pressure_drop(
    z: np.ndarray,
    pressure: np.ndarray,
    title: str = "Pressure Distribution",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot pressure distribution along pipe.

    Args:
        z: Axial positions [m]
        pressure: Pressure values [Pa]
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to bar
    pressure_bar = pressure / 1e5

    ax.plot(z, pressure_bar, 'g-', linewidth=2, label='Pressure')
    ax.fill_between(z, pressure_bar.min(), pressure_bar, alpha=0.2, color='green')

    ax.set_xlabel('Axial Position [m]', fontsize=12)
    ax.set_ylabel('Pressure [bar]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_2d_contour(
    r: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    field_name: str = "Field",
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot 2D contour of flow field in axisymmetric coordinates.

    Args:
        r: Radial coordinates [m]
        z: Axial coordinates [m]
        field: Field values to plot
        field_name: Name of the field
        cmap: Colormap name
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    R, Z = np.meshgrid(r, z, indexing='ij')

    # Create contour plot
    levels = 20
    contour = ax.contourf(Z * 1000, R * 1000, field, levels=levels, cmap=cmap)
    ax.contour(Z * 1000, R * 1000, field, levels=levels, colors='k', alpha=0.2, linewidths=0.5)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(field_name, fontsize=12)

    ax.set_xlabel('Axial Position [mm]', fontsize=12)
    ax.set_ylabel('Radial Position [mm]', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_reynolds_number_evolution(
    positions: np.ndarray,
    reynolds: np.ndarray,
    title: str = "Reynolds Number Evolution",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot Reynolds number along pipe length.

    Args:
        positions: Axial positions [m]
        reynolds: Reynolds number values
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(positions, reynolds, 'b-', linewidth=2, label='Reynolds Number')

    # Add regime boundaries
    ax.axhline(y=2300, color='r', linestyle='--', label='Laminar/Transition')
    ax.axhline(y=4000, color='orange', linestyle='--', label='Transition/Turbulent')

    # Shade regions
    ax.fill_between(positions, 0, 2300, alpha=0.1, color='blue', label='Laminar')
    ax.fill_between(positions, 2300, 4000, alpha=0.1, color='yellow', label='Transitional')
    if reynolds.max() > 4000:
        ax.fill_between(positions, 4000, reynolds.max(), alpha=0.1, color='red', label='Turbulent')

    ax.set_xlabel('Axial Position [m]', fontsize=12)
    ax.set_ylabel('Reynolds Number [-]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig
