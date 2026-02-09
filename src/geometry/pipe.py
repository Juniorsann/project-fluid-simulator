"""
Pipe geometry module.

This module defines pipe geometry and material properties for CFD simulations.
"""

from typing import Optional
import numpy as np


class PipeMaterial:
    """Pipe material properties."""

    # Common materials
    CARBON_STEEL = "carbon_steel"
    STAINLESS_STEEL = "stainless_steel"
    COPPER = "copper"
    PVC = "pvc"

    # Thermal conductivity [W/(m·K)]
    THERMAL_CONDUCTIVITY = {
        CARBON_STEEL: 50.0,
        STAINLESS_STEEL: 16.0,
        COPPER: 400.0,
        PVC: 0.19,
    }

    # Absolute roughness [m]
    ROUGHNESS = {
        CARBON_STEEL: 0.000046,  # 0.046 mm
        STAINLESS_STEEL: 0.000015,  # 0.015 mm
        COPPER: 0.0000015,  # 0.0015 mm
        PVC: 0.0000015,  # 0.0015 mm (smooth)
    }


class Pipe:
    """Cylindrical pipe geometry."""

    def __init__(
        self,
        diameter: float,
        length: float,
        roughness: Optional[float] = None,
        material: str = PipeMaterial.CARBON_STEEL,
        wall_thickness: Optional[float] = None,
        insulation_thickness: float = 0.0,
        insulation_conductivity: float = 0.04,
        ambient_temperature: float = 288.15,
        name: str = "pipe"
    ):
        """
        Initialize pipe geometry.

        Args:
            diameter: Internal diameter [m]
            length: Pipe length [m]
            roughness: Absolute roughness [m] (if None, uses material default)
            material: Pipe material
            wall_thickness: Wall thickness [m]
            insulation_thickness: Insulation thickness [m]
            insulation_conductivity: Insulation thermal conductivity [W/(m·K)]
            ambient_temperature: Ambient/external temperature [K]
            name: Pipe name

        Raises:
            ValueError: If diameter or length is not positive
        """
        if diameter <= 0:
            raise ValueError("Diameter must be positive")
        if length <= 0:
            raise ValueError("Length must be positive")

        self.diameter = diameter
        self.length = length
        self.material = material
        self.name = name
        self.ambient_temperature = ambient_temperature

        # Set roughness
        if roughness is not None:
            self.roughness = roughness
        else:
            self.roughness = PipeMaterial.ROUGHNESS.get(material, 0.000046)

        # Wall properties
        self.wall_thickness = wall_thickness if wall_thickness else diameter * 0.05
        self.wall_conductivity = PipeMaterial.THERMAL_CONDUCTIVITY.get(material, 50.0)

        # Insulation
        self.insulation_thickness = insulation_thickness
        self.insulation_conductivity = insulation_conductivity

    def cross_sectional_area(self) -> float:
        """
        Calculate cross-sectional area.

        Returns:
            Area [m²]
        """
        return np.pi * (self.diameter / 2) ** 2

    def radius(self) -> float:
        """
        Get pipe radius.

        Returns:
            Radius [m]
        """
        return self.diameter / 2

    def relative_roughness(self) -> float:
        """
        Calculate relative roughness.

        Returns:
            Relative roughness ε/D [-]
        """
        return self.roughness / self.diameter

    def volume(self) -> float:
        """
        Calculate internal volume.

        Returns:
            Volume [m³]
        """
        return self.cross_sectional_area() * self.length

    def hydraulic_diameter(self) -> float:
        """
        Calculate hydraulic diameter (for circular pipe, equals diameter).

        Returns:
            Hydraulic diameter [m]
        """
        return self.diameter

    def wetted_perimeter(self) -> float:
        """
        Calculate wetted perimeter.

        Returns:
            Perimeter [m]
        """
        return np.pi * self.diameter

    def mean_velocity(self, volumetric_flow_rate: float) -> float:
        """
        Calculate mean velocity from volumetric flow rate.

        Args:
            volumetric_flow_rate: Flow rate [m³/s]

        Returns:
            Mean velocity [m/s]
        """
        return volumetric_flow_rate / self.cross_sectional_area()

    def volumetric_flow_rate(self, mean_velocity: float) -> float:
        """
        Calculate volumetric flow rate from mean velocity.

        Args:
            mean_velocity: Mean velocity [m/s]

        Returns:
            Volumetric flow rate [m³/s]
        """
        return mean_velocity * self.cross_sectional_area()

    def overall_heat_transfer_coefficient(
        self,
        internal_h: float,
        external_h: float = 10.0
    ) -> float:
        """
        Calculate overall heat transfer coefficient including pipe wall and insulation.

        Args:
            internal_h: Internal convection coefficient [W/(m²·K)]
            external_h: External convection coefficient [W/(m²·K)]

        Returns:
            Overall heat transfer coefficient [W/(m²·K)]

        Note:
            Based on resistance network:
            1/U = 1/h_i + R_wall + R_insulation + 1/h_o
        """
        r_i = self.diameter / 2
        r_o = r_i + self.wall_thickness
        r_ins = r_o + self.insulation_thickness

        # Resistances per unit length
        R_conv_i = 1 / (internal_h * 2 * np.pi * r_i)
        R_wall = np.log(r_o / r_i) / (2 * np.pi * self.wall_conductivity)
        R_insulation = 0.0
        if self.insulation_thickness > 0:
            R_insulation = np.log(r_ins / r_o) / (2 * np.pi * self.insulation_conductivity)
        R_conv_o = 1 / (external_h * 2 * np.pi * r_ins)

        R_total = R_conv_i + R_wall + R_insulation + R_conv_o
        U = 1 / (R_total * 2 * np.pi * r_i)

        return U

    def __repr__(self) -> str:
        return (
            f"Pipe(D={self.diameter*1000:.1f}mm, L={self.length:.1f}m, "
            f"material={self.material}, ε={self.roughness*1e6:.1f}μm)"
        )
