"""
Computational grid module.

This module provides grid generation capabilities for CFD simulations,
including 1D uniform grids and 2D axisymmetric grids.
"""

from typing import Tuple, Optional
import numpy as np


class Grid1D:
    """One-dimensional uniform grid."""

    def __init__(self, length: float, n_cells: int):
        """
        Initialize 1D uniform grid.

        Args:
            length: Total length of the domain [m]
            n_cells: Number of cells

        Raises:
            ValueError: If length <= 0 or n_cells < 2
        """
        if length <= 0:
            raise ValueError("Length must be positive")
        if n_cells < 2:
            raise ValueError("Number of cells must be at least 2")

        self.length = length
        self.n_cells = n_cells
        self.dx = length / n_cells

        # Cell centers
        self.x = np.linspace(self.dx / 2, length - self.dx / 2, n_cells)

        # Cell faces (boundaries between cells)
        self.x_faces = np.linspace(0, length, n_cells + 1)

    def __repr__(self) -> str:
        return f"Grid1D(length={self.length}m, n_cells={self.n_cells}, dx={self.dx:.6f}m)"


class Grid2DAxisymmetric:
    """Two-dimensional axisymmetric grid (r, z)."""

    def __init__(
        self,
        radius: float,
        length: float,
        n_radial: int,
        n_axial: int,
        wall_refinement: bool = True,
        wall_refinement_ratio: float = 0.1
    ):
        """
        Initialize 2D axisymmetric grid.

        Args:
            radius: Maximum radius [m]
            length: Axial length [m]
            n_radial: Number of cells in radial direction
            n_axial: Number of cells in axial direction
            wall_refinement: If True, refine grid near wall
            wall_refinement_ratio: Ratio of smallest to largest cell size near wall

        Raises:
            ValueError: If dimensions or cell counts are invalid
        """
        if radius <= 0 or length <= 0:
            raise ValueError("Radius and length must be positive")
        if n_radial < 2 or n_axial < 2:
            raise ValueError("Number of cells must be at least 2 in each direction")

        self.radius = radius
        self.length = length
        self.n_radial = n_radial
        self.n_axial = n_axial

        # Axial direction (uniform spacing)
        self.dz = length / n_axial
        self.z = np.linspace(self.dz / 2, length - self.dz / 2, n_axial)
        self.z_faces = np.linspace(0, length, n_axial + 1)

        # Radial direction (with optional wall refinement)
        if wall_refinement:
            self.r_faces = self._create_refined_radial_grid(
                radius, n_radial, wall_refinement_ratio
            )
        else:
            self.r_faces = np.linspace(0, radius, n_radial + 1)

        self.r = (self.r_faces[:-1] + self.r_faces[1:]) / 2
        self.dr = np.diff(self.r_faces)

        # Create 2D meshgrid
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing='ij')

    @staticmethod
    def _create_refined_radial_grid(
        radius: float,
        n_cells: int,
        refinement_ratio: float
    ) -> np.ndarray:
        """
        Create radial grid with refinement near wall using geometric progression.

        Args:
            radius: Maximum radius
            n_cells: Number of cells
            refinement_ratio: Ratio of smallest to largest cell size

        Returns:
            Array of radial face positions
        """
        # Geometric progression for wall refinement
        # Smallest cell at wall, largest at center
        ratio = (1 / refinement_ratio) ** (1 / (n_cells - 1))
        r_normalized = np.zeros(n_cells + 1)

        for i in range(n_cells + 1):
            if i == 0:
                r_normalized[i] = 0
            else:
                r_normalized[i] = r_normalized[i - 1] + ratio ** (n_cells - i)

        # Normalize to radius
        r_faces = radius * r_normalized / r_normalized[-1]
        # Reverse to have refinement at wall (r = radius)
        r_faces = radius - r_faces[::-1]

        return r_faces

    def cell_volume(self, i_radial: int, i_axial: int) -> float:
        """
        Calculate volume of a cell in axisymmetric coordinates.

        Args:
            i_radial: Radial cell index
            i_axial: Axial cell index

        Returns:
            Cell volume [m³]
        """
        r_inner = self.r_faces[i_radial]
        r_outer = self.r_faces[i_radial + 1]
        dz = self.dz
        # Volume = π * (r_outer² - r_inner²) * dz
        return np.pi * (r_outer ** 2 - r_inner ** 2) * dz

    def __repr__(self) -> str:
        return (
            f"Grid2DAxisymmetric(radius={self.radius}m, length={self.length}m, "
            f"n_radial={self.n_radial}, n_axial={self.n_axial})"
        )
