"""
Computational domain module.

This module defines the computational domain combining geometry and grid.
"""

from typing import Optional
from ..core.grid import Grid1D, Grid2DAxisymmetric
from .pipe import Pipe


class Domain:
    """Computational domain for CFD simulation."""

    def __init__(
        self,
        pipe: Pipe,
        grid: Optional[Grid1D] = None,
        name: str = "domain"
    ):
        """
        Initialize computational domain.

        Args:
            pipe: Pipe geometry
            grid: Computational grid (if None, creates default)
            name: Domain name
        """
        self.pipe = pipe
        self.name = name

        # Create default grid if not provided
        if grid is None:
            self.grid = Grid1D(length=pipe.length, n_cells=100)
        else:
            self.grid = grid

    def __repr__(self) -> str:
        return f"Domain(name='{self.name}', {self.pipe}, {self.grid})"


class Domain2DAxisymmetric:
    """Two-dimensional axisymmetric computational domain."""

    def __init__(
        self,
        pipe: Pipe,
        n_radial: int = 50,
        n_axial: int = 100,
        wall_refinement: bool = True,
        name: str = "domain_2d"
    ):
        """
        Initialize 2D axisymmetric computational domain.

        Args:
            pipe: Pipe geometry
            n_radial: Number of cells in radial direction
            n_axial: Number of cells in axial direction
            wall_refinement: Enable wall refinement
            name: Domain name
        """
        self.pipe = pipe
        self.name = name

        # Create 2D grid
        self.grid = Grid2DAxisymmetric(
            radius=pipe.radius(),
            length=pipe.length,
            n_radial=n_radial,
            n_axial=n_axial,
            wall_refinement=wall_refinement
        )

    def __repr__(self) -> str:
        return f"Domain2DAxisymmetric(name='{self.name}', {self.pipe}, {self.grid})"
