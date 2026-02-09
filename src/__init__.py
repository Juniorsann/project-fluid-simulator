"""
CFD Industrial Simulator - Professional simulator for viscous oil flow in pipelines.

This package provides tools for computational fluid dynamics simulations 
focused on the oil and gas industry.
"""

__version__ = "0.1.0"
__author__ = "Juniorsann"

from . import core, models, geometry, visualization, utils

__all__ = ["core", "models", "geometry", "visualization", "utils"]
