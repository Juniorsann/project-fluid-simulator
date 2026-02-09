"""
Basic Pipe Flow Example

This example demonstrates isothermal flow of oil through a horizontal pipe.
Calculates velocity profile, pressure drop, and flow regime.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.models.oil_properties import create_medium_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC
from src.core.solver import PipeFlowSolver, PoiseuilleFlow
from src.core.grid import Grid2DAxisymmetric
from src.visualization.plotter import plot_velocity_profile, plot_pressure_drop


def main():
    """Run basic pipe flow simulation."""
    print("=" * 60)
    print("BASIC PIPE FLOW SIMULATION")
    print("=" * 60)

    # Define pipe geometry
    pipe = Pipe(
        diameter=0.2,  # 200 mm diameter
        length=100.0,  # 100 m length
        material="carbon_steel",
        name="production_line"
    )
    print(f"\nPipe: {pipe}")

    # Define fluid (medium crude oil)
    oil = create_medium_oil("medium_crude")
    temperature = 293.15  # 20°C
    print(f"\nOil: {oil}")
    print(f"Temperature: {temperature - 273.15:.1f}°C")
    print(f"Density: {oil.density(temperature):.1f} kg/m³")
    print(f"Dynamic viscosity: {oil.dynamic_viscosity(temperature)*1000:.1f} cP")

    # Define boundary conditions
    mean_velocity = 1.0  # m/s
    inlet = InletBC(velocity=mean_velocity, temperature=temperature)
    outlet = OutletBC(pressure=101325.0)  # Atmospheric pressure
    
    print(f"\nInlet velocity: {mean_velocity} m/s")
    print(f"Outlet pressure: {outlet.pressure/1e5:.2f} bar")

    # Solve flow field
    solver = PipeFlowSolver(pipe, oil, inlet, outlet)
    results = solver.solve_flow()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Reynolds number: {results['reynolds_number']:.0f}")
    print(f"Flow regime: {results['flow_regime']}")
    print(f"Pressure drop: {results['pressure_drop']/1e5:.4f} bar")
    print(f"Pressure gradient: {results['pressure_drop']/pipe.length:.2f} Pa/m")

    # Calculate volumetric and mass flow rates
    Q = mean_velocity * pipe.cross_sectional_area()
    m_dot = Q * oil.density(temperature)
    print(f"\nVolumetric flow rate: {Q*3600:.1f} m³/h")
    print(f"Mass flow rate: {m_dot*3600:.1f} kg/h")

    # Calculate velocity profile (for laminar flow)
    if results['flow_regime'] == 'laminar':
        print("\nGenerating velocity profile (Poiseuille flow)...")
        poiseuille = PoiseuilleFlow(pipe, oil, temperature)
        
        # Radial positions
        r = np.linspace(0, pipe.radius(), 100)
        u = poiseuille.velocity_profile(r, mean_velocity)
        
        # Plot velocity profile
        plot_velocity_profile(
            r, u, pipe.radius(),
            title=f"Laminar Velocity Profile - {oil.name}",
            save_path="basic_pipe_flow_velocity.png"
        )
        print("Velocity profile saved to 'basic_pipe_flow_velocity.png'")

    # Plot pressure distribution
    z = np.linspace(0, pipe.length, 100)
    p_inlet = outlet.pressure + results['pressure_drop']
    pressure = p_inlet - (results['pressure_drop'] / pipe.length) * z
    
    plot_pressure_drop(
        z, pressure,
        title=f"Pressure Distribution - {pipe.name}",
        save_path="basic_pipe_flow_pressure.png"
    )
    print("Pressure distribution saved to 'basic_pipe_flow_pressure.png'")

    # Calculate power required for pumping
    power = results['pressure_drop'] * Q  # Watts
    print(f"\nPumping power required: {power:.2f} W ({power/1000:.3f} kW)")

    # 2D velocity profile
    if results['flow_regime'] == 'laminar':
        print("\nGenerating 2D velocity field...")
        grid_2d = Grid2DAxisymmetric(
            radius=pipe.radius(),
            length=min(pipe.length, 1.0),  # Show only 1m for visualization
            n_radial=50,
            n_axial=100,
            wall_refinement=True
        )
        
        u_r, u_z = solver.solve_velocity_profile_2d(grid_2d)
        
        # Import and use 2D plotter
        from src.visualization.plotter import plot_2d_contour
        plot_2d_contour(
            grid_2d.r, grid_2d.z, u_z,
            field_name="Axial Velocity [m/s]",
            title="2D Velocity Field (Axisymmetric)",
            save_path="basic_pipe_flow_2d.png"
        )
        print("2D velocity field saved to 'basic_pipe_flow_2d.png'")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
