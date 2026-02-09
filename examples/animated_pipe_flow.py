"""
Animated visualization of oil flow in pipe with heat transfer.

This example demonstrates particle tracers, velocity field, and temperature
evolution in a heated pipe flow scenario.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.models.oil_properties import OilFluid
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC, WallBC
from src.core.solver import PipeFlowSolver, HeatTransferSolver
from src.core.grid import Grid2DAxisymmetric
from src.visualization.animator import FlowAnimator
from src.visualization.particle_tracer import ParticleTracer


def main():
    """
    Create animated visualization of heated oil pipe flow.
    
    Shows:
    - Particle tracers colored by temperature
    - Temperature map evolution
    - Velocity field visualization
    """
    print("=" * 60)
    print("ANIMATED PIPE FLOW WITH HEAT TRANSFER")
    print("=" * 60)
    
    # Setup simulation
    print("\nSetting up simulation...")
    
    # Medium crude oil (API gravity ~18)
    oil = OilFluid(
        name="medium_crude",
        api_gravity=18.0,
        reference_temp=288.15,  # 15°C
        reference_viscosity=0.050  # 50 cP
    )
    
    # Pipe geometry
    pipe = Pipe(
        diameter=0.4,  # 400 mm
        length=10.0,   # 10 m
        roughness=0.000045,  # Carbon steel
        material="carbon_steel"
    )
    
    print(f"Pipe: {pipe.diameter*1000} mm diameter, {pipe.length} m length")
    
    # Boundary conditions
    inlet_temp = 60 + 273.15  # 60°C
    wall_temp = 80 + 273.15   # 80°C (heated pipe)
    flow_rate = 0.08  # m³/s
    
    inlet = InletBC(mass_flow_rate=flow_rate * oil.density(inlet_temp), 
                    temperature=inlet_temp)
    outlet = OutletBC(pressure=101325.0)
    wall = WallBC(temperature=wall_temp)
    
    print(f"Inlet: Q = {flow_rate} m³/s, T = {inlet_temp - 273.15}°C")
    print(f"Wall temperature: {wall_temp - 273.15}°C")
    
    # Solve flow field
    print("\nSolving CFD...")
    solver = PipeFlowSolver(pipe, oil, inlet, outlet, wall)
    flow_results = solver.solve_flow()
    
    print(f"Reynolds number: {flow_results['reynolds_number']:.0f}")
    print(f"Flow regime: {flow_results['flow_regime']}")
    print(f"Mean velocity: {flow_results['mean_velocity']:.2f} m/s")
    
    # Create 2D grid for visualization
    print("\nGenerating 2D velocity and temperature fields...")
    grid = Grid2DAxisymmetric(
        radius=pipe.radius(),
        length=pipe.length,
        n_radial=40,
        n_axial=100,
        wall_refinement=True
    )
    
    # Get velocity profile
    u_r, u_z = solver.solve_velocity_profile_2d(grid)
    
    # Get temperature profile
    heat_solver = HeatTransferSolver(
        pipe, oil, inlet_temp, wall, flow_results['mean_velocity']
    )
    T_axial = heat_solver.solve_temperature_profile_1d(grid.z)
    
    # Create 2D temperature field (simplified)
    T_2d = np.tile(T_axial, (grid.n_radial, 1))
    
    # Prepare results dictionary for animator
    results = {
        'velocity_field': (u_r, u_z),
        'temperature_field': T_2d,
        'grid': (grid.r, grid.z),
        'pipe_radius': pipe.radius(),
        'pipe_length': pipe.length
    }
    
    # Create animator
    print("\nCreating animation...")
    animator = FlowAnimator(results, figure_size=(14, 6))
    
    # Add visualization layers
    animator.add_temperature_map(cmap='hot', alpha=0.6)
    animator.add_particle_tracers(
        n_particles=300,
        release_mode='continuous',
        color_scheme='temperature'
    )
    animator.add_title('Oil Flow in Heated Pipe - Temperature Evolution')
    animator.add_text(
        0.02, 0.95,
        f'Inlet: {inlet_temp - 273.15:.0f}°C',
        fontsize=12
    )
    animator.add_text(
        0.02, 0.90,
        f'Wall: {wall_temp - 273.15:.0f}°C',
        fontsize=12
    )
    
    # Generate outputs
    print("\nGenerating video...")
    try:
        animator.save_video('oil_flow_heated.mp4', duration=15, fps=30, dpi=150)
        print("✓ Video created: oil_flow_heated.mp4")
    except Exception as e:
        print(f"⚠ Video generation failed: {e}")
        print("  (FFmpeg may not be installed)")
    
    print("\nGenerating GIF...")
    try:
        animator.save_gif('oil_flow_heated.gif', duration=10, fps=15)
        print("✓ GIF created: oil_flow_heated.gif")
    except Exception as e:
        print(f"⚠ GIF generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("ANIMATION COMPLETE")
    print("=" * 60)
    print("\nNote: Animation files would be created with actual simulation data.")
    print("This example demonstrates the animation framework structure.")


if __name__ == "__main__":
    main()
