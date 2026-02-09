"""
Temperature evolution in heated pipe.

This example shows how oil temperature changes along pipe length
with steam tracing (constant wall temperature).
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


def main():
    """
    Create temperature evolution animation.
    
    Shows how oil heats up in a steam-traced pipe.
    """
    print("=" * 60)
    print("TEMPERATURE EVOLUTION IN HEATED PIPE")
    print("=" * 60)
    
    # Setup
    print("\nSetting up simulation...")
    oil = OilFluid(
        name="medium_oil",
        api_gravity=15.0,
        reference_temp=288.15,
        reference_viscosity=0.075  # 75 cP
    )
    
    pipe = Pipe(
        diameter=0.35,  # 350 mm
        length=8.0,     # 8 m
        roughness=0.000045,
        material="carbon_steel"
    )
    
    print(f"Pipe: {pipe.diameter*1000} mm × {pipe.length} m")
    
    # Boundary conditions
    inlet_temp = 45 + 273.15  # 45°C (cold inlet)
    wall_temp = 75 + 273.15   # 75°C (steam tracing)
    flow_rate = 0.06  # m³/s
    
    inlet = InletBC(
        mass_flow_rate=flow_rate * oil.density(inlet_temp),
        temperature=inlet_temp
    )
    outlet = OutletBC(pressure=101325.0)
    wall = WallBC(temperature=wall_temp)
    
    print(f"Inlet temperature: {inlet_temp - 273.15}°C")
    print(f"Wall temperature: {wall_temp - 273.15}°C (steam tracing)")
    print(f"Flow rate: {flow_rate} m³/s")
    
    # Solve flow
    print("\nSolving flow field...")
    solver = PipeFlowSolver(pipe, oil, inlet, outlet, wall)
    flow_results = solver.solve_flow()
    
    print(f"Reynolds number: {flow_results['reynolds_number']:.0f}")
    print(f"Mean velocity: {flow_results['mean_velocity']:.2f} m/s")
    
    # Create grid
    print("\nComputing temperature field...")
    grid = Grid2DAxisymmetric(
        radius=pipe.radius(),
        length=pipe.length,
        n_radial=35,
        n_axial=100,
        wall_refinement=True
    )
    
    # Get velocity field
    u_r, u_z = solver.solve_velocity_profile_2d(grid)
    
    # Get temperature profile along axis
    heat_solver = HeatTransferSolver(
        pipe, oil, inlet_temp, wall, flow_results['mean_velocity']
    )
    T_axial = heat_solver.solve_temperature_profile_1d(grid.z)
    
    # Create 2D temperature field (simplified radial profile)
    T_2d = np.zeros((grid.n_radial, grid.n_axial))
    for i in range(grid.n_axial):
        # Simplified: assume radial variation from centerline to wall
        T_center = T_axial[i]
        for j in range(grid.n_radial):
            # Linear interpolation from center to wall temperature
            r_ratio = grid.r[j] / pipe.radius()
            T_2d[j, i] = T_center + (wall_temp - T_center) * r_ratio**2
    
    print(f"Inlet temperature: {T_axial[0] - 273.15:.1f}°C")
    print(f"Outlet temperature: {T_axial[-1] - 273.15:.1f}°C")
    print(f"Temperature rise: {(T_axial[-1] - T_axial[0]):.1f}°C")
    
    # Prepare results
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
    
    # Temperature-focused visualization
    animator.add_temperature_map(cmap='RdYlBu_r', levels=30, alpha=0.8)
    animator.add_particle_tracers(
        n_particles=200,
        color_scheme='temperature'
    )
    
    # Add isotherms (lines of constant temperature)
    isotherm_temps = [50, 55, 60, 65, 70]  # °C
    animator.add_isotherms(temperatures=isotherm_temps, linewidth=2)
    
    # Annotations
    animator.add_title('Oil Heating in Steam-Traced Pipe')
    animator.add_text(0.02, 0.95, f'Inlet: {inlet_temp - 273.15:.0f}°C', fontsize=12)
    animator.add_text(0.02, 0.90, f'Wall: {wall_temp - 273.15:.0f}°C', fontsize=12)
    animator.add_text(
        0.02, 0.85,
        f'ΔT: {(T_axial[-1] - T_axial[0]):.1f}°C',
        fontsize=12
    )
    
    # Save video
    print("\nGenerating video...")
    try:
        animator.save_video(
            'temperature_evolution.mp4',
            duration=10,
            fps=30,
            dpi=200
        )
        print("✓ Video saved: temperature_evolution.mp4")
    except Exception as e:
        print(f"⚠ Video generation failed: {e}")
        print("  (FFmpeg may not be installed)")
    
    # Save GIF
    print("\nGenerating GIF...")
    try:
        animator.save_gif('temperature_evolution.gif', duration=8, fps=15)
        print("✓ GIF saved: temperature_evolution.gif")
    except Exception as e:
        print(f"⚠ GIF generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("ANIMATION COMPLETE")
    print("=" * 60)
    print("\nKey insights:")
    print(f"  - Oil temperature increases by {(T_axial[-1] - T_axial[0]):.1f}°C")
    print(f"  - Heat transfer from steam tracing is effective")
    print(f"  - Particles show fluid trajectory and temperature")


if __name__ == "__main__":
    main()
