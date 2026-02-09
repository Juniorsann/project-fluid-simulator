"""
Animated velocity field showing development of flow profile.

This example focuses on velocity field visualization with arrows and streamlines.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.models.oil_properties import OilFluid
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC
from src.core.solver import PipeFlowSolver
from src.core.grid import Grid2DAxisymmetric
from src.visualization.animator import FlowAnimator


def main():
    """
    Create animated velocity field visualization.
    
    Shows development of flow profile for heavy oil.
    """
    print("=" * 60)
    print("VELOCITY FIELD ANIMATION")
    print("=" * 60)
    
    # Heavy oil simulation
    print("\nSetting up heavy oil simulation...")
    oil = OilFluid(
        name="heavy_crude",
        api_gravity=12.0,  # Heavy oil
        reference_temp=288.15,
        reference_viscosity=0.200  # 200 cP - very viscous
    )
    
    pipe = Pipe(
        diameter=0.3,  # 300 mm
        length=5.0,    # 5 m
        roughness=0.000045,
        material="carbon_steel"
    )
    
    print(f"Oil: API gravity = {oil.api_gravity}°")
    print(f"Pipe: {pipe.diameter*1000} mm × {pipe.length} m")
    
    # Boundary conditions
    temperature = 40 + 273.15  # 40°C
    flow_rate = 0.05  # m³/s
    
    inlet = InletBC(
        mass_flow_rate=flow_rate * oil.density(temperature),
        temperature=temperature
    )
    outlet = OutletBC(pressure=101325.0)
    
    print(f"Flow rate: {flow_rate} m³/s")
    print(f"Temperature: {temperature - 273.15}°C")
    
    # Solve
    print("\nSolving flow field...")
    solver = PipeFlowSolver(pipe, oil, inlet, outlet)
    results = solver.solve_flow()
    
    print(f"Reynolds number: {results['reynolds_number']:.0f}")
    print(f"Flow regime: {results['flow_regime']}")
    print(f"Mean velocity: {results['mean_velocity']:.2f} m/s")
    print(f"Pressure drop: {results['pressure_drop']/1e5:.3f} bar")
    
    # Create 2D grid
    print("\nGenerating velocity field...")
    grid = Grid2DAxisymmetric(
        radius=pipe.radius(),
        length=pipe.length,
        n_radial=30,
        n_axial=80,
        wall_refinement=True
    )
    
    # Get velocity profile
    u_r, u_z = solver.solve_velocity_profile_2d(grid)
    
    # Prepare results for animator
    anim_results = {
        'velocity_field': (u_r, u_z),
        'grid': (grid.r, grid.z),
        'pipe_radius': pipe.radius(),
        'pipe_length': pipe.length
    }
    
    # Create animator focused on velocity
    print("\nCreating animation...")
    animator = FlowAnimator(anim_results, figure_size=(14, 6))
    
    # Add velocity visualization layers
    animator.add_velocity_field(style='arrows', density=25, scale=1.5)
    animator.add_streamlines(n_lines=40, integration_steps=200)
    
    # Add annotations
    animator.add_colorbar(label='Velocity (m/s)')
    animator.add_title('Heavy Oil Flow - Velocity Field Evolution')
    animator.add_text(
        0.02, 0.95,
        f'Re = {results["reynolds_number"]:.0f}',
        fontsize=12
    )
    animator.add_text(
        0.02, 0.90,
        f'Regime: {results["flow_regime"]}',
        fontsize=12
    )
    
    # Save animation
    print("\nGenerating video...")
    try:
        animator.save_video('velocity_field.mp4', duration=12, fps=30, dpi=150)
        print("✓ Video saved: velocity_field.mp4")
    except Exception as e:
        print(f"⚠ Video generation failed: {e}")
        print("  (FFmpeg may not be installed)")
    
    # Save frames
    print("\nSaving sample frames...")
    try:
        animator.save_frames(
            directory='velocity_frames/',
            prefix='vel_',
            format='png',
            duration=2,
            fps=10
        )
        print("✓ Frames saved to velocity_frames/")
    except Exception as e:
        print(f"⚠ Frame export failed: {e}")
    
    print("\n" + "=" * 60)
    print("ANIMATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
