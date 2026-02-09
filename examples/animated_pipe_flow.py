"""
Animated Pipe Flow Example

This example demonstrates animated visualization of oil flow through a pipe
with particle tracers, velocity fields, and temperature evolution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.models.oil_properties import create_medium_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC, WallBC
from src.core.solver import PipeFlowSolver
from src.core.grid import Grid2DAxisymmetric
from src.visualization.animator import FlowAnimator
from src.visualization.particle_tracer import ParticleTracer
from src.visualization.streamlines import StreamlineGenerator


def main():
    """Run animated pipe flow simulation."""
    print("=" * 70)
    print("ANIMATED PIPE FLOW SIMULATION")
    print("=" * 70)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Define pipe geometry
    pipe = Pipe(
        diameter=0.3,  # 300 mm diameter
        length=10.0,   # 10 m length
        material="carbon_steel",
        name="heated_pipe"
    )
    print(f"\nPipe: {pipe}")
    
    # Define fluid (medium crude oil)
    oil = create_medium_oil("medium_crude")
    temperature_inlet = 70 + 273.15  # 70°C
    temperature_wall = 80 + 273.15   # 80°C (heating)
    
    print(f"\nOil: {oil}")
    print(f"Inlet temperature: {temperature_inlet - 273.15:.1f}°C")
    print(f"Wall temperature: {temperature_wall - 273.15:.1f}°C")
    
    # Define boundary conditions
    mean_velocity = 1.5  # m/s
    inlet = InletBC(velocity=mean_velocity, temperature=temperature_inlet)
    outlet = OutletBC(pressure=101325.0)  # Atmospheric pressure
    wall = WallBC(temperature=temperature_wall)
    
    # Solve flow field
    print("\nSolving flow field...")
    solver = PipeFlowSolver(pipe, oil, inlet, outlet, wall)
    results = solver.solve_flow()
    
    print(f"\nReynolds number: {results['reynolds_number']:.0f}")
    print(f"Flow regime: {results['flow_regime']}")
    print(f"Pressure drop: {results['pressure_drop']/1e5:.4f} bar")
    
    # Create 2D grid for visualization
    print("\nGenerating 2D velocity field...")
    grid_2d = Grid2DAxisymmetric(
        radius=pipe.radius(),
        length=min(pipe.length, 2.0),  # Show first 2m for visualization
        n_radial=50,
        n_axial=100,
        wall_refinement=True
    )
    
    # Get 2D velocity profile
    u_r, u_z = solver.solve_velocity_profile_2d(grid_2d)
    
    # ANIMATION 1: Particle Tracers
    print("\n" + "=" * 70)
    print("Creating Animation 1: Particle Tracers")
    print("=" * 70)
    
    animator = FlowAnimator(figure_size=(14, 4), fps=30)
    
    # Create simple temperature field for coloring (linear approximation)
    # Create 2D field matching grid shape
    Z_mesh, R_mesh = np.meshgrid(grid_2d.z, grid_2d.r, indexing='ij')
    
    # Temperature varies with axial position
    z_normalized = grid_2d.z / grid_2d.z.max()
    temp_field = np.zeros((len(grid_2d.r), len(grid_2d.z)))
    for i in range(len(grid_2d.z)):
        temp_field[:, i] = temperature_inlet + (temperature_wall - temperature_inlet) * z_normalized[i]
    
    anim1 = animator.animate_particle_tracers(
        velocity_field=(u_z, u_r),
        grid_coordinates=(grid_2d.z, grid_2d.r),
        n_particles=800,
        duration=8,
        fps=30,
        colorby=temp_field,
        cmap='hot',
        particle_size=3,
        show_velocity_field=True,
        title="Oil Flow - Particle Tracers"
    )
    
    try:
        animator.save('output/oil_flow_particles.mp4', dpi=150)
        print("✅ Saved: output/oil_flow_particles.mp4")
    except Exception as e:
        print(f"⚠️  Could not save MP4 (ffmpeg may not be available): {e}")
        try:
            animator.save('output/oil_flow_particles.gif', dpi=100)
            print("✅ Saved: output/oil_flow_particles.gif")
        except Exception as e2:
            print(f"⚠️  Could not save animation: {e2}")
    
    plt.close()
    
    # ANIMATION 2: Velocity Field
    print("\n" + "=" * 70)
    print("Creating Animation 2: Velocity Field")
    print("=" * 70)
    
    animator2 = FlowAnimator(figure_size=(14, 4), fps=10)
    
    anim2 = animator2.animate_velocity_field(
        velocity_field=(u_z, u_r),
        grid_coordinates=(grid_2d.z, grid_2d.r),
        style='contour',
        show_magnitude=True,
        title="Velocity Field - Magnitude"
    )
    
    try:
        animator2.save('output/velocity_field.gif', dpi=120)
        print("✅ Saved: output/velocity_field.gif")
    except Exception as e:
        print(f"⚠️  Could not save animation: {e}")
    
    plt.close()
    
    # STATIC VISUALIZATION: Streamlines
    print("\n" + "=" * 70)
    print("Creating Static Visualization: Streamlines")
    print("=" * 70)
    
    streamlines = StreamlineGenerator(
        velocity_field=(u_z, u_r),
        grid_coordinates=(grid_2d.z, grid_2d.r)
    )
    
    streamlines.generate_streamlines(
        seed_points='auto',
        n_streamlines=15,
        max_length=1000,
        step_size=0.005
    )
    
    fig = streamlines.plot_streamlines(
        color_by='velocity_magnitude',
        linewidth='variable',
        cmap='coolwarm',
        background=temp_field,
        background_cmap='hot',
        background_label='Temperature [K]',
        title='Streamlines with Temperature Background',
        save_path='output/streamlines.png'
    )
    print("✅ Saved: output/streamlines.png")
    
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\n🎉 All visualizations created successfully!")
    print("📁 Check the 'output/' directory for results")
    print("\nGenerated files:")
    print("  - oil_flow_particles.mp4/gif (particle animation)")
    print("  - velocity_field.gif (velocity field)")
    print("  - streamlines.png (static streamlines)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
