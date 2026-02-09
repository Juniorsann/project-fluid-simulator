"""
Temperature Evolution Example

This example demonstrates thermal evolution in a heated pipe,
showing how heating affects oil viscosity and flow behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from src.models.oil_properties import create_heavy_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC, WallBC
from src.core.solver import PipeFlowSolver, HeatTransferSolver
from src.core.grid import Grid2DAxisymmetric


def main():
    """Run temperature evolution simulation."""
    print("=" * 70)
    print("TEMPERATURE EVOLUTION - Heated Pipe Flow")
    print("=" * 70)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Heavy oil (high viscosity)
    oil = create_heavy_oil("heavy_crude")
    
    # Long pipe with heating
    pipe = Pipe(diameter=0.3, length=20.0, material="carbon_steel")
    
    # Inlet is cold, wall is heated
    T_inlet = 40 + 273.15   # 40°C (cold inlet)
    T_wall = 90 + 273.15    # 90°C (strong heating)
    flow_rate = 0.08        # m³/s
    
    print(f"\nPipe: {pipe.diameter*1000:.0f} mm diameter, {pipe.length:.0f} m length")
    print(f"Oil: {oil.name}")
    print(f"Inlet temperature: {T_inlet - 273.15:.1f}°C")
    print(f"Wall temperature: {T_wall - 273.15:.1f}°C")
    print(f"Flow rate: {flow_rate:.3f} m³/s")
    
    # Calculate velocity
    velocity = flow_rate / pipe.cross_sectional_area()
    
    # Setup boundary conditions
    inlet = InletBC(velocity=velocity, temperature=T_inlet)
    outlet = OutletBC(pressure=101325.0)
    wall = WallBC(temperature=T_wall)
    
    # Solve flow
    print("\nSolving flow field...")
    solver = PipeFlowSolver(pipe, oil, inlet, outlet, wall)
    results = solver.solve_flow()
    
    print(f"Reynolds number: {results['reynolds_number']:.0f}")
    print(f"Flow regime: {results['flow_regime']}")
    print(f"Pressure drop: {results['pressure_drop']/1e5:.3f} bar")
    
    # Calculate temperature profile along pipe
    print("\nCalculating temperature distribution...")
    heat_solver = HeatTransferSolver(pipe, oil, T_inlet, wall, velocity)
    
    z_positions = np.linspace(0, pipe.length, 200)
    T_profile = heat_solver.solve_temperature_profile_1d(z_positions)
    
    # Calculate viscosity profile (temperature dependent)
    mu_profile = np.array([oil.dynamic_viscosity(T) for T in T_profile])
    
    print(f"Temperature rise: {(T_profile[-1] - T_inlet):.1f} K")
    print(f"Viscosity change: {mu_profile[0]*1000:.1f} → {mu_profile[-1]*1000:.1f} cP")
    
    # Create 2D fields for visualization
    print("\nGenerating 2D fields...")
    grid_2d = Grid2DAxisymmetric(
        radius=pipe.radius(),
        length=min(pipe.length, 5.0),  # Show first 5m
        n_radial=40,
        n_axial=100,
        wall_refinement=True
    )
    
    # Create 2D temperature field (simplified - assumes radial variation)
    Z_mesh, R_mesh = np.meshgrid(grid_2d.z, grid_2d.r, indexing='ij')
    
    # Interpolate axial temperature
    T_axial = np.interp(grid_2d.z, z_positions, T_profile)
    
    # Add radial variation (hotter near wall)
    r_normalized = grid_2d.r / pipe.radius()
    # Simple radial profile: cooler at center, hotter at wall
    T_field_2d = np.zeros((len(grid_2d.z), len(grid_2d.r)))
    for i, z in enumerate(grid_2d.z):
        T_axial_val = np.interp(z, z_positions, T_profile)
        # Radial variation
        for j, r_norm in enumerate(r_normalized):
            T_field_2d[i, j] = T_inlet + (T_axial_val - T_inlet) * (0.5 + 0.5 * r_norm**2)
    
    # Get velocity field
    u_r, u_z = solver.solve_velocity_profile_2d(grid_2d)
    
    # Create animation
    print("\n" + "=" * 70)
    print("Creating thermal evolution visualization...")
    print("=" * 70)
    
    # Create multiple time steps by scaling the development
    n_time_steps = 50
    time_values = np.linspace(0, 100, n_time_steps)  # 0 to 100 seconds
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    
    def update(frame):
        """Update animation frame."""
        t = time_values[frame]
        
        # Progress of heating (0 to 1)
        progress = frame / (n_time_steps - 1)
        
        # Scale temperature field by progress
        T_current = T_inlet + (T_field_2d - T_inlet) * progress
        T_current_1d = T_inlet + (T_profile - T_inlet) * progress
        mu_current = np.array([oil.dynamic_viscosity(T) for T in T_current_1d])
        
        # Clear axes
        for ax in axes:
            ax.clear()
        
        # Plot 1: Temperature along pipe
        axes[0].plot(z_positions, T_current_1d - 273.15, 'r-', linewidth=2.5, label='Bulk Temperature')
        axes[0].axhline(T_wall - 273.15, color='orange', linestyle='--',
                       linewidth=2, label=f'Wall Temp ({T_wall-273.15:.0f}°C)')
        axes[0].fill_between(z_positions, 0, T_current_1d - 273.15, alpha=0.3, color='red')
        
        # Add pour point if available
        if hasattr(oil, 'pour_point'):
            axes[0].axhline(oil.pour_point, color='b', linestyle='--',
                          linewidth=1.5, label=f'Pour Point ({oil.pour_point:.0f}°C)')
        
        axes[0].set_xlabel('Axial Position [m]', fontsize=11)
        axes[0].set_ylabel('Temperature [°C]', fontsize=11)
        axes[0].set_title(f'Temperature Profile - t={t:.1f}s', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, pipe.length])
        axes[0].set_ylim([0, 100])
        
        # Plot 2: Viscosity along pipe
        axes[1].semilogy(z_positions, mu_current * 1000, 'g-', linewidth=2.5)
        axes[1].set_xlabel('Axial Position [m]', fontsize=11)
        axes[1].set_ylabel('Dynamic Viscosity [cP]', fontsize=11)
        axes[1].set_title('Viscosity Profile', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].set_xlim([0, pipe.length])
        
        # Plot 3: 2D temperature field with streamlines
        R, Z = np.meshgrid(grid_2d.r, grid_2d.z, indexing='ij')
        
        # Transpose for correct orientation
        contour = axes[2].contourf(Z.T * 1000, R.T * 1000, T_current.T - 273.15,
                                   levels=30, cmap='hot')
        
        # Add streamlines
        # Downsample for streamplot
        skip = 3
        axes[2].streamplot(Z.T[::skip, ::skip] * 1000, R.T[::skip, ::skip] * 1000,
                          u_z.T[::skip, ::skip], u_r.T[::skip, ::skip],
                          color='white', density=1.2, linewidth=0.8, arrowsize=0.8)
        
        axes[2].set_xlabel('Axial Position [mm]', fontsize=11)
        axes[2].set_ylabel('Radial Position [mm]', fontsize=11)
        axes[2].set_title('2D Temperature Field with Flow Streamlines', fontsize=12, fontweight='bold')
        axes[2].set_aspect('equal')
        
        # Add colorbar only on first frame
        if frame == 0:
            cbar = plt.colorbar(contour, ax=axes[2])
            cbar.set_label('Temperature [°C]', fontsize=11)
        
        plt.tight_layout()
    
    # Create animation
    print("Rendering animation frames...")
    anim = FuncAnimation(fig, update, frames=n_time_steps,
                        interval=100, repeat=True)
    
    # Save animation
    print("\nSaving animation...")
    try:
        writer = FFMpegWriter(fps=15, bitrate=8000)
        anim.save('output/thermal_evolution.mp4', writer=writer, dpi=120)
        print("✅ Saved: output/thermal_evolution.mp4")
    except Exception as e:
        print(f"⚠️  Could not save MP4: {e}")
        try:
            writer = PillowWriter(fps=10)
            anim.save('output/thermal_evolution.gif', writer=writer, dpi=100)
            print("✅ Saved: output/thermal_evolution.gif")
        except Exception as e2:
            print(f"⚠️  Could not save animation: {e2}")
    
    plt.close()
    
    # Also create static summary plot
    print("\nCreating static summary plot...")
    fig_summary, ax_summary = plt.subplots(1, 1, figsize=(12, 4))
    
    R, Z = np.meshgrid(grid_2d.r, grid_2d.z, indexing='ij')
    contour = ax_summary.contourf(Z.T * 1000, R.T * 1000, T_field_2d.T - 273.15,
                                 levels=30, cmap='hot')
    
    skip = 3
    ax_summary.streamplot(Z.T[::skip, ::skip] * 1000, R.T[::skip, ::skip] * 1000,
                         u_z.T[::skip, ::skip], u_r.T[::skip, ::skip],
                         color='white', density=1.5, linewidth=1, arrowsize=1)
    
    ax_summary.set_xlabel('Axial Position [mm]', fontsize=12)
    ax_summary.set_ylabel('Radial Position [mm]', fontsize=12)
    ax_summary.set_title('Temperature Field - Steady State', fontsize=14, fontweight='bold')
    ax_summary.set_aspect('equal')
    
    cbar = plt.colorbar(contour, ax=ax_summary)
    cbar.set_label('Temperature [°C]', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('output/thermal_field_static.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: output/thermal_field_static.png")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("THERMAL ANALYSIS COMPLETE")
    print("=" * 70)
    print("📁 Check the 'output/' directory")
    print("\nGenerated files:")
    print("  - thermal_evolution.mp4/gif (temperature animation)")
    print("  - thermal_field_static.png (steady-state field)")


if __name__ == "__main__":
    main()
