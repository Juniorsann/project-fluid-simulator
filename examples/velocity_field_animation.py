"""
Velocity Field Animation Example

This example demonstrates detailed analysis of velocity profiles across
different flow rates, showing the transition from laminar to turbulent flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from src.models.oil_properties import create_medium_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC
from src.core.solver import PipeFlowSolver, PoiseuilleFlow
from src.core.grid import Grid2DAxisymmetric


def main():
    """Run velocity field animation."""
    print("=" * 70)
    print("VELOCITY FIELD ANIMATION - Flow Regime Analysis")
    print("=" * 70)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Define pipe and fluid
    oil = create_medium_oil("medium_crude")
    pipe = Pipe(diameter=0.2, length=5.0, material="carbon_steel")
    
    temperature = 60 + 273.15  # 60°C
    
    print(f"\nPipe diameter: {pipe.diameter*1000:.0f} mm")
    print(f"Pipe length: {pipe.length:.1f} m")
    print(f"Oil: {oil.name}")
    print(f"Temperature: {temperature - 273.15:.1f}°C")
    
    # Simulate different flow rates
    flow_rates = np.linspace(0.01, 0.15, 40)  # 40 cases
    results_list = []
    
    print(f"\nSimulating {len(flow_rates)} different flow rates...")
    
    for i, Q in enumerate(flow_rates):
        # Calculate velocity from flow rate
        velocity = Q / pipe.cross_sectional_area()
        
        inlet = InletBC(velocity=velocity, temperature=temperature)
        outlet = OutletBC(pressure=101325.0)
        
        solver = PipeFlowSolver(pipe, oil, inlet, outlet)
        results = solver.solve_flow()
        
        # Store results with additional info
        results['flow_rate'] = Q
        results['velocity'] = velocity
        results_list.append(results)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(flow_rates)} - Re={results['reynolds_number']:.0f}")
    
    print(f"✅ Simulations complete!")
    
    # Create animation showing evolution of profiles
    print("\n" + "=" * 70)
    print("Creating velocity field evolution animation...")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Generate grid for 2D visualization
    grid_2d = Grid2DAxisymmetric(
        radius=pipe.radius(),
        length=min(pipe.length, 1.0),
        n_radial=40,
        n_axial=60,
        wall_refinement=False
    )
    
    def update(frame):
        """Update function for animation."""
        results = results_list[frame]
        
        # Clear all axes
        for ax in axes.flat:
            ax.clear()
        
        # Create solver for this flow rate
        velocity = results['velocity']
        
        # Plot 1: Radial velocity profile
        r = np.linspace(0, pipe.radius(), 100)
        
        if results['flow_regime'] == 'laminar':
            poiseuille = PoiseuilleFlow(pipe, oil, temperature)
            u = poiseuille.velocity_profile(r, velocity)
            
            axes[0, 0].plot(r * 1000, u, 'b-', linewidth=2, label='Laminar (Poiseuille)')
            axes[0, 0].fill_between(r * 1000, 0, u, alpha=0.3, color='blue')
        else:
            # Power law profile for turbulent flow
            n = 7
            u_max = (n + 1) * (n + 2) / (2 * n ** 2) * velocity
            u = u_max * (1 - r / pipe.radius()) ** (1 / n)
            
            axes[0, 0].plot(r * 1000, u, 'r-', linewidth=2, label='Turbulent (Power Law)')
            axes[0, 0].fill_between(r * 1000, 0, u, alpha=0.3, color='red')
        
        axes[0, 0].set_xlabel('Radial Position [mm]', fontsize=11)
        axes[0, 0].set_ylabel('Axial Velocity [m/s]', fontsize=11)
        axes[0, 0].set_title(f'Velocity Profile - Re={results["reynolds_number"]:.0f}',
                            fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_xlim([0, pipe.radius() * 1000])
        
        # Plot 2: 2D velocity field
        solver_temp = PipeFlowSolver(pipe, oil,
                                     InletBC(velocity=velocity, temperature=temperature),
                                     OutletBC(pressure=101325.0))
        u_r, u_z = solver_temp.solve_velocity_profile_2d(grid_2d)
        
        R, Z = np.meshgrid(grid_2d.r, grid_2d.z, indexing='ij')
        contour = axes[0, 1].contourf(Z * 1000, R * 1000, u_z,
                                      levels=20, cmap='viridis')
        axes[0, 1].set_xlabel('Axial Position [mm]', fontsize=11)
        axes[0, 1].set_ylabel('Radial Position [mm]', fontsize=11)
        axes[0, 1].set_title('2D Velocity Field', fontsize=12, fontweight='bold')
        axes[0, 1].set_aspect('equal')
        
        # Plot 3: Reynolds number vs Flow rate
        Re_values = [r['reynolds_number'] for r in results_list[:frame+1]]
        Q_values = [r['flow_rate'] for r in results_list[:frame+1]]
        
        axes[1, 0].plot(Q_values, Re_values, 'ko-', markersize=4, linewidth=1.5)
        axes[1, 0].axhline(2300, color='g', linestyle='--', linewidth=2, label='Laminar/Transition')
        axes[1, 0].axhline(4000, color='r', linestyle='--', linewidth=2, label='Transition/Turbulent')
        axes[1, 0].fill_between([0, 0.2], 0, 2300, alpha=0.1, color='blue')
        axes[1, 0].fill_between([0, 0.2], 2300, 4000, alpha=0.1, color='yellow')
        axes[1, 0].fill_between([0, 0.2], 4000, 10000, alpha=0.1, color='red')
        axes[1, 0].set_xlabel('Flow Rate [m³/s]', fontsize=11)
        axes[1, 0].set_ylabel('Reynolds Number', fontsize=11)
        axes[1, 0].set_title('Flow Regime Evolution', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([0, flow_rates.max()])
        axes[1, 0].set_ylim([0, max(Re_values) * 1.1 if Re_values else 5000])
        
        # Plot 4: Information text
        axes[1, 1].axis('off')
        info_text = f"""
SIMULATION PARAMETERS

Flow Rate:     {results['flow_rate']:.4f} m³/s
Mean Velocity: {results['velocity']:.3f} m/s

Reynolds:      {results['reynolds_number']:.0f}
Flow Regime:   {results['flow_regime']}

Pressure Drop: {results['pressure_drop']/1e5:.3f} bar
Δp/L:          {results['pressure_drop']/pipe.length:.1f} Pa/m

Temperature:   {temperature - 273.15:.1f}°C
Viscosity:     {oil.dynamic_viscosity(temperature)*1000:.1f} cP
Density:       {oil.density(temperature):.1f} kg/m³
        """
        
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=11,
                       family='monospace', verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f'Velocity Field Analysis - Frame {frame+1}/{len(results_list)}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
    
    # Create animation
    print("Rendering animation frames...")
    anim = FuncAnimation(fig, update, frames=len(results_list),
                        interval=200, repeat=True)
    
    # Try to save
    print("\nSaving animation...")
    saved = False
    try:
        writer = FFMpegWriter(fps=10, bitrate=5000)
        anim.save('output/velocity_field_evolution.mp4', writer=writer, dpi=120)
        print("✅ Saved: output/velocity_field_evolution.mp4")
        saved = True
    except FileNotFoundError:
        print("⚠️  ffmpeg not available, trying GIF format...")
    except Exception as e:
        print(f"⚠️  Could not save MP4: {e}")
    
    if not saved:
        try:
            writer = PillowWriter(fps=10)
            anim.save('output/velocity_field_evolution.gif', writer=writer, dpi=100)
            print("✅ Saved: output/velocity_field_evolution.gif")
            saved = True
        except Exception as e2:
            print(f"⚠️  Could not save GIF: {e2}")
            import traceback
            traceback.print_exc()
    
    if not saved:
        print("⚠️  Could not save animation in any format")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("ANIMATION COMPLETE")
    print("=" * 70)
    print("📁 Check the 'output/' directory")


if __name__ == "__main__":
    main()
