"""
Side-by-side comparison of different scenarios.

This example compares flow with and without heating to show
the effect of temperature on flow characteristics.
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
from src.visualization.animator import ComparisonAnimator


def solve_case(oil, pipe, inlet_temp, wall_temp=None):
    """
    Solve a single flow case.
    
    Args:
        oil: Oil fluid object
        pipe: Pipe geometry
        inlet_temp: Inlet temperature [K]
        wall_temp: Wall temperature [K] (None for adiabatic)
        
    Returns:
        Dictionary with results
    """
    flow_rate = 0.05  # m³/s
    
    inlet = InletBC(
        mass_flow_rate=flow_rate * oil.density(inlet_temp),
        temperature=inlet_temp
    )
    outlet = OutletBC(pressure=101325.0)
    wall = WallBC(temperature=wall_temp) if wall_temp else None
    
    # Solve flow
    solver = PipeFlowSolver(pipe, oil, inlet, outlet, wall)
    flow_results = solver.solve_flow()
    
    # Create grid
    grid = Grid2DAxisymmetric(
        radius=pipe.radius(),
        length=pipe.length,
        n_radial=30,
        n_axial=80,
        wall_refinement=True
    )
    
    # Get velocity field
    u_r, u_z = solver.solve_velocity_profile_2d(grid)
    
    # Get temperature field
    if wall_temp:
        heat_solver = HeatTransferSolver(
            pipe, oil, inlet_temp, wall, flow_results['mean_velocity']
        )
        T_axial = heat_solver.solve_temperature_profile_1d(grid.z)
        T_2d = np.tile(T_axial, (grid.n_radial, 1))
    else:
        # Adiabatic - constant temperature
        T_2d = np.full((grid.n_radial, grid.n_axial), inlet_temp)
    
    return {
        'velocity_field': (u_r, u_z),
        'temperature_field': T_2d,
        'grid': (grid.r, grid.z),
        'pipe_radius': pipe.radius(),
        'pipe_length': pipe.length,
        'flow_results': flow_results
    }


def main():
    """
    Create comparison animation of heated vs non-heated scenarios.
    """
    print("=" * 60)
    print("SCENARIO COMPARISON: HEATING EFFECT")
    print("=" * 60)
    
    # Setup common parameters
    print("\nSetting up scenarios...")
    oil = OilFluid(
        name="heavy_crude",
        api_gravity=14.0,
        reference_temp=288.15,
        reference_viscosity=0.150  # 150 cP
    )
    
    pipe = Pipe(
        diameter=0.3,
        length=6.0,
        roughness=0.000045,
        material="carbon_steel"
    )
    
    print(f"Oil: {oil.name} (API {oil.api_gravity}°)")
    print(f"Pipe: {pipe.diameter*1000} mm × {pipe.length} m")
    
    # Scenario 1: No heating (ambient temperature)
    print("\n--- Scenario 1: No Heating ---")
    inlet_temp_cold = 35 + 273.15  # 35°C
    results_cold = solve_case(oil, pipe, inlet_temp_cold, wall_temp=None)
    
    print(f"Inlet temp: {inlet_temp_cold - 273.15}°C")
    print(f"Reynolds: {results_cold['flow_results']['reynolds_number']:.0f}")
    print(f"Pressure drop: {results_cold['flow_results']['pressure_drop']/1e5:.3f} bar")
    
    # Scenario 2: With heating
    print("\n--- Scenario 2: Steam Tracing ---")
    inlet_temp_hot = 35 + 273.15   # Same inlet
    wall_temp = 80 + 273.15        # 80°C wall
    results_heated = solve_case(oil, pipe, inlet_temp_hot, wall_temp=wall_temp)
    
    print(f"Inlet temp: {inlet_temp_hot - 273.15}°C")
    print(f"Wall temp: {wall_temp - 273.15}°C")
    print(f"Reynolds: {results_heated['flow_results']['reynolds_number']:.0f}")
    print(f"Pressure drop: {results_heated['flow_results']['pressure_drop']/1e5:.3f} bar")
    
    # Calculate benefit
    dp_reduction = (
        (results_cold['flow_results']['pressure_drop'] -
         results_heated['flow_results']['pressure_drop']) /
        results_cold['flow_results']['pressure_drop'] * 100
    )
    print(f"\nPressure drop reduction: {dp_reduction:.1f}%")
    
    # Create comparison animator
    print("\nCreating comparison animation...")
    comparison = ComparisonAnimator(
        [results_cold, results_heated],
        labels=['No Heating', 'Steam Tracing'],
        figure_size=(16, 6),
        fps=30
    )
    
    # Add visualization layers
    comparison.add_particle_tracers(n_particles=150)
    comparison.add_temperature_map(cmap='hot')
    
    # Save video
    print("\nGenerating comparison video...")
    try:
        comparison.save_video('heating_comparison.mp4', duration=12, fps=30)
        print("✓ Comparison video saved: heating_comparison.mp4")
    except Exception as e:
        print(f"⚠ Video generation failed: {e}")
        print("  (FFmpeg may not be installed)")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print("\nKey findings:")
    print(f"  - Heating reduces pressure drop by {dp_reduction:.1f}%")
    print(f"  - Lower viscosity improves flow efficiency")
    print(f"  - Steam tracing is effective for heavy oils")
    
    # Additional analysis
    print("\nEnergy considerations:")
    flow_rate = 0.05  # m³/s
    power_saved = (
        (results_cold['flow_results']['pressure_drop'] -
         results_heated['flow_results']['pressure_drop']) * flow_rate
    )
    print(f"  - Pumping power saved: {power_saved:.1f} W")
    print(f"  - Must be weighed against heating cost")


if __name__ == "__main__":
    main()
