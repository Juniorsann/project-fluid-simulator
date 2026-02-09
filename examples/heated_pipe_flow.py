"""
Heated Pipe Flow Example

This example demonstrates flow with heating for viscosity reduction.
Shows the effect of temperature on viscosity and pressure drop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.models.oil_properties import create_heavy_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC, WallBC
from src.core.solver import PipeFlowSolver, HeatTransferSolver
from src.core.grid import Grid1D
from src.visualization.plotter import plot_temperature_profile, plot_viscosity_temperature


def main():
    """Run heated pipe flow simulation."""
    print("=" * 60)
    print("HEATED PIPE FLOW SIMULATION")
    print("=" * 60)

    # Define pipe geometry with insulation
    pipe = Pipe(
        diameter=0.15,  # 150 mm diameter
        length=50.0,  # 50 m length
        material="carbon_steel",
        insulation_thickness=0.05,  # 50 mm insulation
        insulation_conductivity=0.04,  # Mineral wool
        ambient_temperature=283.15,  # 10°C ambient
        name="heated_production_line"
    )
    print(f"\nPipe: {pipe}")
    print(f"Insulation thickness: {pipe.insulation_thickness*1000:.0f} mm")

    # Define fluid (heavy crude oil)
    oil = create_heavy_oil("heavy_crude")
    T_inlet = 293.15  # 20°C inlet
    T_wall = 350.15  # 77°C wall temperature (steam heating)
    
    print(f"\nOil: {oil}")
    print(f"Inlet temperature: {T_inlet - 273.15:.1f}°C")
    print(f"Wall temperature: {T_wall - 273.15:.1f}°C")

    # Compare properties at different temperatures
    print("\n" + "-" * 60)
    print("TEMPERATURE EFFECT ON PROPERTIES")
    print("-" * 60)
    
    temps = [T_inlet, 313.15, 333.15, T_wall]  # 20, 40, 60, 77°C
    for T in temps:
        rho = oil.density(T)
        mu = oil.dynamic_viscosity(T)
        print(f"T = {T-273.15:5.1f}°C: ρ = {rho:6.1f} kg/m³, μ = {mu*1000:8.1f} cP")

    # Define boundary conditions
    mean_velocity = 0.5  # m/s
    inlet = InletBC(velocity=mean_velocity, temperature=T_inlet)
    outlet = OutletBC(pressure=101325.0)
    wall = WallBC(temperature=T_wall)

    # Solve flow at inlet conditions
    print("\n" + "=" * 60)
    print("FLOW ANALYSIS AT INLET CONDITIONS")
    print("=" * 60)
    
    solver_inlet = PipeFlowSolver(pipe, oil, inlet, outlet)
    results_inlet = solver_inlet.solve_flow()
    
    print(f"Reynolds number: {results_inlet['reynolds_number']:.0f}")
    print(f"Flow regime: {results_inlet['flow_regime']}")
    print(f"Pressure drop (cold): {results_inlet['pressure_drop']/1e5:.4f} bar")

    # Solve heat transfer
    print("\n" + "=" * 60)
    print("HEAT TRANSFER ANALYSIS")
    print("=" * 60)
    
    heat_solver = HeatTransferSolver(
        pipe, oil, T_inlet, wall, mean_velocity
    )
    
    # Calculate temperature profile along pipe
    grid_1d = Grid1D(pipe.length, n_cells=200)
    z = grid_1d.x
    T_profile = heat_solver.solve_temperature_profile_1d(z)
    
    print(f"Outlet temperature: {T_profile[-1] - 273.15:.1f}°C")
    print(f"Temperature increase: {T_profile[-1] - T_inlet:.1f} K")
    
    # Plot temperature profile
    plot_temperature_profile(
        z, T_profile, pipe.length,
        title="Temperature Profile Along Heated Pipe",
        save_path="heated_pipe_temperature.png"
    )
    print("\nTemperature profile saved to 'heated_pipe_temperature.png'")

    # Calculate viscosity reduction
    mu_inlet = oil.dynamic_viscosity(T_inlet)
    mu_outlet = oil.dynamic_viscosity(T_profile[-1])
    viscosity_reduction = (1 - mu_outlet/mu_inlet) * 100
    
    print(f"\nViscosity at inlet: {mu_inlet*1000:.1f} cP")
    print(f"Viscosity at outlet: {mu_outlet*1000:.1f} cP")
    print(f"Viscosity reduction: {viscosity_reduction:.1f}%")

    # Estimate pressure drop with averaged temperature
    T_avg = np.mean(T_profile)
    inlet_avg = InletBC(velocity=mean_velocity, temperature=T_avg)
    solver_avg = PipeFlowSolver(pipe, oil, inlet_avg, outlet)
    results_avg = solver_avg.solve_flow()
    
    print(f"\nPressure drop (with heating): {results_avg['pressure_drop']/1e5:.4f} bar")
    pressure_reduction = (1 - results_avg['pressure_drop']/results_inlet['pressure_drop']) * 100
    print(f"Pressure drop reduction: {pressure_reduction:.1f}%")

    # Calculate energy savings in pumping
    Q = mean_velocity * pipe.cross_sectional_area()
    power_cold = results_inlet['pressure_drop'] * Q
    power_hot = results_avg['pressure_drop'] * Q
    power_savings = power_cold - power_hot
    
    print(f"\nPumping power (cold): {power_cold:.2f} W")
    print(f"Pumping power (heated): {power_hot:.2f} W")
    print(f"Power savings: {power_savings:.2f} W ({power_savings/power_cold*100:.1f}%)")

    # Plot viscosity vs temperature
    temp_range = np.linspace(273.15, 373.15, 50)  # 0-100°C
    visc_range = np.array([oil.dynamic_viscosity(T) for T in temp_range])
    
    plot_viscosity_temperature(
        temp_range, visc_range,
        oil_names=[oil.name],
        title="Viscosity-Temperature Relationship",
        save_path="heated_pipe_viscosity.png"
    )
    print("Viscosity curve saved to 'heated_pipe_viscosity.png'")

    # Economic analysis
    print("\n" + "=" * 60)
    print("ECONOMIC CONSIDERATIONS")
    print("=" * 60)
    
    hours_per_year = 8000  # Operating hours
    electricity_cost = 0.12  # $/kWh
    
    annual_savings = power_savings / 1000 * hours_per_year * electricity_cost
    print(f"Annual pumping cost savings: ${annual_savings:.2f}")
    print(f"(Assumes {hours_per_year} operating hours/year at ${electricity_cost}/kWh)")
    print("\nNote: Heating costs must be considered for complete economic analysis")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
