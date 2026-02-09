"""
Pressure Drop Analysis Example

This example analyzes pressure drop for different pipe diameters,
performing CAPEX vs OPEX trade-off analysis for pipeline design.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.models.oil_properties import create_heavy_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC
from src.core.solver import PipeFlowSolver
from src.models.turbulence import friction_factor


def main():
    """Run pressure drop analysis."""
    print("=" * 60)
    print("PRESSURE DROP ANALYSIS & PIPELINE OPTIMIZATION")
    print("=" * 60)

    # Define fluid
    oil = create_heavy_oil("heavy_crude")
    temperature = 313.15  # 40°C
    
    print(f"\nOil: {oil}")
    print(f"Temperature: {temperature - 273.15:.1f}°C")
    print(f"Density: {oil.density(temperature):.1f} kg/m³")
    print(f"Viscosity: {oil.dynamic_viscosity(temperature)*1000:.1f} cP")

    # Pipeline parameters
    length = 10000.0  # 10 km
    mass_flow = 50.0  # kg/s (180 tonnes/hour)
    
    print(f"\nPipeline length: {length/1000:.1f} km")
    print(f"Mass flow rate: {mass_flow:.1f} kg/s ({mass_flow*3.6:.0f} tonnes/h)")

    # Analyze different pipe diameters
    diameters = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])  # meters
    
    print("\n" + "=" * 60)
    print("HYDRAULIC ANALYSIS FOR DIFFERENT DIAMETERS")
    print("=" * 60)
    print(f"\n{'D (mm)':<10} {'u (m/s)':<10} {'Re':<12} {'Regime':<12} {'Δp (bar)':<12} {'Power (kW)':<12}")
    print("-" * 80)

    results = {
        'diameter': diameters,
        'velocity': [],
        'reynolds': [],
        'regime': [],
        'pressure_drop': [],
        'power': []
    }

    for D in diameters:
        # Create pipe
        pipe = Pipe(diameter=D, length=length, material="carbon_steel")
        
        # Calculate velocity from mass flow
        A = pipe.cross_sectional_area()
        rho = oil.density(temperature)
        velocity = mass_flow / (rho * A)
        
        # Setup and solve
        inlet = InletBC(velocity=velocity, temperature=temperature)
        outlet = OutletBC(pressure=101325.0)
        solver = PipeFlowSolver(pipe, oil, inlet, outlet)
        res = solver.solve_flow()
        
        # Calculate power
        Q = velocity * A
        power = res['pressure_drop'] * Q / 1000  # kW
        
        # Store results
        results['velocity'].append(velocity)
        results['reynolds'].append(res['reynolds_number'])
        results['regime'].append(res['flow_regime'])
        results['pressure_drop'].append(res['pressure_drop'])
        results['power'].append(power)
        
        print(f"{D*1000:<10.0f} {velocity:<10.2f} {res['reynolds_number']:<12.0f} "
              f"{res['flow_regime']:<12} {res['pressure_drop']/1e5:<12.2f} {power:<12.2f}")

    # Convert lists to arrays
    for key in ['velocity', 'reynolds', 'pressure_drop', 'power']:
        results[key] = np.array(results[key])

    # Economic analysis
    print("\n" + "=" * 60)
    print("ECONOMIC ANALYSIS (CAPEX vs OPEX)")
    print("=" * 60)

    # Cost parameters (typical values)
    pipe_cost_per_kg = 2.0  # $/kg
    steel_density = 7850  # kg/m³
    wall_thickness_ratio = 0.05  # wall thickness / diameter
    installation_multiplier = 2.5  # Total installation cost multiplier
    
    electricity_cost = 0.10  # $/kWh
    operating_hours = 8000  # hours/year
    project_lifetime = 20  # years
    discount_rate = 0.08  # 8% annual discount
    
    print(f"\nEconomic Parameters:")
    print(f"  Electricity cost: ${electricity_cost}/kWh")
    print(f"  Operating hours: {operating_hours} h/year")
    print(f"  Project lifetime: {project_lifetime} years")
    print(f"  Discount rate: {discount_rate*100:.0f}%")

    # Calculate costs
    capex = []
    opex_annual = []
    npv_opex = []
    total_cost = []

    print(f"\n{'D (mm)':<10} {'CAPEX ($M)':<15} {'OPEX ($/yr)':<15} {'NPV Total ($M)':<15}")
    print("-" * 60)

    for i, D in enumerate(diameters):
        # CAPEX: Pipe material + installation
        t_wall = D * wall_thickness_ratio
        volume_steel = np.pi * ((D/2 + t_wall)**2 - (D/2)**2) * length
        mass_steel = volume_steel * steel_density
        pipe_material_cost = mass_steel * pipe_cost_per_kg
        total_capex = pipe_material_cost * installation_multiplier
        
        # OPEX: Pumping energy cost
        annual_opex = results['power'][i] * operating_hours * electricity_cost
        
        # NPV of OPEX over project lifetime
        npv_opex_value = sum([
            annual_opex / (1 + discount_rate)**year 
            for year in range(1, project_lifetime + 1)
        ])
        
        # Total NPV
        total_npv = total_capex + npv_opex_value
        
        capex.append(total_capex)
        opex_annual.append(annual_opex)
        npv_opex.append(npv_opex_value)
        total_cost.append(total_npv)
        
        print(f"{D*1000:<10.0f} {total_capex/1e6:<15.2f} {annual_opex:<15.0f} {total_npv/1e6:<15.2f}")

    # Find optimal diameter
    optimal_idx = np.argmin(total_cost)
    optimal_diameter = diameters[optimal_idx]
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL DIAMETER: {optimal_diameter*1000:.0f} mm")
    print(f"Total NPV Cost: ${total_cost[optimal_idx]/1e6:.2f}M")
    print(f"{'='*60}")

    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Pressure Drop vs Diameter
    ax1.plot(diameters*1000, results['pressure_drop']/1e5, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Pipe Diameter [mm]', fontsize=11)
    ax1.set_ylabel('Pressure Drop [bar]', fontsize=11)
    ax1.set_title('Pressure Drop vs Diameter', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Velocity and Reynolds Number
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(diameters*1000, results['velocity'], 'g-o', linewidth=2, 
                     markersize=8, label='Velocity')
    line2 = ax2_twin.plot(diameters*1000, results['reynolds'], 'r-s', linewidth=2, 
                          markersize=6, label='Reynolds Number')
    ax2_twin.axhline(y=2300, color='orange', linestyle='--', alpha=0.5, label='Laminar limit')
    
    ax2.set_xlabel('Pipe Diameter [mm]', fontsize=11)
    ax2.set_ylabel('Velocity [m/s]', fontsize=11, color='g')
    ax2_twin.set_ylabel('Reynolds Number', fontsize=11, color='r')
    ax2.set_title('Flow Characteristics vs Diameter', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=9)

    # Plot 3: Economic Comparison
    x_pos = np.arange(len(diameters))
    width = 0.35
    
    ax3.bar(x_pos - width/2, np.array(capex)/1e6, width, label='CAPEX', alpha=0.8)
    ax3.bar(x_pos + width/2, np.array(npv_opex)/1e6, width, label='NPV OPEX', alpha=0.8)
    
    ax3.axvline(x=optimal_idx, color='r', linestyle='--', linewidth=2, 
                alpha=0.7, label='Optimal')
    
    ax3.set_xlabel('Pipe Diameter [mm]', fontsize=11)
    ax3.set_ylabel('Cost [Million $]', fontsize=11)
    ax3.set_title('CAPEX vs OPEX Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{int(d*1000)}' for d in diameters])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Total NPV Cost
    ax4.plot(diameters*1000, np.array(total_cost)/1e6, 'b-o', linewidth=2, markersize=8)
    ax4.plot(optimal_diameter*1000, total_cost[optimal_idx]/1e6, 'r*', 
             markersize=20, label=f'Optimal: {optimal_diameter*1000:.0f} mm')
    ax4.set_xlabel('Pipe Diameter [mm]', fontsize=11)
    ax4.set_ylabel('Total NPV Cost [Million $]', fontsize=11)
    ax4.set_title('Total Cost Optimization', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('pressure_drop_optimization.png', dpi=300, bbox_inches='tight')
    print("\nOptimization plot saved to 'pressure_drop_optimization.png'")
    plt.close()

    # Sensitivity analysis on key parameter
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Electricity Cost")
    print("=" * 60)

    electricity_costs = [0.05, 0.10, 0.15, 0.20]  # $/kWh
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for elec_cost in electricity_costs:
        total_costs = []
        for i in range(len(diameters)):
            annual_opex = results['power'][i] * operating_hours * elec_cost
            npv_opex_value = sum([
                annual_opex / (1 + discount_rate)**year 
                for year in range(1, project_lifetime + 1)
            ])
            total_npv = capex[i] + npv_opex_value
            total_costs.append(total_npv)
        
        optimal_idx_sens = np.argmin(total_costs)
        ax.plot(diameters*1000, np.array(total_costs)/1e6, '-o', linewidth=2, 
                markersize=6, label=f'${elec_cost}/kWh (Opt: {diameters[optimal_idx_sens]*1000:.0f}mm)')

    ax.set_xlabel('Pipe Diameter [mm]', fontsize=12)
    ax.set_ylabel('Total NPV Cost [Million $]', fontsize=12)
    ax.set_title('Sensitivity to Electricity Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pressure_drop_sensitivity.png', dpi=300, bbox_inches='tight')
    print("Sensitivity analysis saved to 'pressure_drop_sensitivity.png'")
    plt.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
