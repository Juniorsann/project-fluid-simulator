"""
Viscosity Analysis Example

This example compares different viscosity models and oil types,
analyzing the temperature dependence of viscosity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.models.oil_properties import (
    create_light_oil, create_medium_oil, create_heavy_oil, create_extra_heavy_oil
)
from src.models.viscosity import (
    walther_equation, andrade_equation, beggs_robinson_dead_oil,
    fit_walther_parameters, fit_andrade_parameters
)
from src.visualization.plotter import plot_viscosity_temperature


def main():
    """Run viscosity analysis."""
    print("=" * 60)
    print("VISCOSITY ANALYSIS")
    print("=" * 60)

    # Temperature range for analysis
    temp_C = np.linspace(20, 100, 50)
    temp_K = temp_C + 273.15

    # Compare different oil types
    print("\n" + "=" * 60)
    print("COMPARISON OF OIL TYPES")
    print("=" * 60)

    oils = [
        create_light_oil("Light Crude"),
        create_medium_oil("Medium Crude"),
        create_heavy_oil("Heavy Crude"),
        create_extra_heavy_oil("Extra Heavy Crude")
    ]

    print(f"\n{'Oil Type':<20} {'API°':<10} {'ρ @ 15°C':<15} {'μ @ 20°C (cP)':<15}")
    print("-" * 60)
    
    for oil in oils:
        rho = oil.density(288.15)  # 15°C
        mu = oil.dynamic_viscosity(293.15)  # 20°C
        print(f"{oil.name:<20} {oil.api_gravity:<10.1f} {rho:<15.1f} {mu*1000:<15.1f}")

    # Calculate viscosity vs temperature for all oils
    viscosities = []
    for oil in oils:
        visc = np.array([oil.dynamic_viscosity(T) for T in temp_K])
        viscosities.append(visc)

    # Plot comparison
    viscosities_array = np.array(viscosities)
    oil_names = [oil.name for oil in oils]
    
    plot_viscosity_temperature(
        temp_K, viscosities_array,
        oil_names=oil_names,
        title="Viscosity vs Temperature - Oil Type Comparison",
        save_path="viscosity_oil_comparison.png"
    )
    print("\nOil comparison plot saved to 'viscosity_oil_comparison.png'")

    # Analyze viscosity models for medium crude
    print("\n" + "=" * 60)
    print("VISCOSITY MODEL COMPARISON - MEDIUM CRUDE")
    print("=" * 60)

    oil = create_medium_oil()
    api = oil.api_gravity

    # Reference points for fitting
    T1, T2 = 293.15, 373.15  # 20°C, 100°C
    mu1 = oil.dynamic_viscosity(T1)
    mu2 = oil.dynamic_viscosity(T2)
    nu1 = mu1 / oil.density(T1)
    nu2 = mu2 / oil.density(T2)

    print(f"\nReference points for model fitting:")
    print(f"T1 = {T1-273.15:.0f}°C: μ = {mu1*1000:.2f} cP")
    print(f"T2 = {T2-273.15:.0f}°C: μ = {mu2*1000:.2f} cP")

    # Fit models
    A_walther, B_walther = fit_walther_parameters(T1, nu1, T2, nu2)
    A_andrade, B_andrade = fit_andrade_parameters(T1, mu1, T2, mu2)

    print(f"\nWalther parameters: A = {A_walther:.4f}, B = {B_walther:.4f}")
    print(f"Andrade parameters: A = {A_andrade:.6f} Pa·s, B = {B_andrade:.1f} K")

    # Calculate viscosity with different models
    visc_beggs = np.array([beggs_robinson_dead_oil(T, api) for T in temp_K])
    visc_andrade = np.array([andrade_equation(T, A_andrade, B_andrade) for T in temp_K])
    visc_walther = np.array([
        walther_equation(T, A_walther, B_walther) * oil.density(T) 
        for T in temp_K
    ])

    # Plot model comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.semilogy(temp_C, visc_beggs*1000, 'b-', linewidth=2, label='Beggs-Robinson', marker='o', markersize=4)
    ax.semilogy(temp_C, visc_andrade*1000, 'r--', linewidth=2, label='Andrade')
    ax.semilogy(temp_C, visc_walther*1000, 'g-.', linewidth=2, label='Walther')
    
    # Mark reference points
    ax.plot([T1-273.15, T2-273.15], [mu1*1000, mu2*1000], 
            'ko', markersize=10, label='Reference Points', zorder=5)

    ax.set_xlabel('Temperature [°C]', fontsize=12)
    ax.set_ylabel('Dynamic Viscosity [cP]', fontsize=12)
    ax.set_title('Viscosity Model Comparison - Medium Crude', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('viscosity_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison plot saved to 'viscosity_model_comparison.png'")
    plt.close()

    # Analyze temperature sensitivity
    print("\n" + "=" * 60)
    print("TEMPERATURE SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Calculate viscosity index (rate of change)
    dT = 10  # 10°C increment
    for oil in oils:
        mu_20 = oil.dynamic_viscosity(293.15)
        mu_30 = oil.dynamic_viscosity(303.15)
        mu_40 = oil.dynamic_viscosity(313.15)
        
        # Percentage change per 10°C
        change_20_30 = (mu_20 - mu_30) / mu_20 * 100
        change_30_40 = (mu_30 - mu_40) / mu_30 * 100
        
        print(f"\n{oil.name}:")
        print(f"  20-30°C: {change_20_30:.1f}% reduction")
        print(f"  30-40°C: {change_30_40:.1f}% reduction")
        print(f"  Average: {(change_20_30 + change_30_40)/2:.1f}% per 10°C")

    # Flow regime impact
    print("\n" + "=" * 60)
    print("IMPACT ON FLOW REGIME")
    print("=" * 60)

    # Fixed conditions
    diameter = 0.2  # m
    velocity = 1.0  # m/s

    print(f"\nFixed conditions: D = {diameter*1000:.0f} mm, u = {velocity} m/s")
    print(f"\n{'Oil':<20} {'T (°C)':<10} {'Re':<12} {'Regime':<15}")
    print("-" * 60)

    for oil in oils[:3]:  # Light, Medium, Heavy
        for T in [293.15, 323.15, 353.15]:  # 20, 50, 80°C
            Re = oil.reynolds_number(velocity, diameter, T)
            regime = "Laminar" if Re < 2300 else ("Transitional" if Re < 4000 else "Turbulent")
            print(f"{oil.name:<20} {T-273.15:<10.0f} {Re:<12.0f} {regime:<15}")

    # Create comprehensive summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Viscosity vs Temperature (all oils)
    for i, oil in enumerate(oils):
        visc = np.array([oil.dynamic_viscosity(T) for T in temp_K])
        ax1.semilogy(temp_C, visc*1000, linewidth=2, marker='o', 
                    markersize=3, label=oil.name)
    ax1.set_xlabel('Temperature [°C]', fontsize=11)
    ax1.set_ylabel('Dynamic Viscosity [cP]', fontsize=11)
    ax1.set_title('Viscosity vs Temperature', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=9)

    # Plot 2: Density vs Temperature (all oils)
    for oil in oils:
        density = np.array([oil.density(T) for T in temp_K])
        ax2.plot(temp_C, density, linewidth=2, marker='o', 
                markersize=3, label=oil.name)
    ax2.set_xlabel('Temperature [°C]', fontsize=11)
    ax2.set_ylabel('Density [kg/m³]', fontsize=11)
    ax2.set_title('Density vs Temperature', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Plot 3: Kinematic Viscosity
    for oil in oils:
        nu = np.array([oil.kinematic_viscosity(T) for T in temp_K])
        ax3.semilogy(temp_C, nu*1e6, linewidth=2, marker='o', 
                    markersize=3, label=oil.name)
    ax3.set_xlabel('Temperature [°C]', fontsize=11)
    ax3.set_ylabel('Kinematic Viscosity [cSt]', fontsize=11)
    ax3.set_title('Kinematic Viscosity vs Temperature', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=9)

    # Plot 4: Reynolds Number (fixed D, u)
    for oil in oils:
        Re = np.array([oil.reynolds_number(velocity, diameter, T) for T in temp_K])
        ax4.semilogy(temp_C, Re, linewidth=2, marker='o', 
                    markersize=3, label=oil.name)
    ax4.axhline(y=2300, color='r', linestyle='--', alpha=0.5, label='Laminar limit')
    ax4.axhline(y=4000, color='orange', linestyle='--', alpha=0.5, label='Turbulent limit')
    ax4.set_xlabel('Temperature [°C]', fontsize=11)
    ax4.set_ylabel('Reynolds Number', fontsize=11)
    ax4.set_title(f'Reynolds Number (D={diameter*1000:.0f}mm, u={velocity}m/s)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('viscosity_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive analysis saved to 'viscosity_comprehensive_analysis.png'")
    plt.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
