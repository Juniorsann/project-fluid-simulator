# User Guide - CFD Industrial Simulator

## Introduction

This guide will help you get started with the CFD Industrial Simulator, a specialized tool for analyzing viscous oil flow in pipelines with heat transfer.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/Juniorsann/project-fluid-simulator.git
cd project-fluid-simulator
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

### Example 1: Basic Pipe Flow

```python
from src.models.oil_properties import create_medium_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC
from src.core.solver import PipeFlowSolver

# Define geometry
pipe = Pipe(diameter=0.2, length=100.0)

# Define fluid
oil = create_medium_oil()

# Define boundary conditions
inlet = InletBC(velocity=1.0, temperature=293.15)
outlet = OutletBC(pressure=101325.0)

# Solve
solver = PipeFlowSolver(pipe, oil, inlet, outlet)
results = solver.solve_flow()

# Print results
print(f"Reynolds number: {results['reynolds_number']:.0f}")
print(f"Flow regime: {results['flow_regime']}")
print(f"Pressure drop: {results['pressure_drop']/1e5:.4f} bar")
```

### Example 2: Custom Oil Properties

```python
from src.models.oil_properties import OilFluid

# Create custom oil with API gravity
oil = OilFluid(api_gravity=25.0, name="my_crude")

# Check properties
print(f"Density at 20°C: {oil.density(293.15):.1f} kg/m³")
print(f"Viscosity at 40°C: {oil.dynamic_viscosity(313.15)*1000:.1f} cP")
```

## Core Concepts

### 1. Defining Geometry

The `Pipe` class represents a cylindrical pipe:

```python
from src.geometry.pipe import Pipe

pipe = Pipe(
    diameter=0.15,           # Internal diameter [m]
    length=50.0,             # Pipe length [m]
    material="carbon_steel", # Pipe material
    roughness=0.000046,      # Absolute roughness [m]
    insulation_thickness=0.05,  # Insulation [m]
    ambient_temperature=283.15  # External temp [K]
)
```

**Available materials:**
- `"carbon_steel"`: Common for oil pipelines
- `"stainless_steel"`: Corrosion resistant
- `"copper"`: High thermal conductivity
- `"pvc"`: Low cost, smooth

### 2. Defining Fluids

#### Pre-configured oils:

```python
from src.models.oil_properties import (
    create_light_oil,
    create_medium_oil,
    create_heavy_oil,
    create_extra_heavy_oil
)

light = create_light_oil()      # API ≈ 35°
medium = create_medium_oil()    # API ≈ 27°
heavy = create_heavy_oil()      # API ≈ 15°
extra_heavy = create_extra_heavy_oil()  # API ≈ 8°
```

#### Custom oil:

```python
from src.models.oil_properties import OilFluid

oil = OilFluid(
    api_gravity=30.0,
    name="custom_crude"
)
```

### 3. Boundary Conditions

#### Inlet Conditions:

```python
from src.core.boundary import InletBC

# Specify velocity
inlet = InletBC(velocity=1.5, temperature=293.15)

# OR specify mass flow rate
inlet = InletBC(mass_flow_rate=10.0, temperature=293.15)
```

#### Outlet Conditions:

```python
from src.core.boundary import OutletBC

outlet = OutletBC(pressure=101325.0)  # Atmospheric
```

#### Wall Conditions:

```python
from src.core.boundary import WallBC

# Isothermal wall
wall = WallBC(temperature=350.15)

# Constant heat flux
wall = WallBC(heat_flux=1000.0)  # W/m²

# Adiabatic (insulated)
wall = WallBC()
```

### 4. Solving Flow Problems

#### Flow Analysis:

```python
from src.core.solver import PipeFlowSolver

solver = PipeFlowSolver(pipe, oil, inlet, outlet)
results = solver.solve_flow()

# Access results
Re = results['reynolds_number']
regime = results['flow_regime']  # 'laminar', 'transitional', or 'turbulent'
dp = results['pressure_drop']    # Pa
```

#### Heat Transfer Analysis:

```python
from src.core.solver import HeatTransferSolver
import numpy as np

heat_solver = HeatTransferSolver(
    pipe, oil,
    inlet_temperature=293.15,
    wall_bc=wall,
    mean_velocity=1.0
)

# Calculate temperature along pipe
z = np.linspace(0, pipe.length, 100)
T = heat_solver.solve_temperature_profile_1d(z)
```

## Visualization

### Plotting Results

```python
from src.visualization.plotter import (
    plot_velocity_profile,
    plot_temperature_profile,
    plot_viscosity_temperature
)

# Velocity profile
plot_velocity_profile(r, u, radius, save_path="velocity.png")

# Temperature profile
plot_temperature_profile(z, T, length, save_path="temperature.png")

# Viscosity vs temperature
plot_viscosity_temperature(T_array, mu_array, save_path="viscosity.png")
```

## Running Examples

The `examples/` directory contains complete simulation scripts:

```bash
# Basic pipe flow
python examples/basic_pipe_flow.py

# Heated pipe flow
python examples/heated_pipe_flow.py

# Viscosity analysis
python examples/viscosity_analysis.py

# Pressure drop optimization
python examples/pressure_drop_analysis.py
```

## Best Practices

### 1. Grid Resolution

- **1D problems:** Use 100-200 cells for smooth profiles
- **2D problems:** Use 50-100 radial cells, more near walls
- Enable wall refinement for accurate boundary layer resolution

### 2. Temperature Range

- Keep temperatures between 250K and 400K for validity of correlations
- For Beggs-Robinson: use 294K-419K (70°F-295°F)

### 3. Reynolds Number

- Re < 2300: Laminar (analytical solutions available)
- 2300 < Re < 4000: Transitional (use turbulent correlations)
- Re > 4000: Turbulent (friction factor from Colebrook-White)

### 4. Physical Validation

Always check:
- Velocity is reasonable (typically 0.5-5 m/s for oil pipelines)
- Pressure drop is positive
- Temperature increases/decreases make physical sense
- Reynolds number matches expected flow regime

## Troubleshooting

### Issue: Very high pressure drop

**Possible causes:**
- Oil too viscous (try heating)
- Pipe diameter too small
- Flow rate too high

**Solutions:**
- Increase temperature
- Increase pipe diameter
- Reduce flow rate

### Issue: Unrealistic viscosity values

**Possible causes:**
- Temperature out of valid range
- Incorrect API gravity

**Solutions:**
- Check temperature units (should be Kelvin)
- Verify API gravity is correct
- Use different viscosity model

### Issue: Flow regime unexpected

**Possible causes:**
- Incorrect fluid properties
- Wrong diameter or velocity

**Solutions:**
- Check Reynolds number calculation
- Verify all properties at correct temperature
- Ensure consistent units

## Export Results

```python
from src.utils.exporters import export_to_csv

# Export data
data = {
    'position': z,
    'temperature': T,
    'pressure': p
}

export_to_csv(data, 'results.csv')
```

## Advanced Topics

### Custom Viscosity Models

```python
from src.models.viscosity import fit_walther_parameters, WaltherViscosity

# Fit from two data points
T1, nu1 = 293.15, 50e-6
T2, nu2 = 373.15, 10e-6

A, B = fit_walther_parameters(T1, nu1, T2, nu2)
model = WaltherViscosity(A, B)

# Use in OilFluid
oil = OilFluid(api_gravity=30.0, viscosity_model=model)
```

### 2D Velocity Profiles

```python
from src.core.grid import Grid2DAxisymmetric

# Create 2D grid
grid = Grid2DAxisymmetric(
    radius=pipe.radius(),
    length=1.0,  # Just show first meter
    n_radial=50,
    n_axial=100,
    wall_refinement=True
)

# Solve 2D velocity field
u_r, u_z = solver.solve_velocity_profile_2d(grid)

# Visualize
from src.visualization.plotter import plot_2d_contour
plot_2d_contour(grid.r, grid.z, u_z, 
                field_name="Velocity [m/s]",
                save_path="velocity_2d.png")
```

## Support

For questions or issues:
- GitHub Issues: https://github.com/Juniorsann/project-fluid-simulator/issues
- Documentation: See `docs/` directory
- Examples: See `examples/` directory
