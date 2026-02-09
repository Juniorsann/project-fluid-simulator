# API Reference

## Core Modules

### `src.core.properties`

#### `FluidProperties`

Base class for fluid properties.

```python
FluidProperties(density, viscosity, thermal_conductivity=None, 
                specific_heat=None, name="fluid")
```

**Methods:**
- `kinematic_viscosity()` → float: Calculate ν = μ/ρ
- `reynolds_number(velocity, length_scale)` → float: Calculate Re
- `prandtl_number()` → float: Calculate Pr = μcp/k

### `src.core.grid`

#### `Grid1D`

One-dimensional uniform grid.

```python
Grid1D(length, n_cells)
```

**Attributes:**
- `x`: Cell center positions
- `x_faces`: Cell face positions
- `dx`: Cell spacing

#### `Grid2DAxisymmetric`

Two-dimensional axisymmetric (r,z) grid.

```python
Grid2DAxisymmetric(radius, length, n_radial, n_axial, 
                   wall_refinement=True, wall_refinement_ratio=0.1)
```

**Attributes:**
- `r`, `z`: Cell center positions
- `r_faces`, `z_faces`: Cell face positions
- `R`, `Z`: 2D meshgrid

**Methods:**
- `cell_volume(i_radial, i_axial)` → float

### `src.core.boundary`

#### `InletBC`

Inlet boundary condition.

```python
InletBC(velocity=None, mass_flow_rate=None, temperature=None, name="inlet")
```

One of `velocity` or `mass_flow_rate` must be specified.

#### `OutletBC`

Outlet boundary condition.

```python
OutletBC(pressure=101325.0, name="outlet")
```

#### `WallBC`

Wall boundary condition (no-slip).

```python
WallBC(temperature=None, heat_flux=None, name="wall")
```

Cannot specify both `temperature` and `heat_flux`.

### `src.core.solver`

#### `PoiseuilleFlow`

Analytical laminar pipe flow.

```python
PoiseuilleFlow(pipe, fluid, temperature)
```

**Methods:**
- `velocity_profile(r, mean_velocity)` → ndarray: Parabolic profile
- `pressure_drop(mean_velocity, length=None)` → float: Hagen-Poiseuille
- `reynolds_number(mean_velocity)` → float

#### `PipeFlowSolver`

General pipe flow solver.

```python
PipeFlowSolver(pipe, fluid, inlet_bc, outlet_bc, wall_bc=None)
```

**Methods:**
- `solve_flow()` → dict: Returns {mean_velocity, reynolds_number, flow_regime, pressure_drop}
- `solve_velocity_profile_2d(grid)` → (u_r, u_z): 2D velocity field

#### `HeatTransferSolver`

Heat transfer in pipe flow.

```python
HeatTransferSolver(pipe, fluid, inlet_temperature, wall_bc, mean_velocity)
```

**Methods:**
- `solve_temperature_profile_1d(z)` → ndarray: Temperature along pipe

## Model Modules

### `src.models.viscosity`

#### Functions

- `walther_equation(temperature, A, B)` → float: ASTM D341
- `andrade_equation(temperature, A, B)` → float: Exponential model
- `beggs_robinson_dead_oil(temperature, api_gravity)` → float: Crude oil
- `fit_walther_parameters(T1, nu1, T2, nu2)` → (A, B)
- `fit_andrade_parameters(T1, mu1, T2, mu2)` → (A, B)

#### Classes

- `WaltherViscosity(A, B)`: ASTM D341 model
- `AndradeViscosity(A, B)`: Exponential model
- `BeggsRobinsonViscosity(api_gravity)`: Beggs-Robinson model

### `src.models.oil_properties`

#### `OilFluid`

Oil with temperature-dependent properties.

```python
OilFluid(api_gravity, temperature_ref=288.15, viscosity_model=None, name="crude_oil")
```

**Methods:**
- `density(temperature=None)` → float: ρ [kg/m³]
- `dynamic_viscosity(temperature)` → float: μ [Pa·s]
- `kinematic_viscosity(temperature)` → float: ν [m²/s]
- `thermal_conductivity(temperature=None)` → float: k [W/(m·K)]
- `specific_heat(temperature=None)` → float: cp [J/(kg·K)]
- `reynolds_number(velocity, diameter, temperature)` → float
- `prandtl_number(temperature=None)` → float

#### Helper Functions

- `create_light_oil(name="light_crude")` → OilFluid: API ≈ 35°
- `create_medium_oil(name="medium_crude")` → OilFluid: API ≈ 27°
- `create_heavy_oil(name="heavy_crude")` → OilFluid: API ≈ 15°
- `create_extra_heavy_oil(name="extra_heavy_crude")` → OilFluid: API ≈ 8°

### `src.models.turbulence`

#### Functions

- `friction_factor_laminar(reynolds)` → float: f = 64/Re
- `friction_factor_turbulent(reynolds, relative_roughness=0.0)` → float: Colebrook-White
- `friction_factor(reynolds, relative_roughness=0.0)` → float: Auto-select

#### Classes

- `MixingLengthModel(kappa=0.41)`: Prandtl mixing length
- `KEpsilonConstants`: Constants for k-ε model

## Geometry Modules

### `src.geometry.pipe`

#### `Pipe`

Cylindrical pipe geometry.

```python
Pipe(diameter, length, roughness=None, material="carbon_steel",
     wall_thickness=None, insulation_thickness=0.0,
     insulation_conductivity=0.04, ambient_temperature=288.15, name="pipe")
```

**Methods:**
- `cross_sectional_area()` → float: A = πD²/4
- `radius()` → float: R = D/2
- `relative_roughness()` → float: ε/D
- `volume()` → float: Internal volume
- `mean_velocity(volumetric_flow_rate)` → float
- `volumetric_flow_rate(mean_velocity)` → float
- `overall_heat_transfer_coefficient(internal_h, external_h=10.0)` → float

**Material Constants:**
- `PipeMaterial.CARBON_STEEL`
- `PipeMaterial.STAINLESS_STEEL`
- `PipeMaterial.COPPER`
- `PipeMaterial.PVC`

### `src.geometry.domain`

#### `Domain`

1D computational domain.

```python
Domain(pipe, grid=None, name="domain")
```

#### `Domain2DAxisymmetric`

2D axisymmetric domain.

```python
Domain2DAxisymmetric(pipe, n_radial=50, n_axial=100, 
                     wall_refinement=True, name="domain_2d")
```

## Visualization Modules

### `src.visualization.plotter`

#### Functions

- `plot_velocity_profile(r, u, radius, title="...", save_path=None)` → Figure
- `plot_temperature_profile(z, T, length, title="...", save_path=None)` → Figure
- `plot_viscosity_temperature(temperatures, viscosities, oil_names=None, ...)` → Figure
- `plot_pressure_drop(z, pressure, title="...", save_path=None)` → Figure
- `plot_2d_contour(r, z, field, field_name="...", cmap='viridis', ...)` → Figure
- `plot_reynolds_number_evolution(positions, reynolds, ...)` → Figure

### `src.visualization.animator`

#### `FlowAnimator`

Animation creator.

```python
FlowAnimator(figure_size=(10, 6), fps=30)
```

**Methods:**
- `animate_1d_profile(time_steps, positions, field_data, ...)` → FuncAnimation
- `animate_2d_field(time_steps, x, y, field_data, ...)` → FuncAnimation

## Utility Modules

### `src.utils.validators`

Validation functions:

- `validate_positive(value, name="value")` → float
- `validate_non_negative(value, name="value")` → float
- `validate_range(value, min_val, max_val, name="value")` → float
- `validate_temperature(temperature, min_temp=0)` → float
- `validate_api_gravity(api_gravity)` → float
- `validate_reynolds_number(reynolds)` → (float, str): Returns (Re, regime)
- `validate_pressure(pressure, name="pressure")` → float
- `validate_velocity(velocity)` → float
- `validate_grid_size(n_cells, name="n_cells")` → int

### `src.utils.exporters`

Export functions:

- `export_to_csv(data, filename, header=None)`: Export dict of arrays
- `export_results_to_csv(positions, results, filename, metadata=None)`
- `export_to_json(data, filename, indent=2)`: Export to JSON
- `save_simulation_summary(results, filename="simulation_summary.json")`

#### `ResultsExporter`

```python
ResultsExporter(output_dir="output")
```

**Methods:**
- `export(data, base_name, formats=['csv', 'json'])` → dict

## Usage Examples

### Complete Flow Analysis

```python
from src.models.oil_properties import create_heavy_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC
from src.core.solver import PipeFlowSolver
from src.visualization.plotter import plot_velocity_profile

# Setup
pipe = Pipe(diameter=0.2, length=100.0)
oil = create_heavy_oil()
inlet = InletBC(velocity=1.0, temperature=313.15)
outlet = OutletBC(pressure=101325.0)

# Solve
solver = PipeFlowSolver(pipe, oil, inlet, outlet)
results = solver.solve_flow()

# Results
print(f"Re = {results['reynolds_number']:.0f}")
print(f"Regime: {results['flow_regime']}")
print(f"Δp = {results['pressure_drop']/1e5:.3f} bar")
```

### Export Results

```python
from src.utils.exporters import export_results_to_csv
import numpy as np

z = np.linspace(0, 100, 100)
T = 293.15 + 10 * z / 100

results = {'temperature': T, 'pressure': p}
metadata = {'oil': 'heavy_crude', 'diameter': 0.2}

export_results_to_csv(z, results, 'output.csv', metadata)
```
