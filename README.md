# CFD Industrial Simulator - Viscous Oil Flow in Pipelines

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

A professional Computational Fluid Dynamics (CFD) simulator focused on the oil and gas industry, specifically designed for analyzing viscous oil flow in pipelines with heat transfer capabilities.

## Features

- **Comprehensive Flow Analysis**: Laminar/turbulent flow, analytical solutions, pressure drop
- **Advanced Viscosity Models**: Walther (ASTM D341), Andrade, Beggs-Robinson
- **Oil Characterization**: API gravity-based properties, pre-configured oil types
- **Heat Transfer**: Convective transfer, temperature-dependent viscosity, insulation
- **Visualization**: Velocity profiles, temperature distribution, 2D contours
- **Animation & Dynamics**: Particle tracers, streamlines, animated fields
- **Industrial Applications**: Pipeline optimization, CAPEX vs OPEX analysis

## Quick Start

```python
from src.models.oil_properties import create_medium_oil
from src.geometry.pipe import Pipe
from src.core.boundary import InletBC, OutletBC
from src.core.solver import PipeFlowSolver

pipe = Pipe(diameter=0.2, length=100.0)
oil = create_medium_oil()
inlet = InletBC(velocity=1.0, temperature=293.15)
outlet = OutletBC(pressure=101325.0)

solver = PipeFlowSolver(pipe, oil, inlet, outlet)
results = solver.solve_flow()

print(f"Reynolds: {results['reynolds_number']:.0f}")
print(f"Regime: {results['flow_regime']}")
print(f"Pressure drop: {results['pressure_drop']/1e5:.4f} bar")
```

## Examples

Run complete simulations from the `examples/` directory:

```bash
# Basic simulations
python examples/basic_pipe_flow.py
python examples/heated_pipe_flow.py
python examples/viscosity_analysis.py
python examples/pressure_drop_analysis.py

# Animation examples
python examples/animated_pipe_flow.py
python examples/velocity_field_animation.py
python examples/temperature_evolution.py
```

## 🎬 Visualization and Animation

### Particle Tracers

Visualize fluid motion with massless particles following the flow:

```python
from src.visualization.animator import FlowAnimator
from src.visualization.particle_tracer import ParticleTracer

# After solving flow field
animator = FlowAnimator(fps=30)
anim = animator.animate_particle_tracers(
    velocity_field=(u_z, u_r),
    grid_coordinates=(z_grid, r_grid),
    n_particles=1000,
    duration=10,
    colorby=temperature_field,
    cmap='hot'
)
animator.save('flow_particles.mp4')
```

### Streamlines

Generate and visualize streamlines (lines tangent to velocity):

```python
from src.visualization.streamlines import StreamlineGenerator

streamlines = StreamlineGenerator(velocity_field, grid_coordinates)
streamlines.generate_streamlines(seed_points='auto', n_streamlines=20)
streamlines.plot_streamlines(
    color_by='velocity_magnitude',
    linewidth='variable',
    background=temperature_field,
    save_path='streamlines.png'
)
```

### Animated Fields

Create animations of evolving fields:

```python
# Velocity field animation
animator.animate_velocity_field(
    velocity_field,
    grid_coordinates,
    style='streamplot',
    show_magnitude=True
)
animator.save('velocity_field.gif')

# Temperature evolution
animator.animate_temperature_evolution(
    temperature_field,
    grid_coordinates,
    cmap='hot',
    show_isotherms=True
)
animator.save('thermal_evolution.mp4')
```

**Features:**
- Lagrangian particle tracking with RK4 integration
- Streamline generation with forward/backward integration
- Multiple visualization styles (quiver, streamplot, contours)
- Color mapping by temperature, velocity, or custom fields
- Export to MP4, GIF, or static images (PNG)
- Multi-view animations with synchronized subplots

## Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Theory](docs/theory.md)

## Testing

```bash
pytest tests/ -v
```

## Author

**Juniorsann** - [GitHub](https://github.com/Juniorsann)
