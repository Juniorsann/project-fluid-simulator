# Visualization Gallery

This gallery showcases the visualization capabilities of the CFD Industrial Simulator, demonstrating various techniques for visualizing oil flow in pipes.

## Table of Contents

1. [Particle Tracers](#particle-tracers)
2. [Streamlines](#streamlines)
3. [Temperature Evolution](#temperature-evolution)
4. [Velocity Fields](#velocity-fields)
5. [Comparison Visualizations](#comparison-visualizations)
6. [Advanced Techniques](#advanced-techniques)

---

## Particle Tracers

### Basic Particle Tracing

Particle tracers visualize fluid motion by following massless particles through the velocity field.

```python
from src.visualization.particle_tracer import ParticleTracer

# Setup particle tracer
tracer = ParticleTracer(
    velocity_field=(u_r, u_z),
    grid=(r, z)
)

# Release particles at inlet
inlet_positions = np.column_stack([
    np.linspace(0, pipe.radius(), 50),
    np.zeros(50)
])
tracer.add_particles(inlet_positions)

# Advect particles
for step in range(100):
    tracer.update(dt=0.01)
    positions = tracer.get_positions()
    # Plot positions...
```

**Use Cases**:
- Visualizing flow paths
- Understanding residence time distribution
- Identifying recirculation zones
- Educational demonstrations

### Colored by Velocity

```python
# Color particles by velocity magnitude
colors = tracer.get_colors(color_by='velocity')
```

Shows regions of high and low velocity. Useful for:
- Identifying stagnation points
- Visualizing velocity gradients
- Understanding flow acceleration/deceleration

### Colored by Temperature

```python
# Color particles by temperature
colors = tracer.get_colors(color_by='temperature')
```

Ideal for visualizing:
- Heat transfer effectiveness
- Thermal boundary layers
- Hot/cold zones in the flow

### Colored by Age (Residence Time)

```python
# Color particles by age
colors = tracer.get_colors(color_by='age')
```

Useful for:
- Residence time distribution
- Identifying dead zones
- Optimizing heat exchanger design

---

## Streamlines

### Static Streamlines

Streamlines are curves everywhere tangent to the velocity field at a given instant.

```python
from src.visualization.streamlines import StreamlineGenerator

gen = StreamlineGenerator(
    velocity_field=(u_r, u_z),
    grid=(r, z)
)

# Generate from multiple seed points
seed_points = np.array([
    [0.2*R, 0.0],
    [0.5*R, 0.0],
    [0.8*R, 0.0]
])

streamlines = gen.generate_streamlines(
    seed_points,
    max_length=pipe.length,
    step_size=0.01
)
```

**Interpretation**:
- Streamlines show instantaneous flow direction
- Spacing indicates velocity (close = slow, far = fast)
- Never cross in steady flow

### Colored Streamlines

```python
from src.visualization.plotter import plot_streamlines_static

plot_streamlines_static(
    results,
    n_lines=30,
    color_by='velocity',
    save_path='streamlines.png'
)
```

Color coding options:
- `'velocity'`: Show speed along streamline
- `'temperature'`: Show thermal field

### Pathlines vs Streamlines

In **steady flow**: pathlines = streamlines  
In **unsteady flow**: pathlines ≠ streamlines

```python
# Generate pathlines
time_steps = np.linspace(0, 10, 100)
pathlines = gen.generate_pathlines(seed_points, time_steps)
```

---

## Temperature Evolution

### Axial Temperature Profile

```python
from src.visualization.plotter import plot_temperature_profile

# Plot temperature along pipe axis
plot_temperature_profile(
    z=z_positions,
    T=temperatures,
    length=pipe.length,
    title='Temperature Evolution Along Pipe',
    save_path='temp_profile.png'
)
```

Shows:
- Temperature rise/drop along pipe
- Effectiveness of heating/cooling
- Thermal entrance length

### 2D Temperature Distribution

```python
from src.visualization.plotter import plot_temperature_distribution_2d

plot_temperature_distribution_2d(
    results,
    view='cross_section',
    position=0.5,  # Halfway along pipe
    save_path='temp_2d.png'
)
```

Visualizes:
- Radial temperature gradients
- Thermal boundary layer development
- Hot/cold spots

### Animated Temperature with Particles

```python
animator = FlowAnimator(results)
animator.add_temperature_map(cmap='RdYlBu_r', levels=30, alpha=0.8)
animator.add_particle_tracers(n_particles=200, color_scheme='temperature')
animator.add_isotherms(temperatures=[50, 60, 70, 80], linewidth=2)
animator.save_video('temp_evolution.mp4', duration=10, fps=30)
```

Perfect for:
- Presentations
- Understanding heat transfer dynamics
- Educational videos

---

## Velocity Fields

### Velocity Profile

```python
from src.visualization.plotter import plot_velocity_profile

plot_velocity_profile(
    r=radial_positions,
    u=velocities,
    radius=pipe.radius(),
    title='Radial Velocity Profile',
    save_path='velocity_profile.png'
)
```

Shows:
- Parabolic profile (laminar)
- Flatter profile (turbulent)
- Wall effects

### 2D Velocity Field with Arrows

```python
animator = FlowAnimator(results)
animator.add_velocity_field(
    style='arrows',
    density=25,
    scale=1.5
)
animator.save_video('velocity_arrows.mp4', duration=8, fps=30)
```

Arrow visualization shows:
- Flow direction
- Velocity magnitude (arrow length)
- Velocity field structure

### Velocity Comparison

```python
from src.visualization.plotter import plot_velocity_profile_comparison

plot_velocity_profile_comparison(
    results_list=[(r1, u1), (r2, u2), (r3, u3)],
    labels=['Laminar', 'Transitional', 'Turbulent'],
    save_path='velocity_comparison.png'
)
```

Useful for:
- Regime comparison
- Effect of viscosity
- Temperature influence on flow

---

## Comparison Visualizations

### Side-by-Side Scenarios

```python
from src.visualization.animator import ComparisonAnimator

comparison = ComparisonAnimator(
    [results_case1, results_case2],
    labels=['Base Case', 'Optimized']
)

comparison.add_particle_tracers(n_particles=150)
comparison.add_temperature_map(cmap='hot')
comparison.save_video('comparison.mp4', duration=12, fps=30)
```

Applications:
- Design optimization
- Parameter studies
- What-if analysis

### Multi-Parameter Comparison

Compare effects of:
- Different inlet temperatures
- Various wall temperatures
- Different flow rates
- Multiple viscosity grades

---

## Advanced Techniques

### Vorticity Visualization

```python
from src.visualization.plotter import plot_vorticity_field

plot_vorticity_field(
    results,
    levels=20,
    save_path='vorticity.png'
)
```

Vorticity (∇ × **u**) shows:
- Flow rotation
- Vortex cores
- Shear layers

### Vortex Identification

```python
from src.visualization.streamlines import StreamlineGenerator

gen = StreamlineGenerator(velocity_field=(u_r, u_z), grid=(r, z))

# Q-criterion for vortex detection
vortex_mask = gen.identify_vortices(method='q_criterion', threshold=0.1)

# Or lambda2 method
vortex_mask = gen.identify_vortices(method='lambda2')
```

Methods:
- **Q-criterion**: Based on rotation vs strain
- **Lambda2**: Eigenvalue analysis

### Wall Shear Stress

```python
from src.visualization.plotter import plot_wall_shear_stress

plot_wall_shear_stress(
    results,
    save_path='wall_shear.png'
)
```

Important for:
- Pressure drop calculation
- Erosion prediction
- Heat transfer coefficient

### Combined Visualizations

Layer multiple visualizations:

```python
animator = FlowAnimator(results, figure_size=(16, 8))

# Add multiple layers
animator.add_temperature_map(cmap='hot', alpha=0.5)
animator.add_streamlines(n_lines=40)
animator.add_particle_tracers(n_particles=300, color_scheme='temperature')
animator.add_velocity_field(style='arrows', density=15, scale=1.0)

# Add annotations
animator.add_title('Complete Flow Visualization')
animator.add_text(0.02, 0.95, f'Re = {Re:.0f}', fontsize=12)
animator.add_text(0.02, 0.90, f'ΔT = {dT:.1f}°C', fontsize=12)
animator.add_colorbar(label='Temperature [°C]')

# Save
animator.save_video('combined_viz.mp4', duration=15, fps=30, dpi=200)
```

---

## Best Practices

### For Publications

- Use high DPI (≥200)
- Clear labels and units
- Consistent color schemes
- Publication-quality colormaps (avoid rainbow)

### For Presentations

- Large fonts (≥14pt)
- High contrast colors
- Simple, focused visualizations
- Smooth animations (30 FPS)

### For Web/Social Media

- Moderate resolution (DPI=100-150)
- GIF format for compatibility
- Short duration (5-10 seconds)
- Lower FPS for smaller files (15 FPS)

### Performance Optimization

```python
# Quick preview
animator.save_video('preview.mp4', duration=3, fps=15, dpi=100)

# High quality final
animator.save_video('final.mp4', duration=10, fps=30, dpi=200)
```

---

## Color Schemes

### Temperature

- `'hot'`: Black → Red → Yellow → White
- `'coolwarm'`: Blue → White → Red
- `'RdYlBu_r'`: Red → Yellow → Blue (reversed)

### Velocity

- `'viridis'`: Perceptually uniform
- `'plasma'`: High contrast
- `'jet'`: Classic (use sparingly)

### Pressure

- `'Blues'`: Sequential blues
- `'RdBu'`: Diverging red-blue

---

## Example Workflows

### 1. Quick Flow Check

```python
# Fast visualization for debugging
animator = FlowAnimator(results)
animator.add_streamlines(n_lines=20)
animator.add_particle_tracers(n_particles=100)
animator.save_video('quick_check.mp4', duration=3, fps=15, dpi=100)
```

### 2. Publication Figure

```python
from src.visualization.plotter import plot_2d_contour

plot_2d_contour(
    r=grid.r,
    z=grid.z,
    field=velocity_magnitude,
    field_name='Velocity [m/s]',
    cmap='viridis',
    title='Velocity Distribution in Heated Pipe',
    save_path='publication_fig.png'
)
```

### 3. Educational Animation

```python
animator = FlowAnimator(results, figure_size=(14, 8))
animator.add_temperature_map(cmap='hot', alpha=0.7)
animator.add_particle_tracers(n_particles=500, color_scheme='temperature')
animator.add_title('Oil Heating in Steam-Traced Pipeline')
animator.add_text(0.02, 0.95, 'Inlet: 40°C', fontsize=14)
animator.add_text(0.02, 0.90, 'Wall: 80°C', fontsize=14)
animator.save_video('educational.mp4', duration=20, fps=30, dpi=150)
```

---

## Additional Resources

- See `examples/` directory for complete scripts
- Matplotlib colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
- FFmpeg documentation: https://ffmpeg.org/documentation.html
- Scientific visualization principles: Tufte's "The Visual Display of Quantitative Information"

---

*For more information, see the [User Guide](user_guide.md) and [API Reference](api_reference.md).*
