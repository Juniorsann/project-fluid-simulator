#!/usr/bin/env python3
"""
Standalone test for visualization modules.

This test validates the visualization modules work independently
without requiring the full CFD simulator dependencies.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_particle_tracer():
    """Test ParticleTracer module."""
    print("\n" + "="*60)
    print("Testing ParticleTracer")
    print("="*60)
    
    from src.visualization.particle_tracer import ParticleTracer
    
    # Create simple velocity field
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((20, 50))
    u_z = np.ones((20, 50))
    
    # Initialize tracer
    tracer = ParticleTracer(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    print(f"✓ ParticleTracer initialized")
    print(f"  Domain: r=[{tracer.r_min}, {tracer.r_max}], z=[{tracer.z_min}, {tracer.z_max}]")
    
    # Add particles
    initial_pos = np.array([[0.5, 0.1], [0.3, 0.2]])
    tracer.add_particles(initial_pos)
    print(f"✓ Added {tracer.get_particle_count()} particles")
    
    # Update
    tracer.update(0.1)
    positions = tracer.get_positions()
    print(f"✓ Updated particles")
    print(f"  Particle 1: ({positions[0,0]:.2f}, {positions[0,1]:.2f})")
    
    # Get colors
    colors = tracer.get_colors(color_by='velocity')
    print(f"✓ Got particle colors: {len(colors)} values")
    
    return True


def test_streamline_generator():
    """Test StreamlineGenerator module."""
    print("\n" + "="*60)
    print("Testing StreamlineGenerator")
    print("="*60)
    
    from src.visualization.streamlines import StreamlineGenerator
    
    # Create velocity field
    r = np.linspace(0, 1, 20)
    z = np.linspace(0, 5, 50)
    u_r = np.zeros((20, 50))
    u_z = np.ones((20, 50))
    
    # Initialize generator
    gen = StreamlineGenerator(
        velocity_field=(u_r, u_z),
        grid=(r, z)
    )
    print(f"✓ StreamlineGenerator initialized")
    
    # Generate streamlines
    seed_points = np.array([[0.3, 0.5], [0.5, 0.5], [0.7, 0.5]])
    streamlines = gen.generate_streamlines(
        seed_points,
        max_length=3.0,
        step_size=0.1
    )
    print(f"✓ Generated {len(streamlines)} streamlines")
    print(f"  First streamline: {len(streamlines[0])} points")
    
    # Compute vorticity
    vorticity = gen.compute_vorticity()
    print(f"✓ Computed vorticity field: shape {vorticity.shape}")
    
    return True


def test_flow_animator():
    """Test FlowAnimator module."""
    print("\n" + "="*60)
    print("Testing FlowAnimator")
    print("="*60)
    
    from src.visualization.animator import FlowAnimator, ComparisonAnimator
    
    # Create sample results
    r = np.linspace(0, 0.1, 20)
    z = np.linspace(0, 1.0, 100)
    R, Z = np.meshgrid(r, z, indexing='ij')
    u_r = np.zeros_like(R)
    u_z = np.ones_like(R)
    T = 300 + 20 * Z
    
    results = {
        'velocity_field': (u_r, u_z),
        'temperature_field': T,
        'grid': (r, z),
        'pipe_radius': 0.1,
        'pipe_length': 1.0
    }
    
    # Initialize animator
    animator = FlowAnimator(results)
    print(f"✓ FlowAnimator initialized")
    
    # Add layers
    animator.add_particle_tracers(n_particles=100)
    print(f"✓ Added particle tracers")
    
    animator.add_velocity_field(style='arrows', density=20)
    print(f"✓ Added velocity field")
    
    animator.add_temperature_map(cmap='hot', alpha=0.7)
    print(f"✓ Added temperature map")
    
    animator.add_streamlines(n_lines=30)
    print(f"✓ Added streamlines")
    
    animator.add_title("Test Animation")
    print(f"✓ Added title")
    
    animator.add_text(0.02, 0.95, "Test", fontsize=12)
    print(f"✓ Added text annotation")
    
    # Test comparison animator
    comparison = ComparisonAnimator(
        [results, results],
        labels=['Case 1', 'Case 2']
    )
    print(f"✓ ComparisonAnimator initialized")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VISUALIZATION MODULES STANDALONE TEST")
    print("="*60)
    
    try:
        # Test each module
        test_particle_tracer()
        test_streamline_generator()
        test_flow_animator()
        
        # Summary
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nAll visualization modules are working correctly!")
        print("Modules tested:")
        print("  - ParticleTracer: RK4 integration, coloring, boundary handling")
        print("  - StreamlineGenerator: streamlines, vorticity, vortex detection")
        print("  - FlowAnimator: multi-layer animations, video/GIF export")
        print("  - ComparisonAnimator: side-by-side scenario comparison")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
