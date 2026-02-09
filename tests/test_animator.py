"""
Tests for animator module.

Tests animation creation, video/GIF generation, and frame export.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import tempfile
import shutil
from src.visualization.animator import FlowAnimator, ComparisonAnimator


@pytest.fixture
def sample_results():
    """Create sample simulation results for testing."""
    r = np.linspace(0, 0.1, 20)
    z = np.linspace(0, 1.0, 100)
    
    # Simple velocity field
    R, Z = np.meshgrid(r, z, indexing='ij')
    u_r = np.zeros_like(R)
    u_z = np.ones_like(R) * (1 - (R / 0.1)**2)  # Parabolic
    
    # Temperature field
    T = 300 + 20 * Z
    
    return {
        'velocity_field': (u_r, u_z),
        'temperature_field': T,
        'grid': (r, z),
        'pipe_radius': 0.1,
        'pipe_length': 1.0
    }


def test_animator_initialization(sample_results):
    """Test animator initialization."""
    animator = FlowAnimator(sample_results)
    
    assert animator.results is not None
    assert animator.fps == 30
    assert len(animator.layers) > 0


def test_add_particle_tracers(sample_results):
    """Test adding particle tracer layer."""
    animator = FlowAnimator(sample_results)
    
    animator.add_particle_tracers(
        n_particles=100,
        release_mode='continuous',
        color_scheme='velocity'
    )
    
    assert animator.particle_settings is not None
    assert animator.particle_settings['n_particles'] == 100
    assert animator.layers['particles'] is True


def test_add_velocity_field(sample_results):
    """Test adding velocity field layer."""
    animator = FlowAnimator(sample_results)
    
    animator.add_velocity_field(style='arrows', density=20, scale=1.0)
    
    assert animator.layers['velocity_field'] is not None
    assert animator.layers['velocity_field']['style'] == 'arrows'
    assert animator.layers['velocity_field']['density'] == 20


def test_add_temperature_map(sample_results):
    """Test adding temperature map layer."""
    animator = FlowAnimator(sample_results)
    
    animator.add_temperature_map(cmap='hot', levels=20, alpha=0.7)
    
    assert animator.layers['temperature_map'] is not None
    assert animator.layers['temperature_map']['cmap'] == 'hot'
    assert animator.layers['temperature_map']['levels'] == 20


def test_add_streamlines(sample_results):
    """Test adding streamlines layer."""
    animator = FlowAnimator(sample_results)
    
    animator.add_streamlines(n_lines=30, integration_steps=100)
    
    assert animator.layers['streamlines'] is not None
    assert animator.layers['streamlines']['n_lines'] == 30


def test_add_title(sample_results):
    """Test adding title."""
    animator = FlowAnimator(sample_results)
    
    title = "Test Animation"
    animator.add_title(title)
    
    assert animator.title == title


def test_add_text(sample_results):
    """Test adding text annotation."""
    animator = FlowAnimator(sample_results)
    
    animator.add_text(0.02, 0.95, "Test Text", fontsize=12)
    
    assert len(animator.texts) == 1
    assert animator.texts[0]['text'] == "Test Text"


def test_add_colorbar(sample_results):
    """Test adding colorbar."""
    animator = FlowAnimator(sample_results)
    
    animator.add_colorbar(label="Velocity [m/s]")
    
    assert animator.colorbar_label == "Velocity [m/s]"


def test_animate_creation(sample_results):
    """Test animation object creation."""
    animator = FlowAnimator(sample_results)
    animator.add_title("Test Animation")
    
    # Create short animation
    anim = animator.animate(duration=0.1, fps=10)
    
    assert anim is not None


def test_frame_export(sample_results):
    """Test individual frame export."""
    animator = FlowAnimator(sample_results)
    animator.add_title("Frame Export Test")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save a few frames
        animator.save_frames(
            directory=temp_dir,
            prefix='test_',
            format='png',
            duration=0.1,
            fps=10
        )
        
        # Check files were created
        files = os.listdir(temp_dir)
        png_files = [f for f in files if f.endswith('.png')]
        
        assert len(png_files) > 0
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_1d_profile_animation():
    """Test 1D profile animation."""
    # Create time-varying 1D data
    positions = np.linspace(0, 1, 50)
    time_steps = np.linspace(0, 1, 10)
    
    # Create field data (temperature varying in time and space)
    field_data = np.zeros((len(time_steps), len(positions)))
    for i, t in enumerate(time_steps):
        field_data[i] = 300 + 20 * positions + 10 * np.sin(2 * np.pi * t)
    
    animator = FlowAnimator()
    
    anim = animator.animate_1d_profile(
        time_steps=time_steps,
        positions=positions,
        field_data=field_data,
        xlabel="Position [m]",
        ylabel="Temperature [K]",
        title="Temperature Evolution"
    )
    
    assert anim is not None


def test_2d_field_animation():
    """Test 2D field animation."""
    # Create time-varying 2D data
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 0.5, 20)
    time_steps = np.linspace(0, 1, 5)
    
    # Create field data
    X, Y = np.meshgrid(x, y)
    field_data = np.zeros((len(time_steps), len(y), len(x)))
    for i, t in enumerate(time_steps):
        field_data[i] = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * (1 + t)
    
    animator = FlowAnimator()
    
    anim = animator.animate_2d_field(
        time_steps=time_steps,
        x=x,
        y=y,
        field_data=field_data,
        cmap='viridis',
        title="2D Field Evolution"
    )
    
    assert anim is not None


def test_comparison_animator():
    """Test comparison animator initialization."""
    # Create two dummy result sets
    results1 = {'data': 'scenario1'}
    results2 = {'data': 'scenario2'}
    
    comparison = ComparisonAnimator(
        [results1, results2],
        labels=['Scenario 1', 'Scenario 2']
    )
    
    assert len(comparison.results_list) == 2
    assert len(comparison.labels) == 2


def test_comparison_add_layers():
    """Test adding layers to comparison animator."""
    results1 = {'data': 'scenario1'}
    results2 = {'data': 'scenario2'}
    
    comparison = ComparisonAnimator(
        [results1, results2],
        labels=['Scenario 1', 'Scenario 2']
    )
    
    comparison.add_particle_tracers(n_particles=150)
    comparison.add_temperature_map(cmap='hot')
    
    assert comparison.particle_settings is not None
    assert comparison.temperature_map_settings is not None


# Skip video/GIF tests if FFmpeg/Pillow not available
@pytest.mark.skipif(
    not shutil.which('ffmpeg'),
    reason="FFmpeg not installed"
)
def test_video_generation(sample_results):
    """Test MP4 video generation."""
    animator = FlowAnimator(sample_results)
    animator.add_title("Video Test")
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test.mp4")
    
    try:
        # Generate very short video
        animator.save_video(temp_file, duration=0.1, fps=10, dpi=50)
        
        # Check file was created
        assert os.path.exists(temp_file)
        assert os.path.getsize(temp_file) > 0
        
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.skipif(
    True,  # Always skip for now as it may fail in CI
    reason="Pillow/imageio may not be configured"
)
def test_gif_generation(sample_results):
    """Test GIF generation."""
    animator = FlowAnimator(sample_results)
    animator.add_title("GIF Test")
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test.gif")
    
    try:
        # Generate very short GIF
        animator.save_gif(temp_file, duration=0.1, fps=5)
        
        # Check file was created
        assert os.path.exists(temp_file)
        assert os.path.getsize(temp_file) > 0
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
