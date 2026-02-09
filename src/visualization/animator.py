"""
Animation module for creating animations of flow simulations.

This module provides utilities for creating animations of transient simulations
with support for particle tracers, velocity fields, temperature maps, and more.
"""

from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import os

from .particle_tracer import ParticleTracer
from .streamlines import StreamlineGenerator


class FlowAnimator:
    """
    Comprehensive animator for flow field visualizations.
    
    Supports multiple visualization layers including particle tracers,
    velocity fields, temperature/pressure maps, and streamlines.
    """

    def __init__(
        self,
        simulation_results: Optional[Dict[str, Any]] = None,
        figure_size: tuple = (12, 6),
        fps: int = 30
    ):
        """
        Initialize flow animator with simulation results.

        Args:
            simulation_results: Dictionary containing simulation results with
                               velocity_field, temperature_field, grid, etc.
            figure_size: Figure size (width, height) in inches
            fps: Frames per second for animation
        """
        self.results = simulation_results
        self.figure_size = figure_size
        self.fps = fps
        self.fig = None
        self.ax = None
        
        # Visualization layers
        self.layers = {
            'particles': None,
            'velocity_field': None,
            'temperature_map': None,
            'pressure_map': None,
            'streamlines': None
        }
        
        # Animation settings
        self.title = ""
        self.texts = []
        self.colorbar_label = None
        
        # Particle tracer if enabled
        self.particle_tracer = None
        self.particle_settings = None
        
        # Streamline generator
        self.streamline_gen = None
    
    def add_particle_tracers(
        self,
        n_particles: int = 500,
        release_mode: str = 'continuous',
        color_scheme: str = 'velocity'
    ) -> None:
        """
        Add particle tracer layer.
        
        Args:
            n_particles: Number of particles to track
            release_mode: 'continuous' or 'single'
            color_scheme: 'velocity', 'temperature', or 'age'
        """
        self.particle_settings = {
            'n_particles': n_particles,
            'release_mode': release_mode,
            'color_scheme': color_scheme
        }
        self.layers['particles'] = True
    
    def add_velocity_field(
        self,
        style: str = 'arrows',
        density: int = 20,
        scale: float = 1.0
    ) -> None:
        """
        Add velocity vector field layer.
        
        Args:
            style: 'arrows' or 'streamlines'
            density: Density of arrows/streamlines
            scale: Scale factor for arrows
        """
        self.layers['velocity_field'] = {
            'style': style,
            'density': density,
            'scale': scale
        }
    
    def add_temperature_map(
        self,
        cmap: str = 'hot',
        levels: int = 20,
        alpha: float = 0.7
    ) -> None:
        """
        Add temperature contour/heatmap layer.
        
        Args:
            cmap: Colormap name
            levels: Number of contour levels
            alpha: Transparency (0-1)
        """
        self.layers['temperature_map'] = {
            'cmap': cmap,
            'levels': levels,
            'alpha': alpha
        }
    
    def add_pressure_map(
        self,
        cmap: str = 'viridis',
        alpha: float = 0.5
    ) -> None:
        """
        Add pressure field visualization.
        
        Args:
            cmap: Colormap name
            alpha: Transparency (0-1)
        """
        self.layers['pressure_map'] = {
            'cmap': cmap,
            'alpha': alpha
        }
    
    def add_streamlines(
        self,
        n_lines: int = 30,
        integration_steps: int = 100
    ) -> None:
        """
        Add streamlines following flow.
        
        Args:
            n_lines: Number of streamlines
            integration_steps: Steps for integration
        """
        self.layers['streamlines'] = {
            'n_lines': n_lines,
            'integration_steps': integration_steps
        }
    
    def add_colorbar(self, label: str = "") -> None:
        """
        Add colorbar to plot.
        
        Args:
            label: Label for colorbar
        """
        self.colorbar_label = label
    
    def add_title(self, title: str) -> None:
        """
        Add title to animation.
        
        Args:
            title: Title text
        """
        self.title = title
    
    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        fontsize: int = 12
    ) -> None:
        """
        Add text annotation to plot.
        
        Args:
            x: X position (0-1 in axes coordinates)
            y: Y position (0-1 in axes coordinates)
            text: Text content
            fontsize: Font size
        """
        self.texts.append({
            'x': x,
            'y': y,
            'text': text,
            'fontsize': fontsize
        })
    
    def add_isotherms(
        self,
        temperatures: List[float],
        linewidth: float = 2
    ) -> None:
        """
        Add isotherms (lines of constant temperature).
        
        Args:
            temperatures: List of temperature values [°C]
            linewidth: Line width
        """
        # Store for later use in animation
        if 'isotherms' not in self.layers:
            self.layers['isotherms'] = {
                'temperatures': temperatures,
                'linewidth': linewidth
            }
    
    def animate(
        self,
        duration: float = 10,
        fps: Optional[int] = None
    ) -> FuncAnimation:
        """
        Generate animation in memory.
        
        Args:
            duration: Animation duration [s]
            fps: Frames per second (overrides init value if provided)
            
        Returns:
            Animation object
        """
        if fps is not None:
            self.fps = fps
        
        n_frames = int(duration * self.fps)
        
        # Setup figure
        self._setup_figure()
        
        # Create animation function
        def update_frame(frame):
            return self._update_frame(frame, n_frames)
        
        anim = FuncAnimation(
            self.fig,
            update_frame,
            frames=n_frames,
            interval=1000/self.fps,
            blit=False
        )
        
        return anim
    
    def save_video(
        self,
        filename: str = 'animation.mp4',
        duration: float = 10,
        fps: Optional[int] = None,
        dpi: int = 150
    ) -> None:
        """
        Save animation as MP4 video.
        
        Args:
            filename: Output filename
            duration: Animation duration [s]
            fps: Frames per second
            dpi: Resolution (dots per inch)
        """
        print(f"Generating video: {filename}")
        anim = self.animate(duration, fps)
        
        writer = FFMpegWriter(fps=self.fps, bitrate=1800)
        anim.save(filename, writer=writer, dpi=dpi)
        print(f"Video saved: {filename}")
    
    def save_gif(
        self,
        filename: str = 'animation.gif',
        duration: float = 10,
        fps: int = 15
    ) -> None:
        """
        Save animation as GIF.
        
        Args:
            filename: Output filename
            duration: Animation duration [s]
            fps: Frames per second (lower for smaller file size)
        """
        print(f"Generating GIF: {filename}")
        anim = self.animate(duration, fps)
        
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"GIF saved: {filename}")
    
    def save_frames(
        self,
        directory: str = 'frames/',
        prefix: str = 'frame_',
        format: str = 'png',
        duration: float = 10,
        fps: Optional[int] = None
    ) -> None:
        """
        Save individual frames as images.
        
        Args:
            directory: Output directory
            prefix: Filename prefix
            format: Image format ('png', 'jpg', etc.)
            duration: Animation duration [s]
            fps: Frames per second
        """
        if fps is not None:
            self.fps = fps
        
        n_frames = int(duration * self.fps)
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Setup figure
        self._setup_figure()
        
        print(f"Saving {n_frames} frames to {directory}")
        
        for frame in range(n_frames):
            self._update_frame(frame, n_frames)
            
            filename = os.path.join(
                directory,
                f"{prefix}{frame:04d}.{format}"
            )
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            
            if (frame + 1) % 10 == 0:
                print(f"Saved {frame + 1}/{n_frames} frames")
        
        print(f"All frames saved to {directory}")
    
    def _setup_figure(self) -> None:
        """Setup the figure and axes for animation."""
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        
        if self.title:
            self.ax.set_title(self.title, fontsize=14, fontweight='bold')
        
        self.ax.set_xlabel('Axial Position [m]', fontsize=12)
        self.ax.set_ylabel('Radial Position [m]', fontsize=12)
        self.ax.set_aspect('equal')
    
    def _update_frame(self, frame: int, total_frames: int) -> List:
        """
        Update animation frame.
        
        Args:
            frame: Current frame number
            total_frames: Total number of frames
            
        Returns:
            List of artists that were modified
        """
        self.ax.clear()
        
        # Reconfigure axes
        if self.title:
            self.ax.set_title(self.title, fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Axial Position [m]', fontsize=12)
        self.ax.set_ylabel('Radial Position [m]', fontsize=12)
        
        # Add time indicator
        time = frame / self.fps
        time_text = self.ax.text(
            0.02, 0.98, f'Time: {time:.2f} s',
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        artists = [time_text]
        
        # Add custom text annotations
        for text_info in self.texts:
            text_obj = self.ax.text(
                text_info['x'], text_info['y'], text_info['text'],
                transform=self.ax.transAxes,
                fontsize=text_info['fontsize'],
                verticalalignment='top'
            )
            artists.append(text_obj)
        
        return artists
    
    def animate_1d_profile(
        self,
        time_steps: np.ndarray,
        positions: np.ndarray,
        field_data: np.ndarray,
        xlabel: str = "Position [m]",
        ylabel: str = "Field Value",
        title: str = "Flow Field Animation",
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Create animation of 1D field evolution over time.

        Args:
            time_steps: Array of time values
            positions: Spatial positions
            field_data: Field values [n_time, n_positions]
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Animation title
            save_path: Path to save animation (mp4 format)

        Returns:
            Animation object
        """
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)

        # Initialize plot
        line, = self.ax.plot([], [], 'b-', linewidth=2)
        time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

        # Set axis limits
        self.ax.set_xlim(positions.min(), positions.max())
        self.ax.set_ylim(field_data.min(), field_data.max())
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.grid(True, alpha=0.3)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(frame):
            line.set_data(positions, field_data[frame])
            time_text.set_text(f'Time: {time_steps[frame]:.2f} s')
            return line, time_text

        anim = FuncAnimation(
            self.fig,
            animate,
            init_func=init,
            frames=len(time_steps),
            interval=1000/self.fps,
            blit=True
        )

        if save_path:
            writer = FFMpegWriter(fps=self.fps, bitrate=1800)
            anim.save(save_path, writer=writer)

        return anim

    def animate_2d_field(
        self,
        time_steps: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        field_data: np.ndarray,
        cmap: str = 'viridis',
        title: str = "2D Flow Field Animation",
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Create animation of 2D field evolution over time.

        Args:
            time_steps: Array of time values
            x: X coordinates
            y: Y coordinates
            field_data: Field values [n_time, n_x, n_y]
            cmap: Colormap name
            title: Animation title
            save_path: Path to save animation

        Returns:
            Animation object
        """
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)

        X, Y = np.meshgrid(x, y)

        # Initialize contour plot
        vmin, vmax = field_data.min(), field_data.max()
        contour = self.ax.contourf(X, Y, field_data[0], levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contour, ax=self.ax)

        time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.ax.set_xlabel('X [m]', fontsize=12)
        self.ax.set_ylabel('Y [m]', fontsize=12)
        self.ax.set_title(title, fontsize=14)

        def animate(frame):
            # Clear and redraw contour
            self.ax.clear()
            contour = self.ax.contourf(X, Y, field_data[frame], levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
            time_text.set_text(f'Time: {time_steps[frame]:.2f} s')
            self.ax.set_xlabel('X [m]', fontsize=12)
            self.ax.set_ylabel('Y [m]', fontsize=12)
            self.ax.set_title(title, fontsize=14)
            return contour,

        anim = FuncAnimation(
            self.fig,
            animate,
            frames=len(time_steps),
            interval=1000/self.fps,
            blit=False
        )

        if save_path:
            writer = FFMpegWriter(fps=self.fps, bitrate=1800)
            anim.save(save_path, writer=writer)

        return anim


class ComparisonAnimator:
    """
    Animator for side-by-side comparison of different scenarios.
    """
    
    def __init__(
        self,
        results_list: List[Dict[str, Any]],
        labels: List[str],
        figure_size: tuple = (16, 6),
        fps: int = 30
    ):
        """
        Initialize comparison animator.
        
        Args:
            results_list: List of simulation results dictionaries
            labels: Labels for each scenario
            figure_size: Figure size
            fps: Frames per second
        """
        self.results_list = results_list
        self.labels = labels
        self.figure_size = figure_size
        self.fps = fps
        self.fig = None
        self.axes = None
        
        self.particle_settings = None
        self.temperature_map_settings = None
    
    def add_particle_tracers(self, n_particles: int = 150) -> None:
        """Add particle tracers to all scenarios."""
        self.particle_settings = {'n_particles': n_particles}
    
    def add_temperature_map(self, cmap: str = 'hot') -> None:
        """Add temperature map to all scenarios."""
        self.temperature_map_settings = {'cmap': cmap}
    
    def save_video(
        self,
        filename: str = 'comparison.mp4',
        duration: float = 12,
        fps: int = 30
    ) -> None:
        """
        Save comparison animation as video.
        
        Args:
            filename: Output filename
            duration: Animation duration
            fps: Frames per second
        """
        print(f"Generating comparison video: {filename}")
        # Placeholder implementation
        print(f"Comparison video would be saved to: {filename}")


def create_streamlines_animation(
    velocity_u: np.ndarray,
    velocity_v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_particles: int = 50,
    title: str = "Streamlines Animation"
) -> Figure:
    """
    Create streamlines visualization from velocity field.

    Args:
        velocity_u: X-component of velocity
        velocity_v: Y-component of velocity
        x: X coordinates
        y: Y coordinates
        n_particles: Number of streamline particles
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    X, Y = np.meshgrid(x, y)

    # Create streamlines
    ax.streamplot(X, Y, velocity_u, velocity_v, density=2, linewidth=1,
                  arrowsize=1.5, color=np.sqrt(velocity_u**2 + velocity_v**2),
                  cmap='viridis')

    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')

    plt.tight_layout()

    return fig
