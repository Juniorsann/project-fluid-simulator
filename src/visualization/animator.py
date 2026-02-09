"""
Animation module for creating animations of flow simulations.

This module provides utilities for creating animations of transient simulations.
"""

from typing import Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.figure import Figure


class FlowAnimator:
    """Animator for flow field visualizations."""

    def __init__(
        self,
        figure_size: tuple = (10, 6),
        fps: int = 30
    ):
        """
        Initialize flow animator.

        Args:
            figure_size: Figure size (width, height) in inches
            fps: Frames per second for animation
        """
        self.figure_size = figure_size
        self.fps = fps
        self.fig = None
        self.ax = None

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
