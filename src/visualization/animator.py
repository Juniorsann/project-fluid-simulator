"""
Animation module for creating animations of flow simulations.

This module provides utilities for creating animations of transient simulations
and advanced visualization of CFD results.
"""

from typing import Optional, Callable, Tuple, Literal, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.figure import Figure

from .particle_tracer import ParticleTracer


class FlowAnimator:
    """Animator for flow field visualizations with advanced CFD capabilities."""

    def __init__(
        self,
        simulation_results = None,
        figure_size: tuple = (10, 6),
        fps: int = 30
    ):
        """
        Initialize flow animator.

        Args:
            simulation_results: Optional simulation results object
            figure_size: Figure size (width, height) in inches
            fps: Frames per second for animation
        """
        self.simulation_results = simulation_results
        self.figure_size = figure_size
        self.fps = fps
        self.fig = None
        self.ax = None
        self.current_animation = None

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
    
    def animate_particle_tracers(
        self,
        velocity_field: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        grid_coordinates: Tuple[np.ndarray, np.ndarray],
        n_particles: int = 500,
        duration: float = 10,
        fps: int = 30,
        colorby: Optional[np.ndarray] = None,
        cmap: str = 'viridis',
        particle_size: float = 2,
        show_velocity_field: bool = False,
        show_streamlines: bool = False,
        title: str = "Particle Tracers"
    ) -> FuncAnimation:
        """
        Create animation of particle tracers following velocity field.
        
        Args:
            velocity_field: Velocity field as (u, v) tuple or single array
            grid_coordinates: Tuple of (z, r) coordinate arrays
            n_particles: Number of tracer particles
            duration: Animation duration in seconds
            fps: Frames per second
            colorby: Optional field for coloring particles (e.g., temperature)
            cmap: Colormap for particles
            particle_size: Size of particles in plot
            show_velocity_field: Whether to show velocity field as background
            show_streamlines: Whether to show streamlines
            title: Animation title
        
        Returns:
            Animation object
        """
        # Create particle tracer
        tracer = ParticleTracer(velocity_field, grid_coordinates, n_particles)
        tracer.inject_particles(region='inlet', distribution='random')
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        
        grid_z, grid_r = grid_coordinates
        
        # Handle velocity field
        if isinstance(velocity_field, tuple):
            u_field = velocity_field[0]
        else:
            u_field = velocity_field
        
        # Show velocity field as background if requested
        if show_velocity_field:
            R, Z = np.meshgrid(grid_r, grid_z, indexing='ij')
            contour = self.ax.contourf(Z * 1000, R * 1000, u_field,
                                      levels=15, cmap='gray', alpha=0.3)
        
        # Initialize particle scatter plot
        scatter = self.ax.scatter([], [], s=particle_size, c=[], 
                                 cmap=cmap, alpha=0.7)
        
        self.ax.set_xlim(grid_z.min() * 1000, grid_z.max() * 1000)
        self.ax.set_ylim(grid_r.min() * 1000, grid_r.max() * 1000)
        self.ax.set_xlabel('Axial Position [mm]', fontsize=12)
        self.ax.set_ylabel('Radial Position [mm]', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_aspect('equal')
        
        # Time step
        dt = 1.0 / fps
        n_frames = int(duration * fps)
        
        # Particle injection interval (inject new particles every N frames)
        inject_interval = max(1, fps // 10)  # Inject 10 times per second
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            return scatter,
        
        def animate(frame):
            # Advect particles
            tracer.advect(dt)
            
            # Periodically inject new particles
            if frame % inject_interval == 0:
                # Re-inject particles at inlet to maintain count
                n_inactive = tracer.n_particles - np.sum(tracer.active)
                if n_inactive > 0:
                    # Inject at inlet
                    z_new = np.full(n_inactive, grid_z.min())
                    r_new = np.random.uniform(grid_r.min(), grid_r.max(), n_inactive)
                    new_positions = np.column_stack([z_new, r_new])
                    
                    # Update inactive particles
                    inactive_indices = np.where(~tracer.active)[0][:n_inactive]
                    tracer.positions[inactive_indices] = new_positions
                    tracer.active[inactive_indices] = True
            
            # Get active particle positions
            positions = tracer.get_active_positions()
            
            if len(positions) > 0:
                # Convert to mm for display
                z_mm = positions[:, 0] * 1000
                r_mm = positions[:, 1] * 1000
                
                scatter.set_offsets(np.column_stack([z_mm, r_mm]))
                
                # Color particles if colorby field provided
                if colorby is not None:
                    colors = tracer.get_particle_properties(colorby)
                    scatter.set_array(colors)
                else:
                    scatter.set_array(np.ones(len(positions)))
            
            return scatter,
        
        anim = FuncAnimation(
            self.fig, animate, init_func=init,
            frames=n_frames, interval=1000/fps, blit=True
        )
        
        self.current_animation = anim
        return anim
    
    def animate_velocity_field(
        self,
        velocity_field: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        grid_coordinates: Tuple[np.ndarray, np.ndarray],
        style: Literal['quiver', 'streamplot', 'contour'] = 'quiver',
        interval: int = 50,
        show_magnitude: bool = True,
        title: str = "Velocity Field"
    ) -> FuncAnimation:
        """
        Animate velocity field visualization.
        
        Args:
            velocity_field: Velocity as (u, v) tuple or single array
            grid_coordinates: Tuple of (z, r) coordinate arrays
            style: Visualization style
            interval: Frame interval in milliseconds
            show_magnitude: Whether to show velocity magnitude
            title: Animation title
        
        Returns:
            Animation object
        """
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        
        grid_z, grid_r = grid_coordinates
        
        if isinstance(velocity_field, tuple):
            u_field, v_field = velocity_field
        else:
            u_field = velocity_field
            v_field = np.zeros_like(u_field)
        
        R, Z = np.meshgrid(grid_r, grid_z, indexing='ij')
        
        if style == 'quiver':
            # Downsample for quiver plot
            skip = max(1, len(grid_z) // 20)
            quiver = self.ax.quiver(Z[::skip, ::skip] * 1000, R[::skip, ::skip] * 1000,
                                   u_field[::skip, ::skip], v_field[::skip, ::skip],
                                   scale=50, width=0.003)
            
            if show_magnitude:
                magnitude = np.sqrt(u_field**2 + v_field**2)
                contour = self.ax.contourf(Z * 1000, R * 1000, magnitude,
                                          levels=15, cmap='viridis', alpha=0.5)
                plt.colorbar(contour, ax=self.ax, label='Velocity Magnitude [m/s]')
        
        elif style == 'streamplot':
            # Transpose for streamplot
            self.ax.streamplot(Z.T * 1000, R.T * 1000, u_field.T, v_field.T,
                              density=1.5, color='k', linewidth=1)
            
            if show_magnitude:
                magnitude = np.sqrt(u_field**2 + v_field**2)
                contour = self.ax.contourf(Z * 1000, R * 1000, magnitude,
                                          levels=15, cmap='viridis', alpha=0.7)
                plt.colorbar(contour, ax=self.ax, label='Velocity Magnitude [m/s]')
        
        elif style == 'contour':
            magnitude = np.sqrt(u_field**2 + v_field**2)
            contour = self.ax.contourf(Z * 1000, R * 1000, magnitude,
                                      levels=20, cmap='viridis')
            plt.colorbar(contour, ax=self.ax, label='Velocity Magnitude [m/s]')
        
        self.ax.set_xlabel('Axial Position [mm]', fontsize=12)
        self.ax.set_ylabel('Radial Position [mm]', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_aspect('equal')
        
        # Create simple animation (static for now, can be enhanced)
        def animate(frame):
            return []
        
        anim = FuncAnimation(self.fig, animate, frames=1, interval=interval)
        self.current_animation = anim
        return anim
    
    def animate_temperature_evolution(
        self,
        temperature_field: np.ndarray,
        grid_coordinates: Tuple[np.ndarray, np.ndarray],
        time_steps: Optional[np.ndarray] = None,
        cmap: str = 'hot',
        show_isotherms: bool = True,
        isotherm_levels: int = 10,
        title: str = "Temperature Evolution"
    ) -> FuncAnimation:
        """
        Animate temperature field evolution.
        
        Args:
            temperature_field: Temperature field (single frame or time series)
            grid_coordinates: Tuple of (z, r) coordinate arrays
            time_steps: Optional time values for each frame
            cmap: Colormap
            show_isotherms: Whether to show isotherm contours
            isotherm_levels: Number of isotherm levels
            title: Animation title
        
        Returns:
            Animation object
        """
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        
        grid_z, grid_r = grid_coordinates
        R, Z = np.meshgrid(grid_r, grid_z, indexing='ij')
        
        # Handle single frame or time series
        if temperature_field.ndim == 2:
            # Single frame - create static visualization
            T_data = [temperature_field]
        else:
            # Time series
            T_data = temperature_field
        
        vmin, vmax = temperature_field.min(), temperature_field.max()
        
        def animate(frame):
            self.ax.clear()
            
            T = T_data[min(frame, len(T_data)-1)]
            
            contour = self.ax.contourf(Z * 1000, R * 1000, T,
                                      levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if show_isotherms:
                self.ax.contour(Z * 1000, R * 1000, T,
                               levels=isotherm_levels, colors='k', alpha=0.3, linewidths=0.5)
            
            self.ax.set_xlabel('Axial Position [mm]', fontsize=12)
            self.ax.set_ylabel('Radial Position [mm]', fontsize=12)
            self.ax.set_title(title, fontsize=14, fontweight='bold')
            self.ax.set_aspect('equal')
            
            if frame == 0:
                plt.colorbar(contour, ax=self.ax, label='Temperature [°C]')
            
            return contour,
        
        anim = FuncAnimation(
            self.fig, animate,
            frames=len(T_data), interval=1000/self.fps, blit=False
        )
        
        self.current_animation = anim
        return anim
    
    def animate_pressure_contours(
        self,
        pressure_field: np.ndarray,
        grid_coordinates: Tuple[np.ndarray, np.ndarray],
        levels: int = 20,
        title: str = "Pressure Contours"
    ) -> FuncAnimation:
        """
        Animate pressure contours.
        
        Args:
            pressure_field: Pressure field
            grid_coordinates: Tuple of (z, r) coordinate arrays
            levels: Number of contour levels
            title: Animation title
        
        Returns:
            Animation object
        """
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        
        grid_z, grid_r = grid_coordinates
        R, Z = np.meshgrid(grid_r, grid_z, indexing='ij')
        
        # Convert to bar for display
        P_bar = pressure_field / 1e5
        
        contour = self.ax.contourf(Z * 1000, R * 1000, P_bar,
                                  levels=levels, cmap='RdYlBu_r')
        plt.colorbar(contour, ax=self.ax, label='Pressure [bar]')
        
        self.ax.set_xlabel('Axial Position [mm]', fontsize=12)
        self.ax.set_ylabel('Radial Position [mm]', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_aspect('equal')
        
        def animate(frame):
            return []
        
        anim = FuncAnimation(self.fig, animate, frames=1, interval=100)
        self.current_animation = anim
        return anim
    
    def create_multi_view_animation(
        self,
        views_data: dict,
        duration: float = 10,
        fps: int = 30,
        title: str = "Multi-View Animation"
    ) -> FuncAnimation:
        """
        Create multi-view animation with multiple subplots.
        
        Args:
            views_data: Dictionary with view configurations
            duration: Animation duration in seconds
            fps: Frames per second
            title: Overall title
        
        Returns:
            Animation object
        """
        n_views = len(views_data)
        
        # Create subplots
        if n_views <= 2:
            nrows, ncols = 1, n_views
        elif n_views <= 4:
            nrows, ncols = 2, 2
        else:
            nrows = int(np.ceil(np.sqrt(n_views)))
            ncols = int(np.ceil(n_views / nrows))
        
        self.fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*5))
        
        if n_views == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        self.fig.suptitle(title, fontsize=16, fontweight='bold')
        
        n_frames = int(duration * fps)
        
        def animate(frame):
            # Simple placeholder animation
            return []
        
        anim = FuncAnimation(
            self.fig, animate,
            frames=n_frames, interval=1000/fps, blit=False
        )
        
        self.current_animation = anim
        return anim
    
    def save(
        self,
        filename: str,
        codec: str = 'h264',
        bitrate: int = 5000,
        dpi: int = 150
    ):
        """
        Save current animation to file.
        
        Args:
            filename: Output filename (.mp4, .gif, .avi)
            codec: Video codec for mp4/avi
            bitrate: Bitrate for video encoding
            dpi: DPI for output
        """
        if self.current_animation is None:
            raise ValueError("No animation to save. Create an animation first.")
        
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext in ['mp4', 'avi']:
            writer = FFMpegWriter(fps=self.fps, codec=codec, bitrate=bitrate)
            self.current_animation.save(filename, writer=writer, dpi=dpi)
        elif file_ext == 'gif':
            writer = PillowWriter(fps=self.fps)
            self.current_animation.save(filename, writer=writer, dpi=dpi)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        print(f"Animation saved to: {filename}")


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
