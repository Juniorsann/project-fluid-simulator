"""
Streamline Generator module for visualizing flow patterns.

This module generates and visualizes streamlines (lines tangent to velocity field)
using Runge-Kutta 4th order integration.
"""

from typing import Optional, List, Tuple, Literal, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RegularGridInterpolator


class StreamlineGenerator:
    """
    Generates streamlines (lines tangent to the velocity field at every point).
    """
    
    def __init__(
        self,
        velocity_field: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        grid_coordinates: Tuple[np.ndarray, np.ndarray]
    ):
        """
        Initialize streamline generator.
        
        Args:
            velocity_field: Velocity field as tuple (u, v) or single array u
            grid_coordinates: Tuple of (z, r) coordinate arrays
        """
        self.grid_z, self.grid_r = grid_coordinates
        
        # Handle velocity field format
        if isinstance(velocity_field, tuple):
            self.u_field, self.v_field = velocity_field
        else:
            self.u_field = velocity_field
            self.v_field = np.zeros_like(velocity_field)
        
        self.streamlines = []
        
        # Create interpolators
        self._setup_interpolators()
    
    def _setup_interpolators(self):
        """Setup velocity field interpolators."""
        self.u_interpolator = RegularGridInterpolator(
            (self.grid_r, self.grid_z),
            self.u_field,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        self.v_interpolator = RegularGridInterpolator(
            (self.grid_r, self.grid_z),
            self.v_field,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
    
    def _get_velocity(self, position: np.ndarray) -> np.ndarray:
        """
        Get velocity at a position.
        
        Args:
            position: Position as (z, r)
        
        Returns:
            Velocity as (u, v)
        """
        # Convert to (r, z) for interpolator
        point = np.array([[position[1], position[0]]])
        u = self.u_interpolator(point)[0]
        v = self.v_interpolator(point)[0]
        return np.array([u, v])
    
    def _integrate_streamline(
        self,
        seed_point: np.ndarray,
        direction: Literal['forward', 'backward'],
        max_length: int,
        step_size: float
    ) -> np.ndarray:
        """
        Integrate a single streamline from a seed point.
        
        Args:
            seed_point: Starting point (z, r)
            direction: Integration direction
            max_length: Maximum number of integration steps
            step_size: Step size for integration
        
        Returns:
            Array of streamline points
        """
        points = [seed_point.copy()]
        current_pos = seed_point.copy()
        
        sign = 1.0 if direction == 'forward' else -1.0
        
        z_min, z_max = self.grid_z.min(), self.grid_z.max()
        r_min, r_max = self.grid_r.min(), self.grid_r.max()
        
        for _ in range(max_length):
            # RK4 integration
            vel1 = self._get_velocity(current_pos)
            k1 = sign * vel1
            
            pos2 = current_pos + 0.5 * step_size * k1
            vel2 = self._get_velocity(pos2)
            k2 = sign * vel2
            
            pos3 = current_pos + 0.5 * step_size * k2
            vel3 = self._get_velocity(pos3)
            k3 = sign * vel3
            
            pos4 = current_pos + step_size * k3
            vel4 = self._get_velocity(pos4)
            k4 = sign * vel4
            
            # Update position
            current_pos = current_pos + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Check if out of bounds
            if (current_pos[0] < z_min or current_pos[0] > z_max or
                current_pos[1] < r_min or current_pos[1] > r_max):
                break
            
            # Check if velocity is too small (stagnation point)
            vel_mag = np.linalg.norm(self._get_velocity(current_pos))
            if vel_mag < 1e-10:
                break
            
            points.append(current_pos.copy())
        
        return np.array(points)
    
    def generate_streamlines(
        self,
        seed_points: Union[Literal['auto'], np.ndarray] = 'auto',
        integration_direction: Literal['forward', 'backward', 'both'] = 'both',
        max_length: int = 1000,
        step_size: float = 0.01,
        n_streamlines: int = 20
    ) -> List[np.ndarray]:
        """
        Generate streamlines using RK4 integration.
        
        Args:
            seed_points: Seed points for streamlines or 'auto' for automatic
            integration_direction: Direction to integrate
            max_length: Maximum number of steps per streamline
            step_size: Integration step size
            n_streamlines: Number of streamlines (if seed_points='auto')
        
        Returns:
            List of streamline arrays, each array is (n_points, 2)
        """
        # Generate seed points if auto
        if isinstance(seed_points, str) and seed_points == 'auto':
            # Distribute seed points at inlet
            z_start = self.grid_z.min()
            r_seeds = np.linspace(self.grid_r.min(), self.grid_r.max(), n_streamlines)
            seed_points = np.column_stack([
                np.full(n_streamlines, z_start),
                r_seeds
            ])
        
        self.streamlines = []
        
        for seed in seed_points:
            if integration_direction == 'forward':
                streamline = self._integrate_streamline(seed, 'forward', max_length, step_size)
                self.streamlines.append(streamline)
            
            elif integration_direction == 'backward':
                streamline = self._integrate_streamline(seed, 'backward', max_length, step_size)
                self.streamlines.append(streamline)
            
            elif integration_direction == 'both':
                # Integrate both directions and combine
                forward = self._integrate_streamline(seed, 'forward', max_length, step_size)
                backward = self._integrate_streamline(seed, 'backward', max_length, step_size)
                # Reverse backward and concatenate
                streamline = np.vstack([backward[::-1][:-1], forward])
                self.streamlines.append(streamline)
        
        return self.streamlines
    
    def plot_streamlines(
        self,
        color_by: Literal['velocity_magnitude', 'uniform', 'index'] = 'velocity_magnitude',
        linewidth: Union[float, Literal['variable']] = 1.5,
        cmap: str = 'coolwarm',
        background: Optional[np.ndarray] = None,
        background_cmap: str = 'viridis',
        background_label: str = 'Background Field',
        figsize: Tuple[float, float] = (12, 4),
        title: str = 'Streamlines',
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot static streamlines.
        
        Args:
            color_by: How to color streamlines
            linewidth: Line width (constant or 'variable' for velocity-based)
            cmap: Colormap for streamlines
            background: Optional background field to display
            background_cmap: Colormap for background
            background_label: Label for background colorbar
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background if provided
        if background is not None:
            R, Z = np.meshgrid(self.grid_r, self.grid_z, indexing='ij')
            contour = ax.contourf(Z * 1000, R * 1000, background, 
                                  levels=20, cmap=background_cmap, alpha=0.7)
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label(background_label, fontsize=11)
        
        # Plot streamlines
        for i, streamline in enumerate(self.streamlines):
            if len(streamline) < 2:
                continue
            
            z = streamline[:, 0] * 1000  # Convert to mm
            r = streamline[:, 1] * 1000
            
            if color_by == 'velocity_magnitude':
                # Calculate velocity magnitude along streamline
                velocities = []
                for point in streamline:
                    vel = self._get_velocity(point)
                    velocities.append(np.linalg.norm(vel))
                velocities = np.array(velocities)
                
                # Plot with colormap
                points = np.array([z, r]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                from matplotlib.collections import LineCollection
                if linewidth == 'variable':
                    # Variable width based on velocity
                    widths = 0.5 + 2.0 * velocities / velocities.max()
                    lc = LineCollection(segments, linewidths=widths, cmap=cmap)
                else:
                    lc = LineCollection(segments, linewidths=linewidth, cmap=cmap)
                
                lc.set_array(velocities)
                ax.add_collection(lc)
            
            elif color_by == 'uniform':
                ax.plot(z, r, 'k-', linewidth=linewidth, alpha=0.7)
            
            elif color_by == 'index':
                color = plt.cm.get_cmap(cmap)(i / len(self.streamlines))
                ax.plot(z, r, color=color, linewidth=linewidth)
        
        if color_by == 'velocity_magnitude':
            # Add colorbar for velocity
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            if background is None:
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Velocity Magnitude [m/s]', fontsize=11)
        
        ax.set_xlabel('Axial Position [mm]', fontsize=12)
        ax.set_ylabel('Radial Position [mm]', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def animate_streamlines_growth(
        self,
        fps: int = 30,
        duration: float = 5.0,
        figsize: Tuple[float, float] = (12, 4),
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Animate progressive growth of streamlines.
        
        Args:
            fps: Frames per second
            duration: Animation duration in seconds
            figsize: Figure size
            save_path: Path to save animation
        
        Returns:
            Animation object
        """
        if not self.streamlines:
            raise ValueError("No streamlines generated. Call generate_streamlines() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up axis limits
        all_points = np.vstack(self.streamlines)
        ax.set_xlim(all_points[:, 0].min() * 1000, all_points[:, 0].max() * 1000)
        ax.set_ylim(all_points[:, 1].min() * 1000, all_points[:, 1].max() * 1000)
        ax.set_xlabel('Axial Position [mm]', fontsize=12)
        ax.set_ylabel('Radial Position [mm]', fontsize=12)
        ax.set_title('Streamlines Growth', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        n_frames = int(fps * duration)
        lines = [ax.plot([], [], 'b-', linewidth=1.5)[0] for _ in self.streamlines]
        
        def init():
            for line in lines:
                line.set_data([], [])
            return lines
        
        def animate(frame):
            progress = frame / n_frames
            
            for i, (line, streamline) in enumerate(zip(lines, self.streamlines)):
                n_points = len(streamline)
                n_show = int(progress * n_points)
                
                if n_show > 0:
                    z = streamline[:n_show, 0] * 1000
                    r = streamline[:n_show, 1] * 1000
                    line.set_data(z, r)
            
            return lines
        
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=1000/fps, blit=True
        )
        
        if save_path:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=5000)
            anim.save(save_path, writer=writer)
        
        return anim
