"""
Exporters module for saving simulation results.

This module provides functions to export results to various formats including
CSV, VTK, and JSON.
"""

from typing import Dict, Any, Optional
import numpy as np
import csv
import json
from pathlib import Path


def export_to_csv(
    data: Dict[str, np.ndarray],
    filename: str,
    header: Optional[str] = None
) -> None:
    """
    Export data to CSV file.

    Args:
        data: Dictionary with column names as keys and numpy arrays as values
        filename: Output filename
        header: Optional header comment

    Example:
        >>> data = {'x': np.array([1, 2, 3]), 'y': np.array([4, 5, 6])}
        >>> export_to_csv(data, 'output.csv')
    """
    # Ensure all arrays have the same length
    lengths = [len(v) for v in data.values()]
    if len(set(lengths)) > 1:
        raise ValueError("All arrays must have the same length")

    # Create output directory if needed
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header comment if provided
        if header:
            f.write(f"# {header}\n")

        # Write column names
        writer.writerow(data.keys())

        # Write data rows
        n_rows = lengths[0]
        for i in range(n_rows):
            row = [data[key][i] for key in data.keys()]
            writer.writerow(row)


def export_results_to_csv(
    positions: np.ndarray,
    results: Dict[str, np.ndarray],
    filename: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export simulation results to CSV with metadata.

    Args:
        positions: Position array
        results: Dictionary of result arrays
        filename: Output filename
        metadata: Optional metadata dictionary
    """
    # Combine positions and results
    data = {'position': positions}
    data.update(results)

    # Create header with metadata
    header = "Simulation Results"
    if metadata:
        header += " | " + " | ".join([f"{k}={v}" for k, v in metadata.items()])

    export_to_csv(data, filename, header)


def export_to_vtk_structured_grid(
    coordinates: Dict[str, np.ndarray],
    fields: Dict[str, np.ndarray],
    filename: str
) -> None:
    """
    Export data to VTK structured grid format (legacy ASCII).

    Args:
        coordinates: Dictionary with 'x', 'y', 'z' coordinate arrays
        fields: Dictionary with field names and values
        filename: Output filename (.vtk)

    Note:
        This creates a simple VTK legacy format file.
        For more complex VTK export, use pyvista library.
    """
    # Ensure .vtk extension
    if not filename.endswith('.vtk'):
        filename += '.vtk'

    # Create output directory if needed
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    x = coordinates.get('x', np.array([0]))
    y = coordinates.get('y', np.array([0]))
    z = coordinates.get('z', np.array([0]))

    nx, ny, nz = len(x), len(y), len(z)
    n_points = nx * ny * nz

    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("CFD Simulation Results\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"POINTS {n_points} float\n")

        # Write points
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{x[i]} {y[j]} {z[k]}\n")

        # Write point data
        f.write(f"\nPOINT_DATA {n_points}\n")

        for field_name, field_data in fields.items():
            f.write(f"\nSCALARS {field_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in field_data.flatten():
                f.write(f"{value}\n")


def export_to_json(
    data: Dict[str, Any],
    filename: str,
    indent: int = 2
) -> None:
    """
    Export data to JSON file.

    Args:
        data: Dictionary with data to export
        filename: Output filename
        indent: Indentation level

    Note:
        Numpy arrays are converted to lists.
    """
    # Create output directory if needed
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    data_converted = convert_numpy(data)

    with open(filename, 'w') as f:
        json.dump(data_converted, f, indent=indent)


def save_simulation_summary(
    results: Dict[str, Any],
    filename: str = "simulation_summary.json"
) -> None:
    """
    Save a comprehensive simulation summary to JSON.

    Args:
        results: Dictionary with simulation results and parameters
        filename: Output filename
    """
    export_to_json(results, filename, indent=2)


class ResultsExporter:
    """Class for managing result exports."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize results exporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        data: Dict[str, Any],
        base_name: str,
        formats: list = ['csv', 'json']
    ) -> Dict[str, str]:
        """
        Export data to multiple formats.

        Args:
            data: Data to export
            base_name: Base filename (without extension)
            formats: List of formats to export ('csv', 'json', 'vtk')

        Returns:
            Dictionary mapping format to output filename
        """
        output_files = {}

        for fmt in formats:
            if fmt == 'csv':
                filename = self.output_dir / f"{base_name}.csv"
                export_to_csv(data, str(filename))
                output_files['csv'] = str(filename)

            elif fmt == 'json':
                filename = self.output_dir / f"{base_name}.json"
                export_to_json(data, str(filename))
                output_files['json'] = str(filename)

        return output_files
