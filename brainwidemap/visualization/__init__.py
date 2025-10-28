"""Visualization utilities for Brainwide Map datasets."""

from .interactive_probe_trajectories import (
    build_probe_trajectory_figure,
    fetch_probe_trajectories,
    main as plot_probe_trajectories_cli,
)

__all__ = [
    "build_probe_trajectory_figure",
    "fetch_probe_trajectories",
    "plot_probe_trajectories_cli",
]
