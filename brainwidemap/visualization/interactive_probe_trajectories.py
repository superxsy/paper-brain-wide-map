"""Interactive 3D Neuropixels probe trajectory viewer.

This module downloads Neuropixels probe trajectories for the Brainwide Map dataset and
renders them in an interactive Plotly figure.  The viewer allows orbiting, panning and
zooming directly from a web browser and can optionally save the output to an HTML file.

Example
-------
To generate a viewer for all probes included in the 2023-12 release:

    python -m brainwidemap.visualization.interactive_probe_trajectories \
        --freeze 2023_12_bwm_release \
        --output neuropixels_trajectories.html

The command opens the figure in a browser (if supported) and stores a self-contained
HTML file that can be shared with collaborators.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from one.api import ONE

from brainwidemap.bwm_loading import bwm_query

_LOGGER = logging.getLogger(__name__)


@dataclass
class TrajectoryRecord:
    """Container holding per-probe trajectory metadata and sampled coordinates."""

    pid: str
    lab: str
    subject: str
    probe_name: str
    session_date: str
    session_number: int
    points: np.ndarray  # Shape (n_points, 3) with columns (x, y, z)

    @property
    def hover_text(self) -> str:
        return (
            f"PID: {self.pid}<br>"
            f"Lab: {self.lab}<br>"
            f"Subject: {self.subject}<br>"
            f"Probe: {self.probe_name}<br>"
            f"Session: {self.session_date} (#{self.session_number})"
        )


def _compute_end_point(traj: Dict[str, float]) -> np.ndarray:
    """Return the tip coordinate computed from the straight trajectory parameters."""

    origin = np.array([traj.get("x"), traj.get("y"), traj.get("z")], dtype=float)
    depth = float(traj.get("depth", 0.0))
    theta = np.deg2rad(float(traj.get("theta", 0.0)))
    phi = np.deg2rad(float(traj.get("phi", 0.0)))

    # Convert spherical coordinates (Allen atlas convention) to Cartesian deltas
    dx = depth * np.sin(theta) * np.cos(phi)
    dy = depth * np.sin(theta) * np.sin(phi)
    dz = depth * np.cos(theta)
    return origin + np.array([dx, dy, dz])


def _extract_xyz_sequence(traj: Dict, n_points: int) -> np.ndarray:
    """Return an ordered sequence of XYZ coordinates for the trajectory."""

    picks = traj.get("xyz_picks") or []

    coords: List[Sequence[float]] = []
    if isinstance(picks, Sequence) and picks and not isinstance(picks, (str, bytes)):
        first = picks[0]
        if isinstance(first, dict):
            coords = [
                (p.get("x"), p.get("y"), p.get("z"))
                for p in picks
                if p is not None and {"x", "y", "z"} <= p.keys()
            ]
        elif isinstance(first, Sequence) and len(first) >= 3:
            coords = [tuple(p[:3]) for p in picks]

    if coords:
        arr = np.asarray(coords, dtype=float)
        # Ensure ordering from entry to tip
        return arr

    start = np.array([traj.get("x"), traj.get("y"), traj.get("z")], dtype=float)
    end = _compute_end_point(traj)
    return np.linspace(start, end, n_points)


def fetch_probe_trajectories(
    one: ONE,
    metadata: pd.DataFrame,
    provenance: str = "Histology track",
    n_points: int = 25,
    max_probes: Optional[int] = None,
) -> List[TrajectoryRecord]:
    """Download the trajectory coordinates for each probe described in *metadata*.

    Parameters
    ----------
    one
        Connected ONE client instance.
    metadata
        DataFrame returned by :func:`brainwidemap.bwm_loading.bwm_query`.
    provenance
        Preferred trajectory provenance (default: "Histology track").
    n_points
        Number of samples along the straight trajectory used when the trajectory
        does not include manual picks.
    max_probes
        Optional limit on the number of probes to download (useful for quick tests).
    """

    records: List[TrajectoryRecord] = []
    for row in metadata.itertuples(index=False):
        if max_probes is not None and len(records) >= max_probes:
            break

        try:
            traj = _get_preferred_trajectory(one, row.pid, provenance=provenance)
        except RuntimeError as exc:
            _LOGGER.warning("Skipping %s (%s): %s", row.pid, row.lab, exc)
            continue

        points = _extract_xyz_sequence(traj, n_points=n_points)
        record = TrajectoryRecord(
            pid=row.pid,
            lab=row.lab,
            subject=row.subject,
            probe_name=row.probe_name,
            session_date=str(row.date),
            session_number=int(row.session_number),
            points=points,
        )
        records.append(record)

    if not records:
        raise RuntimeError(
            "No trajectories were retrieved. Ensure your ONE credentials are "
            "configured and that the requested probes have histology tracks."
        )

    return records


def _get_preferred_trajectory(one: ONE, pid: str, provenance: str) -> Dict:
    """Return the trajectory dictionary matching *provenance* for a probe."""

    trajectories = one.alyx.rest(
        "trajectories",
        "list",
        probe_insertion=pid,
    )

    if not trajectories:
        raise RuntimeError("no trajectories available")

    for traj in trajectories:
        if traj.get("provenance") == provenance:
            return traj

    _LOGGER.debug(
        "Falling back to first trajectory for %s (no %s provenance)", pid, provenance
    )
    return trajectories[0]


def _color_mapping(values: Iterable[str]) -> Dict[str, str]:
    palette = plotly.colors.qualitative.Dark24
    colors = {}
    for idx, value in enumerate(sorted(set(values))):
        colors[value] = palette[idx % len(palette)]
    return colors


def build_probe_trajectory_figure(
    records: Sequence[TrajectoryRecord],
    color_by: str = "lab",
    brain_mesh: Optional[Dict[str, np.ndarray]] = None,
) -> go.Figure:
    """Create an interactive Plotly figure for the given trajectories."""

    if color_by not in {"lab", "subject", "probe_name"}:
        raise ValueError("color_by must be one of 'lab', 'subject', 'probe_name'")

    color_values = [getattr(rec, color_by) for rec in records]
    palette = _color_mapping(color_values)
    legend_tracker: Dict[str, bool] = {}

    fig = go.Figure()

    if brain_mesh is not None:
        fig.add_trace(
            go.Mesh3d(
                x=brain_mesh["x"],
                y=brain_mesh["y"],
                z=brain_mesh["z"],
                i=brain_mesh["i"],
                j=brain_mesh["j"],
                k=brain_mesh["k"],
                opacity=0.05,
                color="lightgray",
                name="Brain volume",
                showscale=False,
                hoverinfo="skip",
            )
        )

    for rec in records:
        color_value = getattr(rec, color_by)
        show_legend = not legend_tracker.get(color_value, False)
        fig.add_trace(
            go.Scatter3d(
                x=rec.points[:, 0],
                y=rec.points[:, 1],
                z=rec.points[:, 2],
                mode="lines",
                line=dict(color=palette[color_value], width=3),
                name=str(color_value),
                hoverinfo="text",
                text=rec.hover_text,
                showlegend=show_legend,
            )
        )
        legend_tracker[color_value] = True

    fig.update_layout(
        scene=dict(
            xaxis_title="ML (µm)",
            yaxis_title="AP (µm)",
            zaxis_title="DV (µm)",
            aspectmode="data",
        ),
        legend_title=color_by.capitalize(),
        margin=dict(l=0, r=0, b=0, t=40),
        title="Neuropixels probe trajectories",
    )

    return fig


def _load_brain_mesh(resolution_um: int = 100) -> Optional[Dict[str, np.ndarray]]:
    try:
        from iblatlas.atlas import AllenAtlas

        atlas = AllenAtlas(res_um=resolution_um)
        verts, faces = atlas.mesh_from_label(997)  # 997 = brain label in CCF
        return {
            "x": verts[:, 0],
            "y": verts[:, 1],
            "z": verts[:, 2],
            "i": faces[:, 0],
            "j": faces[:, 1],
            "k": faces[:, 2],
        }
    except Exception as exc:  # pragma: no cover - best effort optional dependency
        _LOGGER.warning("Unable to load Allen atlas mesh: %s", exc)
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        default="2023_12_bwm_release",
        help=(
            "Name of the Brainwide Map data freeze to use. Pass 'None' to query the "
            "database directly."
        ),
    )
    parser.add_argument(
        "--color-by",
        choices=["lab", "subject", "probe_name"],
        default="lab",
        help="Metadata field used to color trajectories.",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=None,
        help="Optional limit on the number of probes (useful for quick previews).",
    )
    parser.add_argument(
        "--no-brain",
        action="store_true",
        help="Disable loading the Allen atlas surface mesh.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the generated HTML file. If omitted the figure is only displayed.",
    )
    parser.add_argument(
        "--provenance",
        default="Histology track",
        help="Preferred trajectory provenance to load from Alyx.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=25,
        help="Number of samples along straight trajectories when no manual picks are present.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    one = ONE()

    freeze = None if args.freeze in {"None", "none", ""} else args.freeze
    if freeze is None:
        metadata, _ = bwm_query(one=one, return_details=True, freeze=None)
    else:
        metadata = bwm_query(freeze=freeze)

    records = fetch_probe_trajectories(
        one=one,
        metadata=metadata,
        provenance=args.provenance,
        n_points=args.n_points,
        max_probes=args.max_probes,
    )

    brain_mesh = None if args.no_brain else _load_brain_mesh()
    fig = build_probe_trajectory_figure(records, color_by=args.color_by, brain_mesh=brain_mesh)

    if args.output:
        fig.write_html(args.output)
        _LOGGER.info("Saved figure to %s", args.output)

    fig.show()


if __name__ == "__main__":
    main()
