from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import PLOTLY_SURFACES_DIR


GLOBAL_SPEED_SURFACE_FILENAME = "global_speed_surface.json"


def load_global_speed_surface(repository_root: Path) -> dict[str, Any] | None:
    for directory in _surface_candidate_dirs(repository_root):
        surface = _load_global_speed_surface_from_dir(directory)
        if surface is not None:
            return surface
    return None


def _surface_candidate_dirs(repository_root: Path) -> list[Path]:
    return [PLOTLY_SURFACES_DIR]


def _load_global_speed_surface_from_dir(surface_dir: Path) -> dict[str, Any] | None:
    json_path = surface_dir / GLOBAL_SPEED_SURFACE_FILENAME
    if not json_path.exists():
        return None

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    pressure_axis = np.asarray(payload.get("x"), dtype=float)
    pressure_rotation = np.asarray(payload.get("y"), dtype=float)
    speed = np.asarray(payload.get("z"), dtype=float)
    if pressure_axis.size == 0 or pressure_rotation.size == 0 or speed.size == 0:
        return None

    pressure_axis, pressure_rotation, speed = _as_indexing_ij(
        pressure_axis,
        pressure_rotation,
        speed,
    )
    if pressure_axis.ndim != 2 or pressure_rotation.ndim != 2 or speed.ndim != 2:
        return None

    return {
        "x": pressure_axis,
        "y": pressure_rotation,
        "z": speed,
        "source": str(payload.get("source", "global_speed_surface")),
        "metadata": payload.get("metadata", {}),
    }


def _as_indexing_ij(
    pressure_axis: np.ndarray,
    pressure_rotation: np.ndarray,
    speed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pressure_axis.ndim != 2 or pressure_rotation.ndim != 2 or speed.ndim != 2:
        return pressure_axis, pressure_rotation, speed

    axis_changes_by_column = float(np.nanstd(pressure_axis[0, :]))
    axis_changes_by_row = float(np.nanstd(pressure_axis[:, 0]))
    if axis_changes_by_column > axis_changes_by_row:
        return pressure_axis.T, pressure_rotation.T, speed.T
    return pressure_axis, pressure_rotation, speed
