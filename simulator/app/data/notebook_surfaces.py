from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np


ENERGY_TYPES = (
    "soft_low_energy",
    "medium_low_energy",
    "medium_high_energy",
    "hard_high_energy",
)


def load_notebook_surface(energy_type: str, repository_root: Path) -> dict[str, Any] | None:
    surface_dir = _find_surface_dir(repository_root)
    if surface_dir is None:
        return None
    return _load_notebook_surface_from_dir(energy_type, surface_dir)


def load_notebook_surfaces(repository_root: Path) -> dict[str, dict[str, Any]]:
    surfaces: dict[str, dict[str, Any]] = {}
    _print_surface_search_diagnostics(repository_root)
    surface_dir = _find_surface_dir(repository_root)
    if surface_dir is None:
        print("Notebook surfaces not found, using advisory fallback surfaces")
        return surfaces

    for energy_type in ENERGY_TYPES:
        surface = _load_notebook_surface_from_dir(energy_type, surface_dir)
        if surface is not None:
            surfaces[energy_type] = surface
    print(f"Loaded notebook surfaces from: {surface_dir}")
    print(f"Loaded energy surfaces: {list(surfaces)}")
    return surfaces


def _find_surface_dir(repository_root: Path) -> Path | None:
    for directory in _surface_candidate_dirs(repository_root):
        if directory.exists() and any((directory / f"surface_{energy_type}.html").exists() for energy_type in ENERGY_TYPES):
            return directory
    return None


def _surface_candidate_dirs(repository_root: Path) -> list[Path]:
    return [
        repository_root / "notebooks" / "plotly_surfaces_html",
        repository_root / "plotly_surfaces_html",
        repository_root / "simulator" / "app" / "data" / "plotly_surfaces_html",
    ]


def _print_surface_search_diagnostics(repository_root: Path) -> None:
    for directory in _surface_candidate_dirs(repository_root):
        existing = [energy_type for energy_type in ENERGY_TYPES if (directory / f"surface_{energy_type}.html").exists()]
        if existing:
            print(f"Notebook surface candidate found: {directory} ({existing})")
        else:
            print(f"Notebook surface candidate missing or empty: {directory}")


def _load_notebook_surface_from_dir(energy_type: str, surface_dir: Path) -> dict[str, Any] | None:
    html_path = surface_dir / f"surface_{energy_type}.html"
    if not html_path.exists():
        return None

    text = html_path.read_text(encoding="utf-8")
    data_start = _find_plotly_data_start(text)
    if data_start < 0:
        return None

    data_end = _find_matching_bracket(text, data_start)
    if data_end < 0:
        return None

    traces = json.loads(text[data_start : data_end + 1])
    if not traces:
        return None

    trace = traces[0]
    pressure_axis = _decode_plotly_array(trace.get("x"))
    pressure_rotation = _decode_plotly_array(trace.get("y"))
    speed = _decode_plotly_array(trace.get("z"))
    if pressure_axis is None or pressure_rotation is None or speed is None:
        return None

    pressure_axis, pressure_rotation, speed = _as_indexing_ij(
        pressure_axis,
        pressure_rotation,
        speed,
    )
    return {
        "x": pressure_axis,
        "y": pressure_rotation,
        "z": speed,
        "source": "notebook_plotly_surface",
    }


def _find_plotly_data_start(text: str) -> int:
    call_start = text.rfind("Plotly.newPlot")
    if call_start < 0:
        return -1

    first_comma = text.find(",", call_start)
    if first_comma < 0:
        return -1

    return text.find("[", first_comma)


def _find_matching_bracket(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _decode_plotly_array(value: Any) -> np.ndarray | None:
    if isinstance(value, dict) and "bdata" in value:
        dtype = np.dtype(value.get("dtype", "f8"))
        raw = base64.b64decode(value["bdata"])
        array = np.frombuffer(raw, dtype=dtype).astype(float, copy=False)
        shape_text = str(value.get("shape", "")).strip()
        if shape_text:
            shape = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
            array = array.reshape(shape)
        return array

    if value is None:
        return None
    return np.asarray(value, dtype=float)


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
