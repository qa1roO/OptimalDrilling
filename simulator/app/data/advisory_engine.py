from __future__ import annotations

import json
import math
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.exceptions import InconsistentVersionWarning


EPS = 1e-9
HISTORY_COLUMNS = (
    "pressure_axis",
    "pressure_rotation",
    "rotation",
    "speed",
    "hardness_score_smooth",
    "energy_input_proxy",
    "pressure_balance",
)
LAG_STEPS = (1, 3, 6, 12)
ROLL_WINDOWS = (6, 12, 30)
DIFF_COLUMNS = (
    "pressure_axis",
    "pressure_rotation",
    "rotation",
    "speed",
    "hardness_score_smooth",
)


class AdvisoryEngine:
    def __init__(
        self,
        artifact_dir: str | Path,
        buffer_size: int = 60,
        min_buffer_size: int = 30,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.rows: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._last_recommendation: dict[str, Any] | None = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            self.rotation_model = joblib.load(self.artifact_dir / "rotation_model_near5.joblib")
            self.speed_model = joblib.load(self.artifact_dir / "speed_model_near5.joblib")

        self.feature_config = _read_json(self.artifact_dir / "feature_config.json")
        self.optimizer_config = _read_json(self.artifact_dir / "optimizer_config.json")
        self.surface_ranges = _read_json(self.artifact_dir / "surface_ranges_by_energy_type.json")

        self.base_numeric_features = list(self.feature_config["base_numeric_features"])
        self.categorical_features = list(self.feature_config["categorical_features"])
        self.speed_numeric_features = list(self.feature_config["speed_numeric_features"])
        self.energy_type_column = str(self.feature_config["energy_type_column"])

        self.grid_size = int(self.optimizer_config.get("grid_size_default", 21))
        self.max_delta_frac = float(self.optimizer_config.get("max_delta_frac_default", 0.08))
        self.change_penalty_weight = float(self.optimizer_config.get("change_penalty_weight", 0.010))
        self.boundary_penalty_weight = float(self.optimizer_config.get("boundary_penalty_weight", 0.020))
        self.boundary_start = float(self.optimizer_config.get("boundary_start", 0.85))

    def update(self, telemetry_row: dict[str, Any]) -> dict[str, Any] | None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PerformanceWarning)
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            self.rows.append(_normalize_telemetry_row(telemetry_row))
            if len(self.rows) < self.min_buffer_size:
                self._last_recommendation = None
                return None

            features = self._build_feature_frame()
            current = features.iloc[-1].copy()
            recommendation = self._build_recommendation(current)
            self._last_recommendation = recommendation
            return recommendation

    def get_recommendation(self) -> dict[str, Any] | None:
        return self._last_recommendation

    def reset(self) -> None:
        self.rows.clear()
        self._last_recommendation = None

    def _build_feature_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self.rows)).copy()
        _coerce_numeric_columns(
            df,
            [
                "pressure_axis",
                "pressure_rotation",
                "rotation",
                "speed",
                "hardness_score_smooth",
            ],
        )
        df["processing_time"] = pd.to_datetime(df["processing_time"], errors="coerce")
        df["dt"] = df["processing_time"].diff().dt.total_seconds()
        median_dt = float(df["dt"].dropna().median()) if df["dt"].notna().any() else 0.0
        df["dt"] = df["dt"].fillna(median_dt).fillna(0.0)
        self._add_derived_features(df)

        for column in HISTORY_COLUMNS:
            for lag in LAG_STEPS:
                df[f"{column}_lag{lag}"] = df[column].shift(lag)
            shifted = df[column].shift(1)
            for window in ROLL_WINDOWS:
                df[f"{column}_roll_mean_{window}"] = shifted.rolling(window).mean()
                df[f"{column}_roll_std_{window}"] = shifted.rolling(window).std()

        for column in DIFF_COLUMNS:
            df[f"{column}_diff1"] = df[column].diff()
            previous = df[column].shift(1)
            df[f"{column}_rel_diff1"] = df[f"{column}_diff1"] / (previous.abs() + EPS)

        numeric_features = sorted(set(self.base_numeric_features + self.speed_numeric_features))
        for column in numeric_features:
            if column not in df.columns:
                df[column] = 0.0
        df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors="coerce")
        df[numeric_features] = df[numeric_features].ffill().bfill().fillna(0.0)

        for column in self.categorical_features:
            if column not in df.columns:
                df[column] = "unknown"
            df[column] = df[column].astype(str).fillna("unknown")

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> None:
        total_pressure = df["pressure_axis"] + df["pressure_rotation"]
        df["total_pressure"] = total_pressure
        df["pressure_balance"] = df["pressure_axis"] / (total_pressure + EPS)
        df["axis_over_rot_pressure"] = df["pressure_axis"] / (df["pressure_rotation"] + EPS)
        df["rot_pressure_over_axis"] = df["pressure_rotation"] / (df["pressure_axis"] + EPS)
        df["rotation_efficiency"] = df["rotation"] / (df["pressure_rotation"] + EPS)
        df["axis_x_rotation"] = df["pressure_axis"] * df["rotation"]
        df["rot_pressure_x_rotation"] = df["pressure_rotation"] * df["rotation"]
        df["energy_input_proxy"] = df["pressure_axis"] + df["pressure_rotation"] * df["rotation"]
        df["log_energy_input_proxy"] = np.log1p(np.maximum(df["energy_input_proxy"], 0.0))

    def _build_recommendation(self, current: pd.Series) -> dict[str, Any]:
        grid = self._candidate_grid(current)
        self._add_candidate_derived_features(grid)

        rotation_features = self.base_numeric_features + self.categorical_features
        speed_features = self.speed_numeric_features + self.categorical_features

        grid["candidate_target_rotation"] = self.rotation_model.predict(grid[rotation_features])
        grid["pred_target_speed"] = self.speed_model.predict(grid[speed_features])

        current_grid = pd.DataFrame([current.to_dict()])
        current_grid["candidate_target_rotation"] = self.rotation_model.predict(current_grid[rotation_features])
        current_pred_speed = float(self.speed_model.predict(current_grid[speed_features])[0])

        axis_delta_frac = grid["pressure_axis"] / float(current["pressure_axis"]) - 1.0
        rot_delta_frac = grid["pressure_rotation"] / float(current["pressure_rotation"]) - 1.0
        axis_delta_norm = axis_delta_frac.abs() / self.max_delta_frac
        rot_delta_norm = rot_delta_frac.abs() / self.max_delta_frac

        change_penalty = (
            self.change_penalty_weight
            * current_pred_speed
            * (axis_delta_norm**2 + rot_delta_norm**2)
            / 2.0
        )
        axis_edge = np.clip((axis_delta_norm - self.boundary_start) / (1.0 - self.boundary_start), 0.0, 1.0)
        rot_edge = np.clip((rot_delta_norm - self.boundary_start) / (1.0 - self.boundary_start), 0.0, 1.0)
        boundary_penalty = (
            self.boundary_penalty_weight
            * current_pred_speed
            * (axis_edge**2 + rot_edge**2)
            / 2.0
        )

        grid["optimizer_score"] = grid["pred_target_speed"] - change_penalty - boundary_penalty
        best = grid.loc[int(grid["optimizer_score"].idxmax())]
        uplift = _safe_uplift(best["pred_target_speed"], current_pred_speed)

        surface = self._surface_payload(grid)
        return {
            "energy_type": str(current[self.energy_type_column]),
            "current": {
                "pressure_axis": float(current["pressure_axis"]),
                "pressure_rotation": float(current["pressure_rotation"]),
                "rotation": float(current["rotation"]),
                "speed": float(current["speed"]),
                "predicted_target_speed": current_pred_speed,
            },
            "recommended": {
                "pressure_axis": float(best["pressure_axis"]),
                "pressure_rotation": float(best["pressure_rotation"]),
                "predicted_target_rotation": float(best["candidate_target_rotation"]),
                "predicted_target_speed": float(best["pred_target_speed"]),
                "optimizer_score": float(best["optimizer_score"]),
                "predicted_uplift_pct": uplift,
                "delta_pressure_axis_pct": float(axis_delta_frac.loc[best.name] * 100.0),
                "delta_pressure_rotation_pct": float(rot_delta_frac.loc[best.name] * 100.0),
            },
            "surface": surface,
        }

    def _candidate_grid(self, current: pd.Series) -> pd.DataFrame:
        p_ax = float(current["pressure_axis"])
        p_rot = float(current["pressure_rotation"])
        local_ax_low = p_ax * (1.0 - self.max_delta_frac)
        local_ax_high = p_ax * (1.0 + self.max_delta_frac)
        local_rot_low = p_rot * (1.0 - self.max_delta_frac)
        local_rot_high = p_rot * (1.0 + self.max_delta_frac)

        energy_type = str(current[self.energy_type_column])
        ranges = self.surface_ranges.get(energy_type, {})
        p_ax_min = max(float(ranges.get("pressure_axis_q05", local_ax_low)), local_ax_low)
        p_ax_max = min(float(ranges.get("pressure_axis_q95", local_ax_high)), local_ax_high)
        p_rot_min = max(float(ranges.get("pressure_rotation_q05", local_rot_low)), local_rot_low)
        p_rot_max = min(float(ranges.get("pressure_rotation_q95", local_rot_high)), local_rot_high)

        if p_ax_min >= p_ax_max:
            p_ax_min, p_ax_max = local_ax_low, local_ax_high
        if p_rot_min >= p_rot_max:
            p_rot_min, p_rot_max = local_rot_low, local_rot_high

        p_ax_grid = np.linspace(p_ax_min, p_ax_max, self.grid_size)
        p_rot_grid = np.linspace(p_rot_min, p_rot_max, self.grid_size)
        pa, pr = np.meshgrid(p_ax_grid, p_rot_grid, indexing="ij")
        grid = pd.DataFrame(
            {
                "pressure_axis": pa.ravel(),
                "pressure_rotation": pr.ravel(),
            }
        )
        for column, value in current.items():
            if column not in grid.columns:
                grid[column] = value
        grid["pressure_axis"] = pa.ravel()
        grid["pressure_rotation"] = pr.ravel()
        return grid

    def _add_candidate_derived_features(self, grid: pd.DataFrame) -> None:
        self._add_derived_features(grid)
        numeric_features = sorted(set(self.base_numeric_features + self.speed_numeric_features))
        for column in numeric_features:
            if column not in grid.columns:
                grid[column] = 0.0
        for column in self.categorical_features:
            if column not in grid.columns:
                grid[column] = "unknown"
            grid[column] = grid[column].astype(str).fillna("unknown")
        grid[numeric_features] = grid[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def _surface_payload(self, grid: pd.DataFrame) -> dict[str, Any]:
        shape = (self.grid_size, self.grid_size)
        return {
            "x": grid["pressure_axis"].to_numpy(dtype=float).reshape(shape),
            "y": grid["pressure_rotation"].to_numpy(dtype=float).reshape(shape),
            "z": grid["pred_target_speed"].to_numpy(dtype=float).reshape(shape),
            "score": grid["optimizer_score"].to_numpy(dtype=float).reshape(shape),
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _normalize_telemetry_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for column in (
        "pressure_axis",
        "pressure_rotation",
        "rotation",
        "speed",
        "hardness_score_smooth",
    ):
        result[column] = _to_float(result.get(column, 0.0))
    result["processing_time"] = result.get("processing_time", "")
    result["well_id"] = str(result.get("well_id", ""))
    result["rock_energy_type_final"] = str(result.get("rock_energy_type_final", "unknown"))
    return result


def _to_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(number):
        return number
    return 0.0


def _safe_uplift(best_speed: float, current_speed: float) -> float:
    if abs(current_speed) < EPS:
        return 0.0
    return float(100.0 * (best_speed / current_speed - 1.0))
