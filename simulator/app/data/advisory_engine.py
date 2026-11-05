"""Streaming adapter for the exact same causal pipeline used by training."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import drilling_advisory as core


class AdvisoryEngine:
    def __init__(self, artifact_dir: str | Path, buffer_size: int = 60, min_buffer_size: int = 30):
        self.artifact_dir = Path(artifact_dir)
        if buffer_size < 30 or min_buffer_size < 13 or min_buffer_size > buffer_size:
            raise ValueError("History requires buffer >=30 and 13 <= warm-up <= buffer.")
        self.buffer_size, self.min_buffer_size = buffer_size, min_buffer_size
        self.rows: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._last_recommendation = None
        self.feature_config = self._read_json("feature_config.json")
        self.optimizer_config = self._read_json("optimizer_config.json")
        if self.feature_config.get("schema_version") != core.SCHEMA_VERSION:
            raise ValueError("Artifacts use an old feature schema; run train_advisory.py first.")
        self.calibration = core.EnergyCalibration.from_dict(self._read_json("energy_calibration.json"))
        self.surface_ranges = self._read_json("candidate_pressure_ranges_by_energy_type.json")
        self.rotation_model = joblib.load(self.artifact_dir / "rotation_model_near5.joblib")
        self.speed_model = joblib.load(self.artifact_dir / "speed_model_near5.joblib")
        self.base_numeric_features = core.NUMERIC_FEATURES
        self.categorical_features = core.CATEGORICAL_FEATURES
        self.speed_numeric_features = core.SPEED_NUMERIC_FEATURES
        self.energy_type_column = "rock_energy_type_final"
        self.grid_size = int(self.optimizer_config["grid_size_default"])

    def _read_json(self, name):
        return json.loads((self.artifact_dir / name).read_text(encoding="utf-8"))

    def reset(self):
        self.rows.clear()
        self._last_recommendation = None

    def get_recommendation(self):
        return self._last_recommendation

    def update(self, telemetry_row: dict[str, Any], compute: bool = True):
        # Ignore any precomputed energy labels/scores: derive them from raw past measurements.
        raw = {c: telemetry_row[c] for c in core.RAW_COLUMNS}
        raw["well_id"] = str(raw["well_id"])
        if self.rows and str(self.rows[-1]["well_id"]) != raw["well_id"]:
            self.reset()
        self.rows.append(raw)
        if len(self.rows) < self.min_buffer_size:
            self._last_recommendation = None
            return None
        if not compute and self._last_recommendation is not None:
            return self._last_recommendation
        current = self._build_feature_frame().iloc[-1]
        self._last_recommendation = self._build_recommendation(current)
        return self._last_recommendation

    def _build_feature_frame(self):
        return self.calibration.transform(core.prepare_history(pd.DataFrame(list(self.rows))))

    def _build_recommendation(self, current):
        best, grid = core.recommend(pd.DataFrame([current]), self.rotation_model, self.speed_model,
                                    self.surface_ranges, self.grid_size)
        selected = best.iloc[0]
        baseline = float(selected.current_predicted_target_speed)
        recommended = float(selected.predicted_target_speed)
        surface_grid = grid.loc[grid.candidate_id != 0]
        shape = (self.grid_size, self.grid_size)
        return {
            "energy_type": str(current[self.energy_type_column]),
            "current": {"pressure_axis": float(current.pressure_axis),
                        "pressure_rotation": float(current.pressure_rotation),
                        "rotation": float(current.rotation), "speed": float(current.speed),
                        "predicted_target_speed": baseline},
            "recommended": {"pressure_axis": float(selected.pressure_axis),
                            "pressure_rotation": float(selected.pressure_rotation),
                            "predicted_target_rotation": float(selected.candidate_target_rotation),
                            "predicted_target_speed": recommended,
                            "optimizer_score": float(selected.optimizer_score),
                            "predicted_uplift_pct": float(100*(recommended/baseline-1)),
                            "delta_pressure_axis_pct": float(100*selected.delta_pressure_axis_frac),
                            "delta_pressure_rotation_pct": float(100*selected.delta_pressure_rotation_frac),
                            "changed_action": bool(selected.candidate_id != 0)},
            "surface": {"x": surface_grid.pressure_axis.to_numpy().reshape(shape),
                        "y": surface_grid.pressure_rotation.to_numpy().reshape(shape),
                        "z": surface_grid.predicted_target_speed.to_numpy().reshape(shape),
                        "score": surface_grid.optimizer_score.to_numpy().reshape(shape)},
        }
