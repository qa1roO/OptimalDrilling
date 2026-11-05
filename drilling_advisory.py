"""Shared causal feature preparation and advisory inference.

Rows describe completed measurements at t. Only targets use t+1..t+h.
Energy is a trailing operational resistance index, not physical rock hardness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import (
    BOUNDARY_PENALTY_WEIGHT, BOUNDARY_START, CHANGE_PENALTY_WEIGHT,
    ENERGY_LABELS_4, EPS, GRID_SIZE, MAX_DELTA_FRAC, TARGET_HORIZON,
)

SCHEMA_VERSION = 2
RAW_COLUMNS = ["well_id", "processing_time", "depth", "d_depth", "d_time",
               "pressure_axis", "pressure_rotation", "rotation", "speed"]
HISTORY_COLUMNS = ["pressure_axis", "pressure_rotation", "rotation", "speed"]
NUMERIC_FEATURES = [
    "pressure_axis", "pressure_rotation", "pressure_balance", "rotation", "speed", "dt",
    "rotation_efficiency", "axis_x_rotation", "energy_input_proxy", "hardness_score_smooth",
    "pressure_axis_lag1", "pressure_axis_lag3", "pressure_axis_lag6",
    "rotation_lag1", "rotation_lag3", "rotation_lag6", "speed_lag1", "speed_lag3",
    *[f"{c}_roll_{s}_12" for c in HISTORY_COLUMNS for s in ["mean", "std"]],
    *[f"{c}_diff1" for c in HISTORY_COLUMNS],
]
CATEGORICAL_FEATURES = ["rock_energy_type_final"]
ROTATION_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SPEED_NUMERIC_FEATURES = NUMERIC_FEATURES + ["candidate_target_rotation"]
SPEED_FEATURES = SPEED_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_control_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["total_pressure"] = out["pressure_axis"] + out["pressure_rotation"]
    out["pressure_balance"] = out["pressure_axis"] / (out["total_pressure"] + EPS)
    out["rotation_efficiency"] = out["rotation"] / (out["pressure_rotation"] + EPS)
    out["axis_x_rotation"] = out["pressure_axis"] * out["rotation"]
    out["energy_input_proxy"] = out["pressure_axis"] + out["pressure_rotation"] * out["rotation"]
    # Lagged pressures are observed history; candidate pressure is the proposed current action.
    for c in ["pressure_axis", "pressure_rotation"]:
        if f"{c}_lag1" in out:
            out[f"{c}_diff1"] = out[c] - out[f"{c}_lag1"]
    return out


def prepare_history(raw: pd.DataFrame) -> pd.DataFrame:
    """No fitted statistics, centered windows, backfilling, or future segment labels."""
    out = raw[RAW_COLUMNS].copy()
    out["processing_time"] = pd.to_datetime(out["processing_time"], format="mixed")
    out["well_id"] = out["well_id"].astype(str)
    for c in RAW_COLUMNS[2:]:
        out[c] = pd.to_numeric(out[c], errors="raise")
    positive = ["d_depth", "d_time", "pressure_axis", "pressure_rotation", "rotation", "speed"]
    if not np.isfinite(out[positive].to_numpy()).all() or (out[positive] <= 0).any().any():
        raise ValueError("Expected finite positive telemetry; preprocess invalid rows first.")
    if out.duplicated(["well_id", "processing_time"]).any():
        raise ValueError("Duplicate timestamps within a well must be resolved before training.")
    out = out.sort_values(["well_id", "processing_time"], kind="stable").reset_index(drop=True)
    out["_row_in_well"] = out.groupby("well_id").cumcount()
    out["speed"] = out["d_depth"] / out["d_time"]
    out["dt"] = out["d_time"]  # identical source in training and live inference
    for c in HISTORY_COLUMNS:
        group = out.groupby("well_id", sort=False)[c]
        for lag in [1, 3, 6]:
            out[f"{c}_lag{lag}"] = group.shift(lag)
        for stat in ["mean", "std"]:
            out[f"{c}_roll_{stat}_12"] = group.transform(
                lambda s, stat=stat: getattr(s.shift(1).rolling(12, min_periods=4), stat)()
            )
        out[f"{c}_diff1"] = out[c] - group.shift(1)
    out = add_control_features(out)
    out["pseudo_mse"] = out["energy_input_proxy"] / (out["speed"] + 1e-6)
    out["log_pseudo_mse"] = np.log1p(out["pseudo_mse"])
    out["log_pseudo_mse_roll_median_30"] = out.groupby("well_id", sort=False)["log_pseudo_mse"].transform(
        lambda s: s.rolling(30, min_periods=10).median()
    )
    return out


def future_sum(frame: pd.DataFrame, column: str, horizon: int) -> pd.Series:
    return frame.groupby("well_id", sort=False)[column].transform(
        lambda s: s.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    )


def add_targets(frame: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    out = frame.copy()
    out["future_depth_m"] = future_sum(out, "d_depth", horizon)
    out["future_time_sec"] = future_sum(out, "d_time", horizon)
    out["_rotation_time"] = out["rotation"] * out["d_time"]
    out["target_speed_near5"] = out["future_depth_m"] / out["future_time_sec"]
    out["target_rotation_near5"] = future_sum(out, "_rotation_time", horizon) / out["future_time_sec"]
    return out.drop(columns="_rotation_time")


@dataclass(frozen=True)
class EnergyCalibration:
    mean: float
    std: float
    cutpoints: list[float]

    @classmethod
    def fit(cls, training_history: pd.DataFrame) -> "EnergyCalibration":
        values = training_history["log_pseudo_mse_roll_median_30"].dropna()
        if len(values) < 4:
            raise ValueError("Insufficient training history for energy calibration.")
        return cls(float(values.mean()), max(float(values.std()), EPS),
                   [float(x) for x in values.quantile([.25, .5, .75])])

    def transform(self, history: pd.DataFrame) -> pd.DataFrame:
        out = history.copy()
        values = out["log_pseudo_mse_roll_median_30"]
        out["hardness_score_smooth"] = (values - self.mean) / self.std
        codes = np.searchsorted(self.cutpoints, values.to_numpy(), side="left")
        labels = np.asarray(ENERGY_LABELS_4, dtype=object)[codes]
        labels[values.isna().to_numpy()] = "unknown"
        out["rock_energy_type_final"] = labels
        return out

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "schema_version": SCHEMA_VERSION,
                "method": "trailing_30_median_current_quantile_train_only",
                "window": 30, "min_periods": 10, "labels": ENERGY_LABELS_4,
                "fit_scope": "training_wells_only"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnergyCalibration":
        return cls(data["mean"], data["std"], data["cutpoints"])


def pressure_ranges(training: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for label, part in [("__global__", training), *list(training.groupby("rock_energy_type_final"))]:
        result[label] = {f"{c}_q{q:02d}": float(part[c].quantile(q / 100))
                         for c in ["pressure_axis", "pressure_rotation"] for q in [5, 95]}
    return result


def candidate_grid(rows: pd.DataFrame, ranges: dict, grid_size: int = GRID_SIZE) -> pd.DataFrame:
    """Rectangular empirical bounds intersect +/-8%; first candidate is always no change.

    An empty intersection falls back to no change, never to a broader search.
    These empirical limits are not a substitute for equipment safety interlocks.
    """
    rows = rows.reset_index(drop=True)
    aa, rr = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), indexing="ij")
    per_row = grid_size ** 2 + 1
    candidate_columns = list(dict.fromkeys(ROTATION_FEATURES + ["pressure_rotation_lag1"]))
    grid = rows[candidate_columns].loc[rows.index.repeat(per_row)].reset_index(drop=True)
    grid["row_id"] = np.repeat(np.arange(len(rows)), per_row)
    grid["candidate_id"] = np.tile(np.arange(per_row), len(rows))
    bounds = {}
    valid = np.ones(len(rows), dtype=bool)
    for c in ["pressure_axis", "pressure_rotation"]:
        current = rows[c].to_numpy(dtype=float)
        limits = [ranges.get(label, ranges["__global__"]) for label in rows["rock_energy_type_final"]]
        low = np.maximum(current * (1 - MAX_DELTA_FRAC), [r[f"{c}_q05"] for r in limits])
        high = np.minimum(current * (1 + MAX_DELTA_FRAC), [r[f"{c}_q95"] for r in limits])
        valid &= low <= high
        bounds[c] = current, low, high
    for c, fractions in [("pressure_axis", aa.ravel()), ("pressure_rotation", rr.ravel())]:
        current, low, high = bounds[c]
        proposed = low[:, None] + fractions * (high - low)[:, None]
        proposed[~valid] = current[~valid, None]
        values = np.column_stack([current, proposed]).ravel()
        grid[c] = values
        grid[f"delta_{c}_frac"] = (values - np.repeat(current, per_row)) / np.repeat(current, per_row)
    return add_control_features(grid)


def predict_chain(rows, rotation_model, speed_model) -> np.ndarray:
    speed_input = rows.copy()
    speed_input["candidate_target_rotation"] = rotation_model.predict(rows[ROTATION_FEATURES])
    return speed_model.predict(speed_input[SPEED_FEATURES])


def score_candidates(grid: pd.DataFrame, current_predictions: np.ndarray) -> pd.DataFrame:
    out = grid.copy()
    out["current_predicted_target_speed"] = current_predictions[out["row_id"].to_numpy(dtype=int)]
    scale = np.maximum(np.abs(out["current_predicted_target_speed"].to_numpy()), EPS)
    ax = out["delta_pressure_axis_frac"].abs().to_numpy() / MAX_DELTA_FRAC
    rot = out["delta_pressure_rotation_frac"].abs().to_numpy() / MAX_DELTA_FRAC
    out["change_penalty"] = CHANGE_PENALTY_WEIGHT * scale * (ax ** 2 + rot ** 2)
    edge_ax = np.maximum(0, (ax - BOUNDARY_START) / (1 - BOUNDARY_START))
    edge_rot = np.maximum(0, (rot - BOUNDARY_START) / (1 - BOUNDARY_START))
    out["boundary_penalty"] = BOUNDARY_PENALTY_WEIGHT * scale * (edge_ax ** 2 + edge_rot ** 2)
    out["optimizer_score"] = out["predicted_target_speed"] - out["change_penalty"] - out["boundary_penalty"]
    return out


def recommend(rows, rotation_model, speed_model, ranges, grid_size=GRID_SIZE):
    grid = candidate_grid(rows, ranges, grid_size)
    grid["candidate_target_rotation"] = rotation_model.predict(grid[ROTATION_FEATURES])
    grid["predicted_target_speed"] = speed_model.predict(grid[SPEED_FEATURES])
    # Infer the baseline from the exact same candidate batch to avoid numerical drift.
    current = grid.loc[grid["candidate_id"] == 0, "predicted_target_speed"].to_numpy()
    grid = score_candidates(grid, current)
    selected = grid.loc[grid.groupby("row_id", sort=False)["optimizer_score"].idxmax()].copy()
    if (selected["optimizer_score"] < current - 1e-12).any():
        raise AssertionError("Recommendation is worse than the explicit no-change candidate.")
    return selected.reset_index(drop=True), grid


def paired_time_summary(depth, operator_time, current_speed, recommended_speed):
    """Model scenarios and residual-anchored diagnostics; no measured causal effect."""
    depth, operator_time, current_speed, recommended_speed = map(
        lambda a: np.asarray(a, dtype=float), [depth, operator_time, current_speed, recommended_speed]
    )
    if any((a <= 0).any() or not np.isfinite(a).all()
           for a in [depth, operator_time, current_speed, recommended_speed]):
        raise ValueError("Time diagnostics require positive finite quantities.")
    current_time = depth / current_speed
    rec_time = depth / recommended_speed
    anchored_time = operator_time * current_speed / recommended_speed
    return pd.DataFrame({"operator_block_time_sec": operator_time,
                         "current_model_time_sec": current_time, "model_block_time_sec": rec_time,
                         "anchored_model_time_sec": anchored_time,
                         "model_based_predicted_uplift_pct": 100 * (recommended_speed / current_speed - 1),
                         "paired_time_saved_pct": 100 * (1 - rec_time / current_time),
                         "raw_time_saved_vs_operator_pct": 100 * (1 - rec_time / operator_time)})
