import csv
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASET_PATHS = (
    DATASETS_DIR / "dataset_1.csv",
    DATASETS_DIR / "dataset_2.csv",
)
KMEANS_PATH = Path(r"C:\Users\stas2\Downloads\kmeans_k7.joblib")
SCALER_PATH = Path(r"C:\Users\stas2\Downloads\scaler_k7.joblib")
OPTIMALS_PATH = Path(r"C:\Users\stas2\Downloads\optimals.csv")
FEATURE_COLUMNS = ("pressure_axis", "pressure_rotation", "rotation", "speed")


@dataclass(frozen=True)
class DrillingPoint:
    time_s: float
    rotation: float
    speed: float
    source: str = "model"
    cluster_id: int | None = None


@dataclass(frozen=True)
class ClusterProfile:
    cluster_id: int
    avg_rotation: float
    avg_speed: float
    optimal_rotation: float
    optimal_speed: float


def build_default_transition_series() -> list[DrillingPoint]:
    """Demo trajectory: each segment moves from layer average values to target values."""
    layer_segments = [
        (96.0, 0.010, 112.0, 0.015),
        (104.0, 0.012, 128.0, 0.019),
        (92.0, 0.008, 108.0, 0.013),
        (118.0, 0.011, 136.0, 0.017),
    ]

    points: list[DrillingPoint] = []
    time_s = 0.0
    steps_per_layer = 10

    for start_rotation, start_speed, target_rotation, target_speed in layer_segments:
        for step in range(steps_per_layer):
            k = step / (steps_per_layer - 1)
            points.append(
                DrillingPoint(
                    time_s=time_s,
                    rotation=_lerp(start_rotation, target_rotation, k),
                    speed=_lerp(start_speed, target_speed, k),
                    source="model",
                )
            )
            time_s += 4.0

    return points


def build_dataset_emulator_series(
    dataset_path: Path | None = None,
    max_points: int = 72,
) -> tuple[list[DrillingPoint], list[DrillingPoint]]:
    """Build dataset-vs-model lines from read-only CSV files and ready-made clustering models."""
    candidate_paths = (dataset_path,) if dataset_path is not None else DATASET_PATHS
    dataset_points = []
    for candidate_path in candidate_paths:
        if candidate_path is None:
            continue
        dataset_points = _load_first_well_points(candidate_path, max_points=max_points)
        if dataset_points:
            break

    if not dataset_points:
        dataset_points = build_default_transition_series()

    profiles = load_cluster_profiles()
    model_points = []
    for point in dataset_points:
        if point.cluster_id is None:
            model_points.append(point)
            continue
        profile = profiles.get(point.cluster_id)
        if profile is None:
            model_points.append(
                DrillingPoint(
                    time_s=point.time_s,
                    rotation=point.rotation,
                    speed=point.speed,
                    source="model",
                    cluster_id=point.cluster_id,
                )
            )
            continue
        model_points.append(
            DrillingPoint(
                time_s=point.time_s,
                rotation=profile.optimal_rotation,
                speed=profile.optimal_speed,
                source="model",
                cluster_id=point.cluster_id,
            )
        )

    return dataset_points, model_points


def _load_first_well_points(dataset_path: Path, max_points: int) -> list[DrillingPoint]:
    if not dataset_path.exists():
        return []

    selected_well_id: str | None = None
    raw_points: list[dict[str, float | str]] = []

    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            well_id = row.get("well_id")
            if not well_id:
                continue
            if selected_well_id is None:
                selected_well_id = well_id
            if well_id != selected_well_id:
                if len(raw_points) >= 24:
                    break
                selected_well_id = well_id
                raw_points.clear()

            parsed = _parse_dataset_row(row)
            if parsed is not None:
                raw_points.append(parsed)
            if len(raw_points) >= max_points:
                break

    if not raw_points:
        return []

    start_time = _parse_time(str(raw_points[0]["time"]))
    clusters = _predict_clusters(raw_points)
    points: list[DrillingPoint] = []
    for idx, row in enumerate(raw_points):
        time_raw = str(row["time"])
        current_time = _parse_time(time_raw)
        if start_time is not None and current_time is not None:
            time_s = max((current_time - start_time).total_seconds(), 0.0)
        else:
            time_s = idx * 5.0
        points.append(
            DrillingPoint(
                time_s=time_s,
                rotation=float(row["rotation"]),
                speed=float(row["speed"]),
                source="dataset",
                cluster_id=clusters[idx] if idx < len(clusters) else None,
            )
        )

    return points


@lru_cache(maxsize=1)
def load_cluster_profiles() -> dict[int, ClusterProfile]:
    averages = _load_average_points_by_cluster()
    optimals = _load_optimal_points_by_cluster()
    global_average = _load_global_average_point()
    cluster_ids = sorted(set(averages) | set(optimals))
    profiles: dict[int, ClusterProfile] = {}

    for cluster_id in cluster_ids:
        average = averages.get(cluster_id, global_average)
        optimal = optimals.get(cluster_id, average)
        profiles[cluster_id] = ClusterProfile(
            cluster_id=cluster_id,
            avg_rotation=average["rotation"],
            avg_speed=average["speed"],
            optimal_rotation=optimal["rotation"],
            optimal_speed=optimal["speed"],
        )

    return profiles


def _parse_dataset_row(row: dict[str, str]) -> dict[str, float | str] | None:
    try:
        time_raw = row.get("time", "")
        parsed = {
            "time": time_raw,
            "pressure_axis": float(row.get("pressure_axis", "")),
            "pressure_rotation": float(row.get("pressure_rotation", "")),
            "rotation": float(row.get("rotation", "")),
            "speed": float(row.get("speed", "")),
        }
    except ValueError:
        return None

    if any(float(parsed[column]) <= 0 for column in FEATURE_COLUMNS):
        return None
    return parsed


@lru_cache(maxsize=1)
def _load_ready_models():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        scaler = joblib.load(SCALER_PATH)
        kmeans = joblib.load(KMEANS_PATH)
    return scaler, kmeans


@lru_cache(maxsize=1)
def _load_optimal_points_by_cluster() -> dict[int, dict[str, float]]:
    pressure_targets = _load_optimal_pressure_targets()
    if pressure_targets:
        nearest = _load_nearest_points_to_pressure_targets(pressure_targets)
        if nearest:
            return nearest

    optimal_by_cluster: dict[int, dict[str, float]] = {}
    scaler, kmeans = _load_ready_models()

    for dataset_path in DATASET_PATHS:
        if not dataset_path.exists():
            continue

        with dataset_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            chunk: list[dict[str, float | str]] = []
            for row in reader:
                parsed = _parse_dataset_row(row)
                if parsed is None:
                    continue
                chunk.append(parsed)
                if len(chunk) >= 4096:
                    _update_optimal_clusters(chunk, scaler, kmeans, optimal_by_cluster)
                    chunk.clear()
            if chunk:
                _update_optimal_clusters(chunk, scaler, kmeans, optimal_by_cluster)

    return optimal_by_cluster


@lru_cache(maxsize=1)
def _load_average_points_by_cluster() -> dict[int, dict[str, float]]:
    scaler, kmeans = _load_ready_models()
    cluster_count = int(getattr(kmeans, "n_clusters", 7))
    speed_sums = np.zeros(cluster_count, dtype=float)
    rotation_sums = np.zeros(cluster_count, dtype=float)
    counts = np.zeros(cluster_count, dtype=int)

    for chunk in _iter_valid_feature_chunks():
        features = chunk.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            clusters = kmeans.predict(scaler.transform(features)).astype(int)
        valid_mask = (clusters >= 0) & (clusters < cluster_count)
        clusters = clusters[valid_mask]
        np.add.at(speed_sums, clusters, chunk["speed"].to_numpy(dtype=float)[valid_mask])
        np.add.at(rotation_sums, clusters, chunk["rotation"].to_numpy(dtype=float)[valid_mask])
        np.add.at(counts, clusters, 1)

    return {
        cluster_id: {
            "speed": float(speed_sums[cluster_id] / counts[cluster_id]),
            "rotation": float(rotation_sums[cluster_id] / counts[cluster_id]),
        }
        for cluster_id in range(cluster_count)
        if counts[cluster_id] > 0
    }


def _update_average_clusters(
    rows: list[dict[str, float | str]],
    scaler,
    kmeans,
    sums: dict[int, dict[str, float]],
    counts: dict[int, int],
) -> None:
    clusters = _predict_clusters(rows, scaler=scaler, kmeans=kmeans)
    for row, cluster_id in zip(rows, clusters):
        sums.setdefault(cluster_id, {"speed": 0.0, "rotation": 0.0})
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
        sums[cluster_id]["speed"] += float(row["speed"])
        sums[cluster_id]["rotation"] += float(row["rotation"])


def _load_optimal_pressure_targets() -> dict[int, dict[str, float]]:
    if not OPTIMALS_PATH.exists():
        return {}

    targets: dict[int, dict[str, float]] = {}
    with OPTIMALS_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                cluster_id = int(row["cluster"])
                targets[cluster_id] = {
                    "pressure_axis": float(row["opt_pax"]),
                    "pressure_rotation": float(row["opt_prot"]),
                }
            except (KeyError, ValueError):
                continue
    return targets


def _load_nearest_points_to_pressure_targets(
    targets: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    nearest: dict[int, dict[str, float]] = {}

    for chunk in _iter_valid_feature_chunks():
        axis = chunk["pressure_axis"].to_numpy(dtype=float)
        pressure_rotation = chunk["pressure_rotation"].to_numpy(dtype=float)
        speeds = chunk["speed"].to_numpy(dtype=float)
        rotations = chunk["rotation"].to_numpy(dtype=float)

        for cluster_id, target in targets.items():
            distances = (
                (axis - target["pressure_axis"]) ** 2
                + (pressure_rotation - target["pressure_rotation"]) ** 2
            )
            idx = int(np.argmin(distances))
            distance = float(distances[idx])
            current = nearest.get(cluster_id)
            if current is None or distance < current["distance"]:
                nearest[cluster_id] = {
                    "distance": distance,
                    "speed": float(speeds[idx]),
                    "rotation": float(rotations[idx]),
                }

    return {
        cluster_id: {"speed": item["speed"], "rotation": item["rotation"]}
        for cluster_id, item in nearest.items()
    }


def _update_nearest_targets(
    rows: list[dict[str, float | str]],
    targets: dict[int, dict[str, float]],
    nearest: dict[int, dict[str, float]],
) -> None:
    for row in rows:
        for cluster_id, target in targets.items():
            distance = (
                (float(row["pressure_axis"]) - target["pressure_axis"]) ** 2
                + (float(row["pressure_rotation"]) - target["pressure_rotation"]) ** 2
            )
            current = nearest.get(cluster_id)
            if current is None or distance < current["distance"]:
                nearest[cluster_id] = {
                    "distance": distance,
                    "speed": float(row["speed"]),
                    "rotation": float(row["rotation"]),
                }


@lru_cache(maxsize=1)
def _load_global_average_point() -> dict[str, float]:
    speed_sum = 0.0
    rotation_sum = 0.0
    count = 0

    for chunk in _iter_valid_feature_chunks():
        speed_sum += float(chunk["speed"].sum())
        rotation_sum += float(chunk["rotation"].sum())
        count += int(len(chunk))

    if count == 0:
        return {"speed": 0.012, "rotation": 100.0}
    return {"speed": speed_sum / count, "rotation": rotation_sum / count}


def _iter_valid_feature_chunks(chunksize: int = 100_000):
    for dataset_path in DATASET_PATHS:
        if not dataset_path.exists():
            continue

        for chunk in pd.read_csv(
            dataset_path,
            usecols=list(FEATURE_COLUMNS),
            chunksize=chunksize,
        ):
            for column in FEATURE_COLUMNS:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            chunk = chunk.dropna(subset=list(FEATURE_COLUMNS))
            valid_mask = np.ones(len(chunk), dtype=bool)
            for column in FEATURE_COLUMNS:
                valid_mask &= chunk[column].to_numpy(dtype=float) > 0
            if valid_mask.any():
                yield chunk.loc[valid_mask, list(FEATURE_COLUMNS)]


def _update_optimal_clusters(
    rows: list[dict[str, float | str]],
    scaler,
    kmeans,
    optimal_by_cluster: dict[int, dict[str, float]],
) -> None:
    clusters = _predict_clusters(rows, scaler=scaler, kmeans=kmeans)
    for row, cluster_id in zip(rows, clusters):
        speed = float(row["speed"])
        current_best = optimal_by_cluster.get(cluster_id)
        if current_best is None or speed > current_best["speed"]:
            optimal_by_cluster[cluster_id] = {
                "speed": speed,
                "rotation": float(row["rotation"]),
            }


def _predict_clusters(
    rows: list[dict[str, float | str]],
    scaler=None,
    kmeans=None,
) -> list[int]:
    if not rows:
        return []
    if scaler is None or kmeans is None:
        scaler, kmeans = _load_ready_models()

    features = np.array(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows],
        dtype=float,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        scaled = scaler.transform(features)
    return [int(cluster_id) for cluster_id in kmeans.predict(scaled)]


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _lerp(start: float, end: float, k: float) -> float:
    return start + (end - start) * k
