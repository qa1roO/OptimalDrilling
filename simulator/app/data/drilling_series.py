import csv
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIMULATOR_ROOT = Path(__file__).resolve().parents[2]
SIMULATOR_CORE_DIR = SIMULATOR_ROOT / "simulator_core"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASET_PATHS = (
    DATASETS_DIR / "dataset_1.csv",
    DATASETS_DIR / "dataset_2.csv",
)
KMEANS_PATH = SIMULATOR_CORE_DIR / "kmeans_k7.joblib"
SCALER_PATH = SIMULATOR_CORE_DIR / "scaler_k7.joblib"
OPTIMALS_PATH = SIMULATOR_CORE_DIR / "optimals.csv"
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


def _parse_dataset_row(row: dict[str, str]) -> dict[str, float] | None:
    try:
        parsed = {
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
    rows: list[dict[str, float]],
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
    rows: list[dict[str, float]],
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
