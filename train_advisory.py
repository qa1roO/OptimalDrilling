"""Reproduce training and offline diagnostics: python train_advisory.py --backend lightgbm.

Use --backend sklearn explicitly when LightGBM is unavailable. No test-based tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import drilling_advisory as core
from config import (PROJECT_ROOT, RAW_DATA_PATH, LABELED_DATA_PATH, RANDOM_STATE,
                    DRILLING_ADVISORY_ARTIFACT_DIR as ARTIFACT_DIR,
                    DRILLING_ADVISORY_REPORT_DIR as REPORT_DIR,
                    SIMULATOR_ADVISORY_ARTIFACT_DIR as SIM_ARTIFACT_DIR)


def make_model(backend, numeric, categorical, seed, iterations=650):
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
                              ("num", "passthrough", numeric)], remainder="drop")
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        reg = LGBMRegressor(n_estimators=iterations, learning_rate=.03, num_leaves=63,
                           subsample=.9, colsample_bytree=.9, random_state=seed,
                           n_jobs=6, objective="regression", verbosity=-1)
    else:
        reg = HistGradientBoostingRegressor(max_iter=iterations, learning_rate=.03,
                                           max_leaf_nodes=63, min_samples_leaf=20,
                                           early_stopping=False, random_state=seed)
    return Pipeline([("preprocess", prep), ("model", reg)])


def metrics(name, truth, prediction):
    return {"horizon": 5, "model": name, "MAE": float(mean_absolute_error(truth, prediction)),
            "RMSE": float(root_mean_squared_error(truth, prediction)),
            "R2": float(r2_score(truth, prediction))}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_global_surface(train):
    """Empirical visual background built ONLY from training rows; not a prediction model."""
    from scipy.ndimage import gaussian_filter
    x = train["pressure_axis"].quantile([.05, .95]).to_numpy()
    y = train["pressure_rotation"].quantile([.05, .95]).to_numpy()
    xe, ye = np.linspace(*x, 41), np.linspace(*y, 41)
    part = train.loc[train.pressure_axis.between(*x) & train.pressure_rotation.between(*y),
                     ["pressure_axis", "pressure_rotation", "speed"]].copy()
    part["x"] = np.clip(np.digitize(part.pressure_axis, xe) - 1, 0, 39)
    part["y"] = np.clip(np.digitize(part.pressure_rotation, ye) - 1, 0, 39)
    groups = part.groupby(["x", "y"]).speed.agg(["median", "count"])
    z, weights = np.zeros((40, 40)), np.zeros((40, 40))
    for (i, j), row in groups.iterrows():
        if row["count"] >= 5:
            z[i, j], weights[i, j] = row["median"], 1
    numerator = gaussian_filter(z, 1.6)
    denominator = gaussian_filter(weights, 1.6)
    smooth = np.full((40, 40), float(train.speed.median()))
    np.divide(numerator, denominator, out=smooth, where=denominator > 1e-9)
    xx, yy = np.meshgrid((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2, indexing="ij")
    write_json(PROJECT_ROOT / "notebooks/surface_data/global_speed_surface.json",
               {"source": "training_only_empirical_speed_surface", "x": xx.tolist(),
                "y": yy.tolist(), "z": smooth.tolist(),
                "metadata": {"fit_scope": "training_wells_only", "rows": len(train),
                             "interpretation": "empirical visualization; not predicted speed"}})


def evaluate_blocks(test, rotation_model, speed_model, ranges, bootstrap_reps=2000):
    # Positions refer to the complete raw history. Dropping warm-up/target rows must
    # never renumber physical future intervals or shift the forecast horizon.
    anchors = test.loc[test["_row_in_well"] % 5 == 4].copy().reset_index(drop=True)
    pieces = []
    for start in range(0, len(anchors), 128):
        part = anchors.iloc[start:start+128].copy().reset_index(drop=True)
        best, _ = core.recommend(part, rotation_model, speed_model, ranges)
        result = part[["well_id", "processing_time", "_row_in_well", "rock_energy_type_final",
                       "future_depth_m", "future_time_sec", "target_speed_near5",
                       "pressure_axis", "pressure_rotation", "pressure_axis_lag1", "pressure_rotation_lag1"]].copy()
        result["recommended_pressure_axis"] = best.pressure_axis
        result["recommended_pressure_rotation"] = best.pressure_rotation
        result["recommended_pressure_axis_diff1"] = best.pressure_axis_diff1
        result["recommended_pressure_rotation_diff1"] = best.pressure_rotation_diff1
        result["current_predicted_target_speed"] = best.current_predicted_target_speed
        result["recommended_predicted_speed"] = best.predicted_target_speed
        result["optimizer_score"] = best.optimizer_score
        result["candidate_id"] = best.candidate_id
        result["changed_action"] = ((best.pressure_axis != part.pressure_axis)
                                    | (best.pressure_rotation != part.pressure_rotation))
        diagnostic = core.paired_time_summary(result.future_depth_m, result.future_time_sec,
                                              best.current_predicted_target_speed, best.predicted_target_speed)
        pieces.append(pd.concat([result, diagnostic], axis=1))
        if start % 2048 == 0:
            print(f"Advisory blocks: {min(start+128, len(anchors))}/{len(anchors)}", flush=True)
    blocks = pd.concat(pieces, ignore_index=True)
    wells = blocks.groupby("well_id", sort=True).agg(
        non_overlap_blocks=("processing_time", "size"), total_depth_modeled_m=("future_depth_m", "sum"),
        operator_time_sec=("operator_block_time_sec", "sum"),
        current_model_time_sec=("current_model_time_sec", "sum"), model_time_sec=("model_block_time_sec", "sum"),
        anchored_model_time_sec=("anchored_model_time_sec", "sum"), changed_action_rate=("changed_action", "mean"),
    ).reset_index()
    wells["paired_time_saved_pct"] = 100 * (1 - wells.model_time_sec / wells.current_model_time_sec)
    wells["paired_equivalent_speed_gain_pct"] = 100 * (wells.current_model_time_sec / wells.model_time_sec - 1)
    wells["anchored_time_saved_pct"] = 100 * (1 - wells.anchored_model_time_sec / wells.operator_time_sec)
    wells["raw_time_saved_vs_operator_pct"] = 100 * (1 - wells.model_time_sec / wells.operator_time_sec)
    # Explicit threshold, separate from strictly positive outcomes.
    v = wells.paired_time_saved_pct
    wells["effect_group"] = np.select([v < -3, v <= 3, v <= 10],
                                      ["negative_below_minus3", "neutral_minus3_to3", "positive_3_to10"],
                                      default="positive_above10")
    baseline, recommended = wells.current_model_time_sec.sum(), wells.model_time_sec.sum()
    operator, anchored = wells.operator_time_sec.sum(), wells.anchored_model_time_sec.sum()
    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    values = wells[["current_model_time_sec", "model_time_sec"]].to_numpy()
    for _ in range(bootstrap_reps):
        sums = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        boot.append(100*(1-sums[1]/sums[0]))
    summary = {
        "horizon": 5, "method": "paired_model_non_overlap_near5", "wells": len(wells),
        "non_overlap_blocks": len(blocks), "telemetry_rows_covered": 5*len(blocks),
        "operator_time_total_min": float(operator/60), "current_model_time_total_min": float(baseline/60),
        "model_time_total_min": float(recommended/60),
        "paired_model_time_saved_pct": float(100*(1-recommended/baseline)),
        "paired_model_equivalent_speed_gain_pct": float(100*(baseline/recommended-1)),
        "anchored_scenario_time_saved_pct": float(100*(1-anchored/operator)),
        "raw_equivalent_speed_vs_operator_pct_DIAGNOSTIC_ONLY": float(100*(operator/recommended-1)),
        "no_change_raw_equivalent_speed_vs_operator_pct_DIAGNOSTIC_ONLY": float(100*(operator/baseline-1)),
        "positive_wells_above_zero": int((v > 1e-10).sum()),
        "positive_wells_above_3pct": int((v > 3).sum()),
        "changed_action_rate": float(blocks.changed_action.mean()),
        "paired_time_saved_well_bootstrap_p025": float(np.quantile(boot, .025)),
        "paired_time_saved_well_bootstrap_p975": float(np.quantile(boot, .975)),
        "interpretation": "model-internal scenarios; observational data; not measured causal uplift",
    }
    energy = blocks.groupby("rock_energy_type_final").agg(
        blocks=("well_id", "size"), current_model_time_sec=("current_model_time_sec", "sum"),
        recommended_model_time_sec=("model_block_time_sec", "sum"), changed_action_rate=("changed_action", "mean"),
    ).reset_index()
    energy["paired_time_saved_pct"] = 100*(1-energy.recommended_model_time_sec/energy.current_model_time_sec)
    return blocks, wells, pd.DataFrame([summary]), energy


def run_training(backend="lightgbm", iterations=650):
    started = time.time()
    for path in [ARTIFACT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    history = core.prepare_history(pd.read_csv(RAW_DATA_PATH))
    data = core.add_targets(history)
    required = [c for c in core.NUMERIC_FEATURES if c != "hardness_score_smooth"]
    required += ["log_pseudo_mse_roll_median_30", "target_speed_near5", "target_rotation_near5"]
    eligible = data.dropna(subset=required).copy()
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=RANDOM_STATE)
                               .split(eligible, groups=eligible.well_id))
    train_raw, test_raw = eligible.iloc[train_idx].copy(), eligible.iloc[test_idx].copy()
    assert set(train_raw.well_id).isdisjoint(test_raw.well_id)
    calibration = core.EnergyCalibration.fit(history.loc[history.well_id.isin(train_raw.well_id)])
    train, test = calibration.transform(train_raw), calibration.transform(test_raw)
    print(f"Backend={backend}; train={len(train)} ({train.well_id.nunique()} wells); "
          f"test={len(test)} ({test.well_id.nunique()} wells)", flush=True)
    oof = np.full(len(train), np.nan)
    folds = np.full(len(train), -1)
    # The speed model sees group-out-of-fold rotation predictions, never future rotation labels.
    for fold, (fit_idx, hold_idx) in enumerate(GroupKFold(n_splits=3).split(train, groups=train.well_id)):
        fit_raw, hold_raw = train_raw.iloc[fit_idx], train_raw.iloc[hold_idx]
        fit_wells = set(fit_raw.well_id)
        fold_cal = core.EnergyCalibration.fit(history.loc[history.well_id.isin(fit_wells)])
        fit, hold = fold_cal.transform(fit_raw), fold_cal.transform(hold_raw)
        assert fit_wells.isdisjoint(hold.well_id)
        model = make_model(backend, core.NUMERIC_FEATURES, core.CATEGORICAL_FEATURES,
                           RANDOM_STATE+5, iterations)
        print(f"Rotation OOF fold {fold+1}/3", flush=True)
        model.fit(fit[core.ROTATION_FEATURES], fit.target_rotation_near5)
        oof[hold_idx] = model.predict(hold[core.ROTATION_FEATURES])
        folds[hold_idx] = fold
    assert np.isfinite(oof).all()
    rotation_model = make_model(backend, core.NUMERIC_FEATURES, core.CATEGORICAL_FEATURES, RANDOM_STATE+5, iterations)
    print("Fitting final rotation model", flush=True)
    rotation_model.fit(train[core.ROTATION_FEATURES], train.target_rotation_near5)
    speed_train = train.copy()
    speed_train["candidate_target_rotation"] = oof
    speed_model = make_model(backend, core.SPEED_NUMERIC_FEATURES, core.CATEGORICAL_FEATURES,
                            RANDOM_STATE+105, iterations)
    print("Fitting speed model on OOF rotation", flush=True)
    speed_model.fit(speed_train[core.SPEED_FEATURES], train.target_speed_near5)
    rotation_pred = rotation_model.predict(test[core.ROTATION_FEATURES])
    speed_pred = core.predict_chain(test, rotation_model, speed_model)
    metric_report = pd.DataFrame([
        metrics("rotation_model_near5", test.target_rotation_near5, rotation_pred),
        metrics("speed_model_chained_near5", test.target_speed_near5, speed_pred),
    ])
    print(metric_report.to_string(index=False), flush=True)
    baselines = pd.DataFrame([
        metrics("current_speed", test.target_speed_near5, test.speed),
        metrics("speed_roll_mean_12", test.target_speed_near5, test.speed_roll_mean_12),
        metrics("rotation_to_speed_chained_near5", test.target_speed_near5, speed_pred),
    ])
    ranges = core.pressure_ranges(train)
    feature_config = {"schema_version": core.SCHEMA_VERSION, "backend": backend, "target_horizons": [5],
                      "default_horizon": 5, "base_numeric_features": core.NUMERIC_FEATURES,
                      "speed_numeric_features": core.SPEED_NUMERIC_FEATURES,
                      "categorical_features": core.CATEGORICAL_FEATURES,
                      "rotation_features": core.ROTATION_FEATURES, "speed_features": core.SPEED_FEATURES,
                      "energy_type_column": "rock_energy_type_final", "speed_extra_features": ["candidate_target_rotation"],
                      "required_live_columns": core.RAW_COLUMNS,
                      "target_speed_formula": "sum(future d_depth)/sum(future d_time)",
                      "speed_training_rotation": "three-fold GroupKFold OOF predictions"}
    optimizer_config = {"schema_version": core.SCHEMA_VERSION, "grid_size_default": core.GRID_SIZE,
                        "max_delta_frac_default": core.MAX_DELTA_FRAC, "change_penalty_weight": core.CHANGE_PENALTY_WEIGHT,
                        "boundary_penalty_weight": core.BOUNDARY_PENALTY_WEIGHT, "boundary_start": core.BOUNDARY_START,
                        "no_change_candidate": "always first; wins exact ties",
                        "empty_support_intersection": "no_change_only",
                        "operator_comparison_interpretation": "model scenarios; not measured causal effect"}
    joblib.dump(rotation_model, ARTIFACT_DIR / "rotation_model_near5.joblib")
    joblib.dump(speed_model, ARTIFACT_DIR / "speed_model_near5.joblib")
    write_json(ARTIFACT_DIR / "feature_config.json", feature_config)
    write_json(ARTIFACT_DIR / "optimizer_config.json", optimizer_config)
    write_json(ARTIFACT_DIR / "energy_calibration.json", calibration.to_dict())
    write_json(ARTIFACT_DIR / "candidate_pressure_ranges_by_energy_type.json", ranges)
    write_json(PROJECT_ROOT / "notebooks/energy_segmentation_config.json", calibration.to_dict())
    # Preserve all original rows including warm-up, so replay reconstructs the same history.
    replay = calibration.transform(history)
    replay["split"] = np.where(replay.well_id.isin(train.well_id), "train", "test")
    replay.to_csv(LABELED_DATA_PATH, index=False)
    build_global_surface(train)
    blocks, wells, summary, energy = evaluate_blocks(test, rotation_model, speed_model, ranges)
    for filename, frame in [("near5_model_metrics.csv", metric_report), ("near5_baseline_metrics.csv", baselines),
                            ("near5_block_offline_evaluation.csv", blocks), ("near5_well_offline_evaluation.csv", wells),
                            ("near5_offline_summary.csv", summary), ("near5_energy_offline_evaluation.csv", energy)]:
        frame.to_csv(REPORT_DIR / filename, index=False)
    pd.DataFrame({"well_id": train.well_id.to_numpy(), "processing_time": train.processing_time.to_numpy(),
                  "fold": folds, "rotation_oof_prediction": oof}).to_csv(REPORT_DIR / "near5_rotation_oof.csv", index=False)
    split = {"random_state": RANDOM_STATE, "test_size": .25, "train_wells": sorted(set(train.well_id)),
             "test_wells": sorted(set(test.well_id))}
    write_json(REPORT_DIR / "split_manifest.json", split)
    manifest = {"schema_version": core.SCHEMA_VERSION, "backend": backend, "iterations": iterations,
                "rows": len(history), "train_rows": len(train), "test_rows": len(test),
                "train_wells": train.well_id.nunique(), "test_wells": test.well_id.nunique(),
                "energy_fit_wells": sorted(set(train.well_id)),
                "source_sha256": hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest(),
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), PROJECT_ROOT / "drilling_advisory.py"]},
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]},
                "elapsed_seconds": time.time()-started,
                "limitations": ["Observational response model; causal intervention effect unvalidated",
                                "Empirical pressure bounds are not equipment safety specifications",
                                "Fixed hyperparameters; test set not used for tuning",
                                "Bootstrap reflects sampling of wells, not model or causal uncertainty"]}
    write_json(ARTIFACT_DIR / "training_manifest.json", manifest)
    SIM_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for p in ARTIFACT_DIR.iterdir():
        if p.is_file():
            shutil.copy2(p, SIM_ARTIFACT_DIR / p.name)
    print(summary.to_string(index=False), flush=True)
    print(f"Completed in {time.time()-started:.1f}s; simulator artifacts synchronized", flush=True)
    return {"metrics": metric_report, "baselines": baselines, "summary": summary,
            "wells": wells, "energy": energy, "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["lightgbm", "sklearn"], default="lightgbm")
    parser.add_argument("--iterations", type=int, default=650)
    args = parser.parse_args()
    run_training(args.backend, args.iterations)
