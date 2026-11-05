"""Evaluate recommendations with models fitted on disjoint wells.

uv run python evaluate_independent.py
Two predeclared directions: A selects / B evaluates, and vice versa. Test wells
are unchanged. The primary result averages model times by well across directions;
bootstrap resamples wells, never directions as independent observations.
This is a cross-model diagnostic, not off-policy identification of causal effects.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import drilling_advisory as core
from config import (PROJECT_ROOT, RAW_DATA_PATH, RANDOM_STATE,
                    DRILLING_ADVISORY_REPORT_DIR, DRILLING_ADVISORY_ARTIFACT_DIR,
                    SIMULATOR_ADVISORY_ARTIFACT_DIR)
from train_advisory import make_model, metrics, write_json

OUTPUT_DIR = DRILLING_ADVISORY_REPORT_DIR / "independent_evaluation"
TIME_COLUMNS = ["evaluator_current_time_sec", "evaluator_recommended_time_sec",
                "selector_current_time_sec", "selector_recommended_time_sec"]


def partition_wells(train_wells, test_wells, seed=RANDOM_STATE):
    train = sorted(map(str, train_wells))
    test = set(map(str, test_wells))
    if len(train) != len(set(train)) or set(train) & test or len(train) < 8:
        raise ValueError("Need unique training wells, disjoint test wells, and >=8 training wells.")
    order = np.random.default_rng(seed).permutation(np.asarray(train))
    midpoint = len(order)//2
    return {"A": sorted(order[:midpoint].tolist()), "B": sorted(order[midpoint:].tolist())}


@dataclass
class Branch:
    name: str
    calibration: core.EnergyCalibration
    rotation_model: object
    speed_model: object
    ranges: dict


def fit_branch(name, wells, history, eligible, output, backend="lightgbm", iterations=650):
    raw = eligible.loc[eligible.well_id.isin(wells)].copy()
    cal = core.EnergyCalibration.fit(history.loc[history.well_id.isin(wells)])
    data = cal.transform(raw)
    oof = np.full(len(raw), np.nan)
    fold_manifest = []
    # Each rotation prediction supplied to the speed model excludes that well,
    # including the fitted preprocessing statistics.
    for fold, (fit_idx, hold_idx) in enumerate(GroupKFold(3).split(raw, groups=raw.well_id)):
        fit_raw, hold_raw = raw.iloc[fit_idx], raw.iloc[hold_idx]
        fit_wells, hold_wells = set(fit_raw.well_id), set(hold_raw.well_id)
        assert fit_wells.isdisjoint(hold_wells)
        fold_cal = core.EnergyCalibration.fit(history.loc[history.well_id.isin(fit_wells)])
        fit, hold = fold_cal.transform(fit_raw), fold_cal.transform(hold_raw)
        rotation = make_model(backend, core.NUMERIC_FEATURES, core.CATEGORICAL_FEATURES,
                              RANDOM_STATE+5, iterations)
        print(f"{name}: rotation OOF fold {fold+1}/3", flush=True)
        rotation.fit(fit[core.ROTATION_FEATURES], fit.target_rotation_near5)
        oof[hold_idx] = rotation.predict(hold[core.ROTATION_FEATURES])
        fold_manifest.append({"fold": fold, "fit_wells": sorted(fit_wells), "holdout_wells": sorted(hold_wells)})
    assert np.isfinite(oof).all()
    rotation = make_model(backend, core.NUMERIC_FEATURES, core.CATEGORICAL_FEATURES,
                          RANDOM_STATE+5, iterations)
    print(f"{name}: final rotation and speed models ({len(wells)} wells / {len(data)} rows)", flush=True)
    rotation.fit(data[core.ROTATION_FEATURES], data.target_rotation_near5)
    speed = make_model(backend, core.SPEED_NUMERIC_FEATURES, core.CATEGORICAL_FEATURES,
                       RANDOM_STATE+105, iterations)
    data["candidate_target_rotation"] = oof
    speed.fit(data[core.SPEED_FEATURES], data.target_speed_near5)
    branch = Branch(name, cal, rotation, speed, core.pressure_ranges(data))
    folder = output / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(rotation, folder / "rotation_model.joblib")
    joblib.dump(speed, folder / "speed_model.joblib")
    write_json(folder / "energy_calibration.json", cal.to_dict())
    write_json(folder / "pressure_ranges.json", branch.ranges)
    write_json(folder / "training_wells_and_oof_folds.json", {"wells": wells, "folds": fold_manifest})
    return branch


def apply_selected_actions(context, selected):
    """Freeze evaluator's own observed state; substitute only the selector's controls."""
    out = context.reset_index(drop=True).copy()
    if len(out) != len(selected):
        raise ValueError("Selected actions and evaluation rows must align.")
    for column in ["pressure_axis", "pressure_rotation"]:
        out[column] = selected[column].to_numpy()
    return core.add_control_features(out)


def evaluate_selected_actions(context, selected, rotation_model, speed_model):
    """The evaluator scores two fixed actions. It never chooses, clips, or vetoes them."""
    current = core.add_control_features(context.reset_index(drop=True))
    recommended = apply_selected_actions(context, selected)
    both = pd.concat([current, recommended], ignore_index=True)
    prediction = core.predict_chain(both, rotation_model, speed_model)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Non-positive evaluator prediction: no clipping is allowed in this experiment.")
    return prediction[:len(context)], prediction[len(context):]


def in_empirical_bounds(context, actions, ranges):
    inside = np.ones(len(context), dtype=bool)
    for i, (_, row) in enumerate(context.iterrows()):
        bounds = ranges.get(row.rock_energy_type_final, ranges["__global__"])
        for c in ["pressure_axis", "pressure_rotation"]:
            inside[i] &= bounds[f"{c}_q05"] <= float(actions.iloc[i][c]) <= bounds[f"{c}_q95"]
    return inside


def evaluate_direction(selector, evaluator, test_raw, chunk_size=128):
    anchors = test_raw.loc[test_raw._row_in_well % 5 == 4].reset_index(drop=True)
    selector_rows = selector.calibration.transform(anchors)
    evaluator_rows = evaluator.calibration.transform(anchors)
    direction = f"{selector.name}_selects_{evaluator.name}_evaluates"
    parts = []
    for start in range(0, len(anchors), chunk_size):
        chosen_context = selector_rows.iloc[start:start+chunk_size].reset_index(drop=True)
        evaluation_context = evaluator_rows.iloc[start:start+chunk_size].reset_index(drop=True)
        selected, _ = core.recommend(chosen_context, selector.rotation_model, selector.speed_model, selector.ranges)
        current, recommended = evaluate_selected_actions(evaluation_context, selected,
                                                         evaluator.rotation_model, evaluator.speed_model)
        out = chosen_context[["well_id", "processing_time", "_row_in_well", "future_depth_m", "future_time_sec",
                              "pressure_axis", "pressure_rotation"]].copy()
        out["direction"] = direction
        out["selector_energy_type"] = chosen_context.rock_energy_type_final
        out["evaluator_energy_type"] = evaluation_context.rock_energy_type_final
        out["recommended_pressure_axis"] = selected.pressure_axis
        out["recommended_pressure_rotation"] = selected.pressure_rotation
        out["changed_action"] = ((selected.pressure_axis != chosen_context.pressure_axis)
                                 | (selected.pressure_rotation != chosen_context.pressure_rotation))
        out["evaluator_current_speed"] = current
        out["evaluator_recommended_speed"] = recommended
        out["evaluator_current_time_sec"] = out.future_depth_m / current
        out["evaluator_recommended_time_sec"] = out.future_depth_m / recommended
        out["selector_current_time_sec"] = out.future_depth_m / selected.current_predicted_target_speed
        out["selector_recommended_time_sec"] = out.future_depth_m / selected.predicted_target_speed
        out["evaluator_speed_gain_pct"] = 100*(recommended/current-1)
        out["selector_speed_gain_pct"] = 100*(selected.predicted_target_speed/selected.current_predicted_target_speed-1)
        out["evaluator_current_in_q05_q95_bounds"] = in_empirical_bounds(evaluation_context, evaluation_context, evaluator.ranges)
        out["evaluator_recommendation_in_q05_q95_bounds"] = in_empirical_bounds(evaluation_context, selected, evaluator.ranges)
        parts.append(out)
        if start % 4096 == 0:
            print(f"{direction}: blocks {min(start+chunk_size,len(anchors))}/{len(anchors)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def well_totals(blocks):
    return blocks.groupby(["direction", "well_id"], sort=True).agg(
        **{c: (c, "sum") for c in TIME_COLUMNS},
        blocks=("processing_time", "size"), changed_action_rate=("changed_action", "mean"),
    ).reset_index()


def combine_directions(wells):
    """Both directions for a well form ONE bootstrap cluster, not two samples."""
    if wells.duplicated(["direction", "well_id"]).any():
        raise ValueError("Duplicate well/direction pairs")
    expected = wells.direction.nunique()
    if not (wells.groupby("well_id").direction.nunique() == expected).all():
        raise ValueError("Every test well must occur in all evaluation directions.")
    return wells.groupby("well_id", sort=True)[TIME_COLUMNS].mean().reset_index()


def summarize_wells(wells, name, bootstrap_reps=10000, seed=RANDOM_STATE):
    values = wells[TIME_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected positive finite modeled times.")
    sums = values.sum(axis=0)
    effect = 100*(1-sums[1]/sums[0])
    speed_gain = 100*(sums[0]/sums[1]-1)
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_reps, 2))
    for start in range(0, bootstrap_reps, 500):
        size = min(500, bootstrap_reps-start)
        samples = rng.integers(len(values), size=(size, len(values)))
        totals = values[samples].sum(axis=1)
        boot[start:start+size, 0] = 100*(1-totals[:, 1]/totals[:, 0])
        boot[start:start+size, 1] = 100*(totals[:, 0]/totals[:, 1]-1)
    ci_time = np.quantile(boot[:, 0], [.025, .975])
    ci_speed = np.quantile(boot[:, 1], [.025, .975])
    own_effect = 100*(1-sums[3]/sums[2])
    per_well = 100*(1-values[:, 1]/values[:, 0])
    return {
        "evaluation": name, "wells": len(wells), "bootstrap_reps": bootstrap_reps,
        "evaluator_time_saved_pct": float(effect), "evaluator_speed_gain_pct": float(speed_gain),
        "time_saved_ci025": float(ci_time[0]), "time_saved_ci975": float(ci_time[1]),
        "speed_gain_ci025": float(ci_speed[0]), "speed_gain_ci975": float(ci_speed[1]),
        "positive_conditional_ci": bool(ci_time[0] > 0),
        "selector_self_time_saved_pct": float(own_effect),
        "self_minus_evaluator_time_saved_pp": float(own_effect-effect),
        "positive_wells_above_zero": int((per_well > 1e-10).sum()),
        "negative_wells_below_zero": int((per_well < -1e-10).sum()),
        "median_well_time_saved_pct": float(np.median(per_well)),
        "evaluator_current_time_min": float(sums[0]/60),
        "evaluator_recommended_time_min": float(sums[1]/60),
        "interpretation": "conditional cross-model diagnostic; not measured causal effect",
    }


def run(backend="lightgbm", iterations=650, bootstrap_reps=10000):
    started = time.time()
    protected = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in [DRILLING_ADVISORY_ARTIFACT_DIR, SIMULATOR_ADVISORY_ARTIFACT_DIR]
                 for p in folder.iterdir() if p.is_file()}
    split = json.loads((DRILLING_ADVISORY_REPORT_DIR/"split_manifest.json").read_text())
    source_hash = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    previous = json.loads((DRILLING_ADVISORY_ARTIFACT_DIR/"training_manifest.json").read_text())
    if source_hash != previous["source_sha256"]:
        raise ValueError("Input data differs from the experiment defining the fixed test set.")
    roles = partition_wells(split["train_wells"], split["test_wells"])
    protocol = {
        "backend": backend, "iterations": iterations, "bootstrap_reps": bootstrap_reps,
        "random_state": RANDOM_STATE, "source_sha256": source_hash,
        "A_wells": roles["A"], "B_wells": roles["B"], "test_wells": split["test_wells"],
        "primary": "symmetric_mean_of_two_directions_clustered_by_test_well",
        "directions": ["A_selects_B_evaluates", "B_selects_A_evaluates"],
        "decision_rule": "positive if primary 95% percentile bootstrap lower bound > 0; report both directions regardless",
        "selection_uses_evaluator": False, "hyperparameters_tuned_on_test": False,
        "production_artifacts_modified": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR/"protocol.json", protocol)
    history = core.prepare_history(pd.read_csv(RAW_DATA_PATH))
    data = core.add_targets(history)
    required = [c for c in core.NUMERIC_FEATURES if c != "hardness_score_smooth"]
    required += ["log_pseudo_mse_roll_median_30", "target_speed_near5", "target_rotation_near5"]
    eligible = data.dropna(subset=required)
    if set(eligible.well_id) != set(split["train_wells"]) | set(split["test_wells"]):
        raise ValueError("Eligible wells differ from original split.")
    test = eligible.loc[eligible.well_id.isin(split["test_wells"])].copy()
    print(f"Disjoint split: A={len(roles['A'])}, B={len(roles['B'])}, test={test.well_id.nunique()}", flush=True)
    branches = {name: fit_branch(name, wells, history, eligible, OUTPUT_DIR, backend, iterations)
                for name, wells in roles.items()}
    quality = []
    for name, branch in branches.items():
        factual = branch.calibration.transform(test)
        quality.append({"branch": name, **metrics("speed_chained", test.target_speed_near5,
                        core.predict_chain(factual, branch.rotation_model, branch.speed_model))})
        quality.append({"branch": name, **metrics("rotation", test.target_rotation_near5,
                        branch.rotation_model.predict(factual[core.ROTATION_FEATURES]))})
    quality = pd.DataFrame(quality)
    quality.to_csv(OUTPUT_DIR/"factual_prediction_metrics.csv", index=False, lineterminator="\n")
    print(quality.to_string(index=False), flush=True)
    blocks = pd.concat([evaluate_direction(branches["A"], branches["B"], test),
                        evaluate_direction(branches["B"], branches["A"], test)], ignore_index=True)
    wells = well_totals(blocks)
    combined = combine_directions(wells)
    summaries = [summarize_wells(part, name, bootstrap_reps)
                 for name, part in wells.groupby("direction", sort=True)]
    summaries.append(summarize_wells(combined, "symmetric_cross_model_PRIMARY", bootstrap_reps))
    summary = pd.DataFrame(summaries)
    for path, frame in [("blocks.csv", blocks), ("wells_by_direction.csv", wells),
                        ("wells_combined.csv", combined), ("summary.csv", summary)]:
        frame.to_csv(OUTPUT_DIR/path, index=False, lineterminator="\n")
    support = blocks.groupby("direction").agg(
        changed_action_rate=("changed_action", "mean"),
        current_inside_evaluator_bounds_rate=("evaluator_current_in_q05_q95_bounds", "mean"),
        recommended_inside_evaluator_bounds_rate=("evaluator_recommendation_in_q05_q95_bounds", "mean"),
        evaluator_positive_block_rate=("evaluator_speed_gain_pct", lambda s: float((s > 1e-10).mean())),
        evaluator_negative_block_rate=("evaluator_speed_gain_pct", lambda s: float((s < -1e-10).mean())),
    ).reset_index()
    support.to_csv(OUTPUT_DIR/"support_and_agreement.csv", index=False, lineterminator="\n")
    for relative, expected_hash in protected.items():
        if hashlib.sha256((PROJECT_ROOT/relative).read_bytes()).hexdigest() != expected_hash:
            raise AssertionError(f"Production artifact unexpectedly modified: {relative}")
    manifest = {**protocol, "test_rows": len(test), "blocks_per_direction": len(blocks)//2,
                "unchanged_production_artifact_sha256": protected,
                "elapsed_seconds": time.time()-started,
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), PROJECT_ROOT/"drilling_advisory.py", PROJECT_ROOT/"train_advisory.py"]},
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib", "lightgbm"]},
                "limitations": ["Both evaluators remain observational surrogate models with potentially shared bias",
                                "Models in each role use roughly half of original training wells; this does not validate the full-data production policy",
                                "Bootstrap is conditional on fitted models, and does not include retraining or spatial clustering uncertainty",
                                "Previously inspected test set is reused; this is not a fresh confirmatory holdout",
                                "Independent empirical bounds are diagnostics only, not joint action support or safety guarantees"]}
    write_json(OUTPUT_DIR/"manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    print(support.to_string(index=False), flush=True)
    print(f"Independent evaluation completed in {time.time()-started:.1f}s", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["lightgbm", "sklearn"], default="lightgbm")
    parser.add_argument("--iterations", type=int, default=650)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    run(args.backend, args.iterations, args.bootstrap_reps)
