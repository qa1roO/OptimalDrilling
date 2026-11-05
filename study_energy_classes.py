"""Train-only research of ordinal energy classes; original test is never evaluated.

uv run python study_energy_classes.py
Three cyclic disjoint roles inside the original train wells: selector fitting,
independent continuous-E evaluator fitting, and validation. All outer validation
predictions are out of well. Full production artifacts remain unchanged.
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
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold

import drilling_advisory as core
from config import PROJECT_ROOT, RAW_DATA_PATH, RANDOM_STATE, DRILLING_ADVISORY_REPORT_DIR, DRILLING_ADVISORY_ARTIFACT_DIR, SIMULATOR_ADVISORY_ARTIFACT_DIR
from train_advisory import make_model, write_json
from evaluate_independent import evaluate_selected_actions

OUT = DRILLING_ADVISORY_REPORT_DIR / "energy_class_study"
SIGNAL = "log_pseudo_mse_roll_median_30"
COUNTS = list(range(2, 9))
VARIANTS = ["no_E", "continuous_E"] + [f"q{k}" for k in COUNTS]


@dataclass
class Scheme:
    mean: float
    std: float
    cuts: list[float]
    k: int

    @classmethod
    def fit(cls, history, k):
        values = history[SIGNAL].dropna()
        cuts = values.quantile(np.arange(1, k)/k).tolist() if k > 1 else []
        if len(cuts) != len(set(cuts)):
            raise ValueError("Duplicate cutpoints: requested class count is unsupported.")
        return cls(float(values.mean()), max(float(values.std()), 1e-9), cuts, k)

    def transform(self, history):
        out = history.copy()
        values = out[SIGNAL]
        out["hardness_score_smooth"] = (values-self.mean)/self.std
        if self.k < 2:
            out["rock_energy_type_final"] = "all"
        else:
            codes = np.searchsorted(self.cuts, values.to_numpy(), side="left")
            labels = np.asarray([f"E{i+1}" for i in range(self.k)], dtype=object)[codes]
            labels[values.isna().to_numpy()] = "unknown"
            out["rock_energy_type_final"] = labels
        return out

    def json(self):
        return {"mean": self.mean, "std": self.std, "cuts_log_index": self.cuts, "k": self.k,
                "cutpoints_z": [(c-self.mean)/self.std for c in self.cuts],
                "fit_scope": "selector_fitting_wells_only"}


def specification(name):
    k = int(name[1:]) if name.startswith("q") else 0
    numeric = [c for c in core.NUMERIC_FEATURES if name != "no_E" or c != "hardness_score_smooth"]
    return k, numeric, core.CATEGORICAL_FEATURES if k else []


@dataclass
class Model:
    scheme: Scheme
    rotation: object
    speed: object
    ranges: dict


def fit_model(name, history, data, wells, folder, iterations):
    k, numeric, categorical = specification(name)
    schema = Scheme.fit(history.loc[history.well_id.isin(wells)], k)
    raw = data.loc[data.well_id.isin(wells)].copy()
    transformed = schema.transform(raw)
    oof = np.full(len(raw), np.nan)
    for fold, (fit_idx, hold_idx) in enumerate(GroupKFold(3).split(raw, groups=raw.well_id)):
        a, b = raw.iloc[fit_idx], raw.iloc[hold_idx]
        assert set(a.well_id).isdisjoint(b.well_id)
        fold_schema = Scheme.fit(history.loc[history.well_id.isin(a.well_id)], k)
        a, b = fold_schema.transform(a), fold_schema.transform(b)
        model = make_model("lightgbm", numeric, categorical, RANDOM_STATE+5, iterations)
        model.fit(a[core.ROTATION_FEATURES], a.target_rotation_near5)
        oof[hold_idx] = model.predict(b[core.ROTATION_FEATURES])
    assert np.isfinite(oof).all()
    rotation = make_model("lightgbm", numeric, categorical, RANDOM_STATE+5, iterations)
    rotation.fit(transformed[core.ROTATION_FEATURES], transformed.target_rotation_near5)
    transformed["candidate_target_rotation"] = oof
    speed = make_model("lightgbm", numeric+["candidate_target_rotation"], categorical, RANDOM_STATE+105, iterations)
    speed.fit(transformed[core.SPEED_FEATURES], transformed.target_speed_near5)
    result = Model(schema, rotation, speed, core.pressure_ranges(transformed))
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, folder/"model.joblib")
    write_json(folder/"scheme.json", schema.json())
    return result


def regression_by_well(raw, model, variant, fold):
    x = model.scheme.transform(raw)
    predicted = core.predict_chain(x, model.rotation, model.speed)
    residual = predicted-x.target_speed_near5.to_numpy()
    values = pd.DataFrame({"well_id": x.well_id.to_numpy(), "sse": residual**2,
                           "sae": np.abs(residual), "y_sum": x.target_speed_near5.to_numpy(),
                           "y2_sum": x.target_speed_near5.to_numpy()**2})
    result = values.groupby("well_id").agg(n=("sse", "size"), sse=("sse", "sum"),
                                          sae=("sae", "sum"), y_sum=("y_sum", "sum"), y2_sum=("y2_sum", "sum")).reset_index()
    result["variant"], result["fold"] = variant, fold
    return result


def choose_anchors(raw, per_well=3):
    pool = raw.loc[(raw._row_in_well >= 29) & (raw._row_in_well % 5 == 4)]
    selected = []
    for _, group in pool.groupby("well_id", sort=True):
        pos = np.unique(np.round(np.linspace(0, len(group)-1, min(per_well, len(group)))).astype(int))
        selected.append(group.iloc[pos])
    return pd.concat(selected).reset_index(drop=True)


def advisory(raw, selector, evaluator, policy, fold, chunk_size=128):
    context = selector.scheme.transform(raw)
    evaluate = evaluator.scheme.transform(raw)
    ranges = selector.ranges
    if policy == "q4_global_bounds":
        ranges = {"__global__": ranges["__global__"]}
    parts = []
    for start in range(0, len(raw), chunk_size):
        a = context.iloc[start:start+chunk_size].reset_index(drop=True)
        b = evaluate.iloc[start:start+chunk_size].reset_index(drop=True)
        selected, _ = core.recommend(a, selector.rotation, selector.speed, ranges)
        cur, rec = evaluate_selected_actions(b, selected, evaluator.rotation, evaluator.speed)
        result = a[["well_id", "processing_time", "_row_in_well", "future_depth_m", "future_time_sec"]].copy()
        result["current_time_sec"] = a.future_depth_m/cur
        result["recommended_time_sec"] = a.future_depth_m/rec
        result["changed"] = (selected.pressure_axis != a.pressure_axis) | (selected.pressure_rotation != a.pressure_rotation)
        result["speed_gain_pct"] = 100*(rec/cur-1)
        result["policy"], result["fold"] = policy, fold
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def natural_group_diagnostics(history):
    # Non-overlapping energy windows reduce duplication for the descriptive density study.
    sample = history.loc[(history._row_in_well % 30 == 29) & history[SIGNAL].notna()].copy()
    x = sample[SIGNAL].to_numpy().reshape(-1, 1)
    lo, hi = np.quantile(x, [.001, .999])
    grid = np.linspace(lo, hi, 600).reshape(-1, 1)
    rng = np.random.default_rng(RANDOM_STATE)
    ids = rng.choice(len(x), min(2500, len(x)), replace=False)
    rows = []
    densities = {"log_index": grid.ravel()}
    for k in range(1, 9):
        gmm = GaussianMixture(k, covariance_type="full", n_init=5, reg_covar=1e-5,
                              max_iter=500, random_state=RANDOM_STATE).fit(x)
        labels = gmm.predict(x)
        posterior = gmm.predict_proba(x)
        shares = np.bincount(labels, minlength=k)/len(x)
        density = np.exp(gmm.score_samples(grid))
        modes = find_peaks(density, prominence=.05*density.max())[0]
        row = {"k": k, "n_thinned": len(x), "gmm_bic": gmm.bic(x), "gmm_aic": gmm.aic(x),
               "min_component_share": float(shares.min()), "mean_max_posterior": float(posterior.max(axis=1).mean()),
               "share_uncertain_posterior_below_07": float((posterior.max(axis=1)<.7).mean()),
               "density_modes_5pct_prominence": len(modes)}
        if k > 1:
            row["gmm_silhouette"] = float(silhouette_score(x[ids], labels[ids])) if len(set(labels[ids])) > 1 else None
            km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(x)
            row["kmeans_silhouette"] = float(silhouette_score(x[ids], km.predict(x[ids])))
            cuts = np.quantile(x.ravel(), np.arange(1, k)/k)
            qlabels = np.searchsorted(cuts, x[ids].ravel(), side="left")
            row["quantile_silhouette"] = float(silhouette_score(x[ids], qlabels))
        rows.append(row)
        densities[f"gmm_{k}"] = density
        write_json(OUT/"natural_groups_progress.json", rows)
    density_frame = pd.DataFrame(densities)
    kde = gaussian_kde(x.ravel())
    modes = []
    for multiplier in [.7, 1., 1.4]:
        curve = gaussian_kde(x.ravel(), bw_method=kde.factor*multiplier)(grid.ravel())
        peaks = find_peaks(curve, prominence=.05*curve.max())[0]
        modes.append({"bandwidth_multiplier": multiplier, "modes_5pct_prominence": len(peaks),
                      "peak_positions_log_index": grid[peaks].ravel().tolist()})
        density_frame[f"kde_{multiplier}"] = curve
    pd.DataFrame(rows).to_csv(OUT/"natural_groups.csv", index=False, lineterminator="\n")
    density_frame.to_csv(OUT/"density_curves.csv", index=False, lineterminator="\n")
    write_json(OUT/"kde_modality.json", modes)
    return sample


def summarize_prediction(well_errors, reps=10000):
    pivot = well_errors.pivot(index="well_id", columns="variant", values="sse")[VARIANTS]
    n = well_errors.loc[well_errors.variant==VARIANTS[0]].set_index("well_id").loc[pivot.index, "n"].to_numpy()
    sse = pivot.to_numpy()
    point = np.sqrt(sse.sum(axis=0)/n.sum())
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.empty((reps, len(VARIANTS)))
    for start in range(0, reps, 200):
        size = min(200, reps-start)
        ids = rng.integers(len(n), size=(size, len(n)))
        boot[start:start+size] = np.sqrt(sse[ids].sum(axis=1)/n[ids].sum(axis=1)[:, None])
    results = []
    for j, name in enumerate(VARIANTS):
        part = well_errors.loc[well_errors.variant==name]
        total = part.sum(numeric_only=True)
        tss = total.y2_sum-total.y_sum**2/total.n
        folds = part.groupby("fold").agg(sse=("sse", "sum"), n=("n", "sum"))
        fold_rmse = np.sqrt(folds.sse/folds.n)
        result = {"variant": name, "classes": specification(name)[0], "n": int(n.sum()), "wells": len(n),
                  "RMSE": point[j], "MAE": total.sae/total.n, "R2": 1-total.sse/tss,
                  "fold_RMSE_mean": float(fold_rmse.mean()), "fold_RMSE_SE": float(fold_rmse.std()/np.sqrt(len(folds))),
                  "RMSE_ci025": float(np.quantile(boot[:, j], .025)), "RMSE_ci975": float(np.quantile(boot[:, j], .975))}
        # Two explicitly named contrast families; intervals are Bonferroni-adjusted within each family.
        for ref in ["q4", "continuous_E"]:
            r = VARIANTS.index(ref)
            delta = 100*(boot[:, j]/boot[:, r]-1)
            alpha = .05/(len(VARIANTS)-1)
            result[f"RMSE_delta_vs_{ref}_pct"] = float(100*(point[j]/point[r]-1))
            result[f"delta_vs_{ref}_family_ci_low"] = float(np.quantile(delta, alpha/2))
            result[f"delta_vs_{ref}_family_ci_high"] = float(np.quantile(delta, 1-alpha/2))
        results.append(result)
    return pd.DataFrame(results), boot


def summarize_policies(blocks, reps=10000):
    wells = blocks.groupby(["policy", "well_id"]).agg(current=("current_time_sec", "sum"),
        recommended=("recommended_time_sec", "sum"), n=("changed", "size"), changed=("changed", "mean")).reset_index()
    policies = list(wells.policy.unique())
    current = wells.pivot(index="well_id", columns="policy", values="current")[policies].to_numpy()
    recommended = wells.pivot(index="well_id", columns="policy", values="recommended")[policies].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.empty((reps, len(policies)))
    for start in range(0, reps, 200):
        size = min(200, reps-start)
        ids = rng.integers(len(current), size=(size, len(current)))
        boot[start:start+size] = 100*(1-recommended[ids].sum(axis=1)/current[ids].sum(axis=1))
    rows = []
    for j, policy in enumerate(policies):
        part = wells.loc[wells.policy==policy]
        b = blocks.loc[blocks.policy==policy]
        effect = 100*(1-recommended[:, j].sum()/current[:, j].sum())
        result = {"policy": policy, "wells": len(current), "sampled_nonoverlap_blocks": len(b),
                  "independent_time_saved_pct": effect,
                  "independent_speed_gain_pct": 100*(current[:, j].sum()/recommended[:, j].sum()-1),
                  "time_ci025": float(np.quantile(boot[:, j], .025)), "time_ci975": float(np.quantile(boot[:, j], .975)),
                  "changed_action_rate": float(b.changed.mean()),
                  "negative_wells": int((part.recommended>part.current+1e-9).sum())}
        r = policies.index("q4")
        delta = boot[:, j]-boot[:, r]
        alpha = .05/(len(policies)-1)
        result["delta_time_vs_q4_pp"] = float(effect-100*(1-recommended[:, r].sum()/current[:, r].sum()))
        result["delta_vs_q4_family_ci_low"] = float(np.quantile(delta, alpha/2))
        result["delta_vs_q4_family_ci_high"] = float(np.quantile(delta, 1-alpha/2))
        rows.append(result)
    return pd.DataFrame(rows), wells


def class_profiles(history, k, bootstrap_reps=500):
    scheme = Scheme.fit(history, k)
    data = scheme.transform(history).dropna(subset=[SIGNAL]).copy()
    data["speed_mh"] = data.speed*3600
    data["smoothed_proxy"] = np.expm1(data[SIGNAL])
    change = data.rock_energy_type_final.ne(data.groupby("well_id").rock_energy_type_final.shift())
    data["run_id"] = change.groupby(data.well_id).cumsum()
    runs = data.groupby(["well_id", "run_id"]).agg(label=("rock_energy_type_final", "first"),
              rows=("speed", "size"), seconds=("d_time", "sum")).reset_index()
    profile = data.groupby("rock_energy_type_final").agg(rows=("speed", "size"), wells=("well_id", "nunique"),
        median_index_z=("hardness_score_smooth", "median"), median_smoothed_proxy=("smoothed_proxy", "median"),
        median_speed_mh=("speed_mh", "median"), median_rotation=("rotation", "median"),
        median_pressure_axis=("pressure_axis", "median"), median_pressure_rotation=("pressure_rotation", "median"),
        median_depth=("depth", "median")).reset_index()
    profile["share"] = profile.rows/profile.rows.sum()
    for label in profile.rock_energy_type_final:
        mask = profile.rock_energy_type_final==label
        relevant = runs.loc[runs.label==label]
        profile.loc[mask, "median_run_records"] = relevant.rows.median()
        profile.loc[mask, "median_run_seconds"] = relevant.seconds.median()
    profile["classes"] = k
    profile.to_csv(OUT/f"profiles_q{k}.csv", index=False, lineterminator="\n")
    write_json(OUT/f"scheme_q{k}_development_only.json", scheme.json())
    # Whole-well bootstrap of actual row-weighted training quantiles; no iid-row bootstrap.
    arrays = [p[SIGNAL].dropna().to_numpy() for _, p in history.groupby("well_id")]
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.empty((bootstrap_reps, k-1))
    for rep in range(bootstrap_reps):
        sampled = np.concatenate([arrays[i] for i in rng.integers(len(arrays), size=len(arrays))])
        boot[rep] = np.quantile(sampled, np.arange(1, k)/k)
    stability = []
    valid_signal = data[SIGNAL].to_numpy()
    reference = np.searchsorted(scheme.cuts, valid_signal, side="left")
    for rep in range(bootstrap_reps):
        stability.append(float((np.searchsorted(boot[rep], valid_signal, side="left")!=reference).mean()))
    cuts = pd.DataFrame({"cut": np.arange(1, k), "point_log_index": scheme.cuts,
                         "bootstrap_ci025": np.quantile(boot, .025, axis=0),
                         "bootstrap_ci975": np.quantile(boot, .975, axis=0)})
    cuts.to_csv(OUT/f"cutpoint_stability_q{k}.csv", index=False, lineterminator="\n")
    return {"classes": k, "switches_per_100_records": float(100*(change.sum()-data.well_id.nunique())/(len(data)-data.well_id.nunique())),
            "median_run_records": float(runs.rows.median()), "median_run_seconds": float(runs.seconds.median()),
            "bootstrap_median_reassigned_fraction": float(np.median(stability)),
            "bootstrap_p95_reassigned_fraction": float(np.quantile(stability, .95))}


def run(iterations=650, bootstrap_reps=10000, anchors_per_well=3):
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    protected = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for d in [DRILLING_ADVISORY_ARTIFACT_DIR, SIMULATOR_ADVISORY_ARTIFACT_DIR]
                 for p in d.iterdir() if p.is_file()}
    split = json.loads((DRILLING_ADVISORY_REPORT_DIR/"split_manifest.json").read_text())
    development = set(map(str, split["train_wells"]))
    excluded_test = set(map(str, split["test_wells"]))
    assert development.isdisjoint(excluded_test)
    raw = pd.read_csv(RAW_DATA_PATH, dtype={"well_id": str})
    raw = raw.loc[raw.well_id.isin(development)].copy()  # exclude original test BEFORE feature construction
    history = core.prepare_history(raw)
    data = core.add_targets(history)
    required = [c for c in core.NUMERIC_FEATURES if c != "hardness_score_smooth"]+[SIGNAL, "target_speed_near5", "target_rotation_near5"]
    data = data.dropna(subset=required).copy()
    partitions = [sorted(set(data.iloc[idx].well_id)) for _, idx in GroupKFold(3).split(data, groups=data.well_id)]
    protocol = {"variants": VARIANTS, "classes": COUNTS, "iterations": iterations, "bootstrap_reps": bootstrap_reps,
        "anchors_per_well": anchors_per_well, "partition_wells": partitions, "excluded_original_test_wells": sorted(excluded_test),
        "design": "cyclic disjoint selector/evaluator/validation roles on original training wells only",
        "primary_metric": "out-of-well pooled validation RMSE of chained speed prediction",
        "secondary_metric": "independent continuous-E evaluator, sampled non-overlapping validation blocks",
        "practical_RMSE_tolerance_pct": .5,
        "multiplicity": "Bonferroni percentile well-bootstrap intervals within each named reference contrast family",
        "natural_clusters": "descriptive GMM BIC/KMeans/quantile silhouette on nonoverlapping 30-record windows; not geology validation",
        "production_modified": False,
        "source_sha256": hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()}
    write_json(OUT/"protocol.json", protocol)
    print(f"Development={len(development)} wells; excluded test={len(excluded_test)}; partitions={[len(p) for p in partitions]}", flush=True)
    all_errors, all_blocks = [], []
    for fold in range(3):
        fit_wells, evaluator_wells, validation_wells = partitions[fold], partitions[(fold+1)%3], partitions[(fold+2)%3]
        assert set(fit_wells).isdisjoint(evaluator_wells) and set(fit_wells).isdisjoint(validation_wells) and set(evaluator_wells).isdisjoint(validation_wells)
        validation = data.loc[data.well_id.isin(validation_wells)].copy()
        anchors = choose_anchors(validation, anchors_per_well)
        folder = OUT/"cache"/f"fold_{fold}"
        print(f"Fold {fold}: evaluator fitting ({len(evaluator_wells)} wells)", flush=True)
        evaluator = fit_model("continuous_E", history, data, evaluator_wells, folder/"evaluator", iterations)
        pd.DataFrame([{"fold": fold, "RMSE": np.sqrt(np.mean((core.predict_chain(evaluator.scheme.transform(validation), evaluator.rotation, evaluator.speed)-validation.target_speed_near5)**2))}]).to_csv(OUT/f"evaluator_quality_fold_{fold}.csv", index=False, lineterminator="\n")
        fitted = {}
        for name in VARIANTS:
            print(f"Fold {fold}: {name} fitting, then independent recommendation check", flush=True)
            model = fit_model(name, history, data, fit_wells, folder/name, iterations)
            error = regression_by_well(validation, model, name, fold)
            blocks = advisory(anchors, model, evaluator, name, fold)
            error.to_csv(OUT/f"errors_fold_{fold}_{name}.csv", index=False, lineterminator="\n")
            blocks.to_csv(OUT/f"blocks_fold_{fold}_{name}.csv", index=False, lineterminator="\n")
            all_errors.append(error); all_blocks.append(blocks)
            print(f"Fold {fold}: {name} RMSE={np.sqrt(error.sse.sum()/error.n.sum()):.8f}, evaluator time saved={100*(1-blocks.recommended_time_sec.sum()/blocks.current_time_sec.sum()):.3f}%", flush=True)
            if name in ["continuous_E", "q4"]:
                fitted[name] = model
        # Factorial controls separate category-as-feature from category-conditioned action bounds.
        continuous = fitted["continuous_E"]
        q4_scheme = Scheme.fit(history.loc[history.well_id.isin(fit_wells)], 4)
        q4_train = q4_scheme.transform(data.loc[data.well_id.isin(fit_wells)])
        bounds_only = Model(q4_scheme, continuous.rotation, continuous.speed, core.pressure_ranges(q4_train))
        for policy, model in [("continuous_E_q4_bounds", bounds_only), ("q4_global_bounds", fitted["q4"])]:
            blocks = advisory(anchors, model, evaluator, policy, fold)
            blocks.to_csv(OUT/f"blocks_fold_{fold}_{policy}.csv", index=False, lineterminator="\n")
            all_blocks.append(blocks)
    errors = pd.concat(all_errors, ignore_index=True)
    blocks = pd.concat(all_blocks, ignore_index=True)
    predictive, _ = summarize_prediction(errors, bootstrap_reps)
    policy, wells = summarize_policies(blocks, bootstrap_reps)
    predictive.to_csv(OUT/"prediction_summary.csv", index=False, lineterminator="\n")
    policy.to_csv(OUT/"policy_summary.csv", index=False, lineterminator="\n")
    errors.to_csv(OUT/"validation_errors_by_well.csv", index=False, lineterminator="\n")
    wells.to_csv(OUT/"policy_by_well.csv", index=False, lineterminator="\n")
    print(predictive[["variant", "RMSE", "R2", "RMSE_delta_vs_q4_pct", "delta_vs_q4_family_ci_low", "delta_vs_q4_family_ci_high"]].to_string(index=False), flush=True)
    print(policy.to_string(index=False), flush=True)
    print("Descriptive density and class stability analysis", flush=True)
    natural_group_diagnostics(history)
    stability = []
    for k in COUNTS:
        print(f"Class profiles and whole-well cutpoint bootstrap: K={k}", flush=True)
        stability.append(class_profiles(history, k))
    pd.DataFrame(stability).to_csv(OUT/"class_stability.csv", index=False, lineterminator="\n")
    for p, expected in protected.items():
        assert hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()==expected
    manifest = {**protocol, "development_rows": len(history), "eligible_rows": len(data),
        "elapsed_seconds": time.time()-started, "protected_artifact_sha256": protected,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), PROJECT_ROOT/"drilling_advisory.py", PROJECT_ROOT/"train_advisory.py"]},
        "versions": {p: importlib.metadata.version(p) for p in ["lightgbm", "numpy", "pandas", "scikit-learn", "scipy"]},
        "limitations": ["Model recommendation effects remain observational cross-model diagnostics, not causal effects",
                        "Selector models use one third of development wells, not the production training budget",
                        "Only three evenly spaced non-overlapping blocks per validation well are used for policy diagnostics",
                        "Confidence intervals are conditional on fitted models and this partition, not full retraining uncertainty",
                        "No lithology, tool-wear or independently measured rock-strength labels are available",
                        "Speed is algebraically part of the index, so class speed separation alone is not external validation"]}
    write_json(OUT/"manifest.json", manifest)
    print(f"Study complete in {time.time()-started:.1f}s; production models unchanged", flush=True)
    return predictive, policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=650)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--anchors-per-well", type=int, default=3)
    args = parser.parse_args()
    run(args.iterations, args.bootstrap_reps, args.anchors_per_well)
