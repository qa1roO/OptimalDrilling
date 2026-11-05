"""Exploratory follow-up: learned four-group boundaries versus quartiles.

Requires the completed study_energy_classes.py outputs and evaluator checkpoints.
It never uses the original test wells or replaces production models.
"""
from dataclasses import dataclass
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold

import drilling_advisory as core
from config import RAW_DATA_PATH, RANDOM_STATE
from train_advisory import make_model, write_json
# Imports also resolve classes in the research CLI's __main__ pickle checkpoints.
from study_energy_classes import (OUT, SIGNAL, VARIANTS, Model, Scheme, advisory, choose_anchors,
                                  regression_by_well, summarize_policies)


class KMeansScheme(Scheme):
    @classmethod
    def fit(cls, history, k=4):
        base = Scheme.fit(history, k)
        x = history[SIGNAL].dropna().to_numpy().reshape(-1, 1)
        centers = np.sort(KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(x).cluster_centers_.ravel())
        return cls(base.mean, base.std, ((centers[:-1]+centers[1:])/2).tolist(), k)


@dataclass
class GMMScheme(Scheme):
    mixture: object = None

    @classmethod
    def fit(cls, history, k=4):
        base = Scheme.fit(history, k)
        sample = history.loc[(history._row_in_well%30==29), SIGNAL].dropna().to_numpy().reshape(-1, 1)
        model = GaussianMixture(k, n_init=10, reg_covar=1e-5, max_iter=500, random_state=RANDOM_STATE).fit(sample)
        return cls(base.mean, base.std, [], k, model)

    def transform(self, history):
        out = super().transform(history)
        mask = out[SIGNAL].notna()
        if not mask.any():
            return out
        order = np.argsort(self.mixture.means_.ravel())
        ranks = np.empty(self.k, dtype=int)
        ranks[order] = np.arange(self.k)
        predicted = ranks[self.mixture.predict(out.loc[mask, [SIGNAL]].to_numpy())]
        out.loc[mask, "rock_energy_type_final"] = np.asarray([f"E{i+1}" for i in range(self.k)])[predicted]
        return out

    def json(self):
        info = super().json()
        order = np.argsort(self.mixture.means_.ravel())
        info.update(method="GMM_MAP_sorted_component_means", means_log=self.mixture.means_.ravel()[order].tolist(),
                    stds_log=np.sqrt(self.mixture.covariances_.ravel()[order]).tolist(),
                    weights=self.mixture.weights_[order].tolist())
        return info


def fit(scheme_type, history, data, wells, iterations=650):
    raw = data.loc[data.well_id.isin(wells)].copy()
    scheme = scheme_type.fit(history.loc[history.well_id.isin(wells)], 4)
    x = scheme.transform(raw)
    oof = np.full(len(raw), np.nan)
    for fit_idx, hold_idx in GroupKFold(3).split(raw, groups=raw.well_id):
        a, b = raw.iloc[fit_idx], raw.iloc[hold_idx]
        assert set(a.well_id).isdisjoint(b.well_id)
        sub = scheme_type.fit(history.loc[history.well_id.isin(a.well_id)], 4)
        a, b = sub.transform(a), sub.transform(b)
        model = make_model("lightgbm", core.NUMERIC_FEATURES, core.CATEGORICAL_FEATURES, RANDOM_STATE+5, iterations)
        model.fit(a[core.ROTATION_FEATURES], a.target_rotation_near5)
        oof[hold_idx] = model.predict(b[core.ROTATION_FEATURES])
    rotation = make_model("lightgbm", core.NUMERIC_FEATURES, core.CATEGORICAL_FEATURES, RANDOM_STATE+5, iterations)
    rotation.fit(x[core.ROTATION_FEATURES], x.target_rotation_near5)
    x["candidate_target_rotation"] = oof
    speed = make_model("lightgbm", core.SPEED_NUMERIC_FEATURES, core.CATEGORICAL_FEATURES, RANDOM_STATE+105, iterations)
    speed.fit(x[core.SPEED_FEATURES], x.target_speed_near5)
    return Model(scheme, rotation, speed, core.pressure_ranges(x))


def summarize(errors, reps=10000):
    variants = VARIANTS+["kmeans4", "gmm4"]
    pivot = errors.pivot(index="well_id", columns="variant", values="sse")[variants]
    sse = pivot.to_numpy()
    n = errors.loc[errors.variant=="q4"].set_index("well_id").loc[pivot.index, "n"].to_numpy()
    point = np.sqrt(sse.sum(axis=0)/n.sum())
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.empty((reps, len(variants)))
    for start in range(0, reps, 200):
        size = min(200, reps-start)
        ids = rng.integers(len(n), size=(size, len(n)))
        boot[start:start+size] = np.sqrt(sse[ids].sum(axis=1)/n[ids].sum(axis=1)[:, None])
    result = []
    for j, name in enumerate(variants):
        total = errors.loc[errors.variant==name].sum(numeric_only=True)
        row = {"variant": name, "RMSE": point[j], "MAE": total.sae/total.n,
               "R2": 1-total.sse/(total.y2_sum-total.y_sum**2/total.n), "n": int(total.n), "wells": len(n)}
        for ref in ["q4", "continuous_E"]:
            r = variants.index(ref)
            delta = 100*(boot[:, j]/boot[:, r]-1)
            alpha = .05/(len(variants)-1)
            row[f"RMSE_delta_vs_{ref}_pct"] = float(100*(point[j]/point[r]-1))
            row[f"delta_vs_{ref}_family_ci_low"] = float(np.quantile(delta, alpha/2))
            row[f"delta_vs_{ref}_family_ci_high"] = float(np.quantile(delta, 1-alpha/2))
        result.append(row)
    return pd.DataFrame(result)


def describe(history, name, scheme):
    raw = history.dropna(subset=[SIGNAL]).copy()
    x = scheme.transform(raw)
    x["speed_mh"] = x.speed*3600
    profile = x.groupby("rock_energy_type_final").agg(rows=("speed", "size"), wells=("well_id", "nunique"),
        median_index_z=("hardness_score_smooth", "median"), median_speed_mh=("speed_mh", "median"),
        median_pressure_axis=("pressure_axis", "median"), median_pressure_rotation=("pressure_rotation", "median")).reset_index()
    profile["share"] = profile.rows/profile.rows.sum()
    profile.to_csv(OUT/f"profiles_{name}.csv", index=False, lineterminator="\n")
    q4 = Scheme.fit(history, 4).transform(raw).rock_energy_type_final
    cross = pd.crosstab(q4.to_numpy(), x.rock_energy_type_final.to_numpy())
    cross.to_csv(OUT/f"q4_vs_{name}_counts.csv", lineterminator="\n")
    grid = np.linspace(raw[SIGNAL].min(), raw[SIGNAL].max(), 3000)
    labels = scheme.transform(pd.DataFrame({SIGNAL: grid})).rock_energy_type_final.to_numpy()
    transitions = np.flatnonzero(labels[1:]!=labels[:-1])+1
    return {"method": name, "ARI_vs_quartiles": float(adjusted_rand_score(q4, x.rock_energy_type_final)),
            "sorted_signal_label_sequence": [labels[0]]+labels[transitions].tolist(),
            "transition_locations_log": grid[transitions].tolist(), "scheme": scheme.json()}


def run():
    started = time.time()
    protocol = json.loads((OUT/"protocol.json").read_text())
    original_manifest = json.loads((OUT/"manifest.json").read_text())
    original_train = set(w for part in protocol["partition_wells"] for w in part)
    raw = pd.read_csv(RAW_DATA_PATH, dtype={"well_id": str})
    history = core.prepare_history(raw.loc[raw.well_id.isin(original_train)])
    data = core.add_targets(history)
    required = [c for c in core.NUMERIC_FEATURES if c!="hardness_score_smooth"]+[SIGNAL, "target_speed_near5", "target_rotation_near5"]
    data = data.dropna(subset=required)
    print("Exploratory learned-boundary comparison; original test still excluded", flush=True)
    all_errors = [pd.read_csv(OUT/"validation_errors_by_well.csv", dtype={"well_id": str})]
    all_blocks = [pd.read_csv(p, dtype={"well_id": str}) for p in sorted(OUT.glob("blocks_fold_*.csv"))
                  if "kmeans4" not in p.name and "gmm4" not in p.name]
    methods = [("kmeans4", KMeansScheme), ("gmm4", GMMScheme)]
    for fold in range(3):
        fit_wells = protocol["partition_wells"][fold]
        val_wells = protocol["partition_wells"][(fold+2)%3]
        validation = data.loc[data.well_id.isin(val_wells)]
        anchors = choose_anchors(validation, protocol["anchors_per_well"])
        evaluator = joblib.load(OUT/"cache"/f"fold_{fold}"/"evaluator"/"model.joblib")
        for name, scheme_type in methods:
            print(f"Fold {fold}: {name} fitting and independent evaluation", flush=True)
            model = fit(scheme_type, history, data, fit_wells, protocol["iterations"])
            errors = regression_by_well(validation, model, name, fold)
            blocks = advisory(anchors, model, evaluator, name, fold)
            errors.to_csv(OUT/f"errors_fold_{fold}_{name}.csv", index=False, lineterminator="\n")
            blocks.to_csv(OUT/f"blocks_fold_{fold}_{name}.csv", index=False, lineterminator="\n")
            all_errors.append(errors); all_blocks.append(blocks)
            print(f"{name}: RMSE={np.sqrt(errors.sse.sum()/errors.n.sum()):.8f}", flush=True)
    predictive = summarize(pd.concat(all_errors), protocol["bootstrap_reps"])
    policy, _ = summarize_policies(pd.concat(all_blocks), protocol["bootstrap_reps"])
    predictive.to_csv(OUT/"prediction_summary_extended.csv", index=False, lineterminator="\n")
    policy.to_csv(OUT/"policy_summary_extended.csv", index=False, lineterminator="\n")
    descriptions = [describe(history, name, typ.fit(history, 4)) for name, typ in methods]
    write_json(OUT/"learned_four_group_interpretation.json", descriptions)
    # Sensitivity of the descriptive BIC result to the phase of nonoverlapping windows.
    density = []
    for offset in [9, 19, 29]:
        x = history.loc[(history._row_in_well>=29)&(history._row_in_well%30==offset), SIGNAL].dropna().to_numpy().reshape(-1,1)
        for k in range(1, 9):
            m = GaussianMixture(k, n_init=10, max_iter=500, reg_covar=1e-5, random_state=RANDOM_STATE).fit(x)
            density.append({"offset": offset, "k": k, "n": len(x), "BIC": m.bic(x)})
    pd.DataFrame(density).to_csv(OUT/"gmm_bic_phase_sensitivity.csv", index=False, lineterminator="\n")
    for relative, expected in original_manifest["protected_artifact_sha256"].items():
        from config import PROJECT_ROOT
        assert hashlib.sha256((PROJECT_ROOT/relative).read_bytes()).hexdigest()==expected
    write_json(OUT/"learned_boundaries_manifest.json", {"exploratory_followup": True,
        "methods": [name for name,_ in methods], "original_test_used": False,
        "original_protocol": protocol, "elapsed_seconds": time.time()-started})
    print(predictive.to_string(index=False), flush=True)
    print(policy.loc[policy.policy.isin(["q4","kmeans4","gmm4","continuous_E"])].to_string(index=False), flush=True)
    print(f"Learned-boundary follow-up complete in {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    run()
