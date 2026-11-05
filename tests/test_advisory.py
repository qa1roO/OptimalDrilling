import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import drilling_advisory as core


def telemetry(wells=2, rows=90):
    parts = []
    for well in range(wells):
        i = np.arange(rows)
        dt = 4.0 + i % 4
        speed = .01 + .002*np.sin(i/8 + well)
        depth = speed*dt
        parts.append(pd.DataFrame({"well_id": str(well),
                                   "processing_time": pd.date_range("2025-01-01", periods=rows, freq="5s"),
                                   "depth": np.cumsum(depth), "d_depth": depth, "d_time": dt,
                                   "speed": speed, "rotation": 95+5*np.cos(i/7),
                                   "pressure_axis": 10000+i*10+well*500,
                                   "pressure_rotation": 9000+i*8+well*400}))
    return pd.concat(parts, ignore_index=True)


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


class CausalPipelineTests(unittest.TestCase):
    def test_changing_future_cannot_change_past_features(self):
        raw = telemetry()
        history = core.prepare_history(raw)
        calibration = core.EnergyCalibration.fit(history.loc[history.well_id == "1"])
        altered = raw.copy()
        mask = (altered.well_id == "0") & (altered.groupby("well_id").cumcount() >= 50)
        altered.loc[mask, ["pressure_axis", "pressure_rotation", "rotation", "d_time", "d_depth"]] *= 5
        before = calibration.transform(history).loc[lambda x: (x.well_id == "0") & (x._row_in_well < 50)]
        after = calibration.transform(core.prepare_history(altered)).loc[lambda x: (x.well_id == "0") & (x._row_in_well < 50)]
        assert_frame_equal(before, after)

    def test_training_energy_calibration_ignores_test_wells(self):
        raw = telemetry()
        first = core.prepare_history(raw)
        altered = raw.copy()
        altered.loc[altered.well_id == "1", "pressure_axis"] *= 100
        second = core.prepare_history(altered)
        self.assertEqual(core.EnergyCalibration.fit(first.loc[first.well_id == "0"]),
                         core.EnergyCalibration.fit(second.loc[second.well_id == "0"]))

    def test_target_matches_physical_future_time_and_does_not_cross_wells(self):
        history = core.prepare_history(telemetry())
        targets = core.add_targets(history)
        for well, part in targets.groupby("well_id"):
            future = part.iloc[21:26]
            anchor = part.iloc[20]
            expected = future.d_depth.sum()/future.d_time.sum()
            self.assertAlmostEqual(anchor.target_speed_near5, expected)
            self.assertAlmostEqual(anchor.future_depth_m/anchor.target_speed_near5, future.d_time.sum())
            self.assertTrue(part.target_speed_near5.tail(5).isna().all())

    def test_candidate_differences_and_history_remain_consistent(self):
        history = core.prepare_history(telemetry())
        cal = core.EnergyCalibration.fit(history)
        frame = cal.transform(history).dropna()
        rows = frame.iloc[[30, 60]].reset_index(drop=True)
        grid = core.candidate_grid(rows, core.pressure_ranges(frame))
        for c in ["pressure_axis", "pressure_rotation"]:
            np.testing.assert_allclose(grid[f"{c}_diff1"], grid[c]-grid[f"{c}_lag1"])
            self.assertTrue((grid[f"delta_{c}_frac"].abs() <= core.MAX_DELTA_FRAC+1e-12).all())
        for row_id, group in grid.groupby("row_id"):
            np.testing.assert_allclose(group.hardness_score_smooth, rows.iloc[row_id].hardness_score_smooth)
            self.assertEqual(group.iloc[0].pressure_axis, rows.iloc[row_id].pressure_axis)

    def test_no_change_wins_ties_and_unreachable_support(self):
        history = core.prepare_history(telemetry())
        frame = core.EnergyCalibration.fit(history).transform(history).dropna()
        ranges = core.pressure_ranges(frame)
        for entry in ranges.values():
            entry["pressure_axis_q05"] = 50000
            entry["pressure_axis_q95"] = 60000
        best, _ = core.recommend(frame.iloc[[30, 60]], ConstantModel(90), ConstantModel(.02), ranges)
        self.assertTrue((best.candidate_id == 0).all())
        self.assertTrue((best.optimizer_score == best.current_predicted_target_speed).all())

    def test_no_change_has_exactly_zero_paired_effect(self):
        result = core.paired_time_summary([.1, .8], [5, 100], [.02, .012], [.02, .012])
        np.testing.assert_array_equal(result.model_based_predicted_uplift_pct, [0, 0])
        np.testing.assert_array_equal(result.paired_time_saved_pct, [0, 0])
        np.testing.assert_array_equal(result.anchored_model_time_sec, [5, 100])

    def test_real_artifacts_match_streaming_and_batch(self):
        from config import (DRILLING_ADVISORY_ARTIFACT_DIR as artifact_dir,
                            SIMULATOR_ADVISORY_ARTIFACT_DIR as simulator_dir, RAW_DATA_PATH)
        config = json.loads((artifact_dir / "feature_config.json").read_text())
        if config.get("schema_version") != core.SCHEMA_VERSION:
            self.skipTest("Retrain schema v2 artifacts first")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))
        from app.data.advisory_engine import AdvisoryEngine
        engine = AdvisoryEngine(simulator_dir)
        raw = pd.read_csv(RAW_DATA_PATH)
        well = raw.loc[raw.well_id == raw.well_id.iloc[0]].sort_values("processing_time").head(75)
        for i, row in enumerate(well.to_dict("records")):
            # Deliberately corrupt supplied scores and labels; runtime must ignore them.
            row["hardness_score_smooth"], row["rock_energy_type_final"] = 999, "leaked_future_label"
            result = engine.update(row, compute=i == len(well)-1)
        batch = engine.calibration.transform(core.prepare_history(well)).iloc[-1]
        live = engine._build_feature_frame().iloc[-1]
        np.testing.assert_allclose(batch[core.NUMERIC_FEATURES].astype(float), live[core.NUMERIC_FEATURES].astype(float))
        self.assertEqual(batch.rock_energy_type_final, live.rock_energy_type_final)
        best, _ = core.recommend(pd.DataFrame([batch]), engine.rotation_model, engine.speed_model, engine.surface_ranges)
        self.assertAlmostEqual(best.iloc[0].predicted_target_speed, result["recommended"]["predicted_target_speed"])
        for path in artifact_dir.iterdir():
            self.assertEqual(path.read_bytes(), (simulator_dir/path.name).read_bytes())


if __name__ == "__main__":
    unittest.main()
