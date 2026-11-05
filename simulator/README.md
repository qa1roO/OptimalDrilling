# OptimalDrilling Simulator

Desktop simulator for drilling replay and advisory-model visualization.

## Current Flow

The application starts from `run.py`, creates `SimulatorMainWindow`, and shows two panels:

- `app/scene/side_view_widget.py` - rig side view, borehole animation, replay well selection, telemetry signals.
- `app/charts/performance_3d_widget_stub.py` - performance dashboard with depth-based 2D plots and a 3D empirical energy surface.

The main runtime path is replay/advisory mode:

1. `SideViewWidget` reads replay telemetry from `telemetry_with_energy_quantiles.csv` when it is available.
2. It emits `replay_sample(row, depth_m)` while the rig animation advances.
3. `Performance3DWidget.append_advisory_telemetry()` updates the 2D charts and the 3D global speed response surface.
4. `AdvisoryEngine` loads the same schema-v2 joblib artifacts and training-only energy calibration exported by `train_advisory.py` into `app/ml_artifacts`.

The engine derives features from raw historical telemetry through the shared
`drilling_advisory.py` pipeline. Precomputed CSV energy labels are not used for
inference. The source of `dt` is `d_time`, exactly as in training. The current
controls are always an explicit no-change candidate.

If replay telemetry is unavailable, the simulator can fall back to synthetic layer drilling and legacy cluster profiles from `simulator_core`.

The 3D background is a single smoothed empirical binned-median speed surface built
only from training rows. Rock energy quantiles remain model features and UI/report
context. The surface is used only for visualization: white and red trajectories
show projected controls. Advisory values are computed separately by the near_5
models. The selected backend is recorded in `feature_config.json` and
`training_manifest.json`; the verified current run uses scikit-learn histogram
gradient boosting. Model-based uplift is not a measured industrial effect.

## Run

```bash
# From the repository root:
uv sync
uv run python simulator/run.py
```

## Important Files

- `run.py` - application entry point.
- `app/window.py` - main window composition.
- `app/scene/` - drilling scene and layer generation.
- `app/charts/` - dashboard charts and 3D surface rendering.
- `app/data/advisory_engine.py` - advisory recommendation engine.
- `app/ml_artifacts/` - model artifacts used by the advisory engine.
- `simulator_core/` - legacy cluster artifacts used only by fallback mode.
