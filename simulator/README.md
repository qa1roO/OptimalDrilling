# OptimalDrilling Simulator

Desktop simulator for drilling replay and advisory-model visualization.

## Current Flow

The application starts from `run.py`, creates `SimulatorMainWindow`, and shows two panels:

- `app/scene/side_view_widget.py` - rig side view, borehole animation, replay well selection, telemetry signals.
- `app/charts/performance_3d_widget_stub.py` - performance dashboard with depth-based 2D plots and a 3D empirical energy surface.

The main runtime path is replay/advisory mode:

1. `SideViewWidget` reads replay telemetry from `united_rock_energy_segment_quantile.csv` when it is available.
2. It emits `replay_sample(row, depth_m)` while the rig animation advances.
3. `Performance3DWidget.append_advisory_telemetry()` updates the 2D charts and the 3D global speed response surface.
4. `AdvisoryEngine` loads joblib/LightGBM artifacts from `app/ml_artifacts`.

If replay telemetry is unavailable, the simulator can fall back to synthetic layer drilling and legacy cluster profiles from `simulator_core`.

The 3D background is a single smoothed empirical binned-median speed surface built from all factual rows, not one surface per energy quantile. Rock energy quantiles remain model features and UI/report context. The 3D surface is used only for visualization: white and red trajectories show projected controls on the global speed response surface. Advisory recommendation values and uplift are computed separately by the near_5 LightGBM models.

## Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Important Files

- `run.py` - application entry point.
- `app/window.py` - main window composition.
- `app/scene/` - drilling scene and layer generation.
- `app/charts/` - dashboard charts and 3D surface rendering.
- `app/data/advisory_engine.py` - advisory recommendation engine.
- `app/ml_artifacts/` - model artifacts used by the advisory engine.
- `simulator_core/` - legacy cluster artifacts used only by fallback mode.
