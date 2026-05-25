"""Static simulator parameters.

Keep visual constants here so tuning does not require searching through
widget rendering code.
"""

# 2D chart axes.
CHART_DEPTH_AXIS_MAX_M = 40.0
CHART_ROTATION_AXIS_RANGE = (45.01, 144.578)
CHART_SPEED_AXIS_RANGE = (0.0, 0.0409571428571429)

# Full replay CSV statistics used to derive the chart ranges above.
REPLAY_CSV_STATS = {
    "source": "notebooks/united_rock_energy_segment_quantile.csv",
    "rows": 415_049,
    "rotation_min": 50.01,
    "rotation_max": 139.578,
    "speed_min": 0.0010016528925619,
    "speed_max": 0.0389571428571429,
    "depth_min": -0.848399999999999,
    "depth_max": 82.0524,
}

# 2D chart rendering.
CHART_SPEED_SMOOTHING_WINDOW = 35
CHART_ROTATION_SMOOTHING_WINDOW = 9
CHART_DEPTH_BIN_SIZE_M = 0.2
CHART_UPDATE_STRIDE = 3
ADVISORY_UPDATE_STRIDE = 10

# Replay pacing. The timer is adapted per replay well so the downward drilling
# pass stays close to 2-3 minutes without changing drilling values.
PLAYBACK_TIMER_INTERVAL_MS = 120
PLAYBACK_TARGET_WELL_DURATION_MS = 150_000
SIMULATION_STEP_S = 0.04
