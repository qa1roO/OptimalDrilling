from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
IMAGES_DIR = PROJECT_ROOT / "images"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SIMULATOR_DIR = PROJECT_ROOT / "simulator"

RAW_DATA_PATH = DATASETS_DIR / "united.csv"
LABELED_DATA_PATH = NOTEBOOKS_DIR / "united_rock_energy_segment_quantile.csv"

ROCK_ENERGY_CONFIG_PATH = NOTEBOOKS_DIR / "rock_energy_segment_quantile_config.json"
ROCK_ENERGY_ARTIFACT_DIR = NOTEBOOKS_DIR / "rock_energy_segment_quantile_artifacts"
ROCK_ENERGY_REPORT_DIR = NOTEBOOKS_DIR / "rock_energy_segment_reports"
PLOTLY_SURFACES_DIR = NOTEBOOKS_DIR / "plotly_surfaces_html"

DRILLING_ADVISORY_ARTIFACT_DIR = NOTEBOOKS_DIR / "drilling_advisory_light_penalty_artifacts"
DRILLING_ADVISORY_REPORT_DIR = NOTEBOOKS_DIR / "drilling_advisory_light_penalty_reports"

SIMULATOR_ML_ARTIFACT_ROOT = SIMULATOR_DIR / "app" / "ml_artifacts"
SIMULATOR_ADVISORY_ARTIFACT_DIR = (
    SIMULATOR_ML_ARTIFACT_ROOT / "drilling_advisory_light_penalty_artifacts"
)

ENERGY_LABELS_4 = [
    "soft_low_energy",
    "medium_low_energy",
    "medium_high_energy",
    "hard_high_energy",
]

REQUIRED_TELEMETRY_COLUMNS = [
    "processing_time",
    "depth_m",
    "well_id",
    "pressure_axis",
    "pressure_rotation",
    "rotation",
    "speed",
]

RANDOM_STATE = 42
EPS = 1e-9

TARGET_HORIZON = 5
SEGMENT_SIZE = 60
ROLL_WINDOWS = [12, 30, 60]

GRID_SIZE = 21
MAX_DELTA_FRAC = 0.08
FINAL_OPTIMIZER_MODE = "light_penalty"
CHANGE_PENALTY_WEIGHT = 0.010
BOUNDARY_PENALTY_WEIGHT = 0.020
BOUNDARY_START = 0.85


def ensure_output_dirs() -> None:
    """Create repository output folders used by notebooks."""
    for path in [
        ROCK_ENERGY_ARTIFACT_DIR,
        ROCK_ENERGY_REPORT_DIR,
        PLOTLY_SURFACES_DIR,
        DRILLING_ADVISORY_ARTIFACT_DIR,
        DRILLING_ADVISORY_REPORT_DIR,
        SIMULATOR_ADVISORY_ARTIFACT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
