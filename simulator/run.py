import sys
from pathlib import Path

SIMULATOR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SIMULATOR_DIR.parent
for path in (PROJECT_ROOT, SIMULATOR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.main import run


if __name__ == "__main__":
    raise SystemExit(run())
