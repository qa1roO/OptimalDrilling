# Drilling Simulator Skeleton (Desktop App)

Test scaffold for a mining drilling rig simulator as a desktop application.

## Stack

- Python 3.11+
- PySide6 (UI)
- pyqtgraph + OpenGL (3D chart placeholder)

## Scope in this skeleton

- Left panel:
  - side view of drilling rig (placeholder rendering)
  - borehole cut
  - rock layer "pie" from stub generator function
- Right panel:
  - 3D chart placeholder with axes:
    - `Pressure`
    - `RPM`
    - `ROP` (rate of penetration)
- Stub model feed to be replaced by trained model output later.

## Project structure

- `run.py` - app entry point
- `app/window.py` - main window composition
- `app/scene/` - side-view scene and rock generation stub
- `app/charts/` - 3D chart widget stub
- `app/data/` - placeholder data from future model

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```
