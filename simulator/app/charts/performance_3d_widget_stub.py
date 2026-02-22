from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.data import get_model_feed_placeholder

try:
    import numpy as np
    import pyqtgraph.opengl as gl

    OPENGL_READY = True
except Exception:  # pragma: no cover
    OPENGL_READY = False


class Performance3DWidgetStub(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(760, 360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if OPENGL_READY:
            layout.addWidget(self._build_gl_view())
        else:
            warning = QLabel(
                "OpenGL backend is unavailable.\n"
                "Install dependencies from requirements.txt.",
                self,
            )
            warning.setStyleSheet("padding: 16px; color: #8a1f1f;")
            layout.addWidget(warning)

    def _build_gl_view(self) -> QWidget:
        view = gl.GLViewWidget()
        view.opts["distance"] = 220
        view.setBackgroundColor((252, 253, 255, 255))

        axis = gl.GLAxisItem()
        axis.setSize(x=100, y=100, z=100)
        view.addItem(axis)

        points = get_model_feed_placeholder()
        data = np.array([[p.pressure, p.rpm, p.rop * 20.0] for p in points], dtype=float)

        line = gl.GLLinePlotItem(
            pos=data,
            color=(0.12, 0.47, 0.71, 1.0),
            width=2,
            antialias=True,
            mode="line_strip",
        )
        scatter = gl.GLScatterPlotItem(
            pos=data,
            color=(0.12, 0.47, 0.71, 1.0),
            size=7,
            pxMode=True,
        )
        view.addItem(line)
        view.addItem(scatter)
        return view
