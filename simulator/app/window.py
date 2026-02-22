from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from app.charts import Performance3DWidgetStub
from app.scene import SideViewWidget


class SimulatorMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OptimalDrilling Simulator (Desktop Skeleton)")
        self.resize(1600, 860)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        root_layout.addWidget(self._panel("Rig Side View + Borehole", SideViewWidget()), 1)
        root_layout.addWidget(self._panel("3D Performance Chart (Stub)", Performance3DWidgetStub()), 1)

        self.setCentralWidget(root)

    def _panel(self, title: str, content: QWidget) -> QWidget:
        frame = QFrame(self)
        frame.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #dce3ea; border-radius: 8px; }"
            "QLabel { color: #1c252d; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header = QLabel(title, frame)
        header.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(header)
        layout.addWidget(content, 1)
        return frame
