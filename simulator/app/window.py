from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from app.charts import Performance3DWidget
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

        self.side_view = SideViewWidget()
        self.performance_chart = Performance3DWidget()
        self.side_view.drilling_sample.connect(self.performance_chart.append_drilling_point)
        self.side_view.drilling_cycle_started.connect(self.performance_chart.on_drilling_cycle_started)

        root_layout.addWidget(self._panel("Rig Side View + Borehole", self.side_view), 1)
        root_layout.addWidget(
            self._panel(
                "3D Performance Chart",
                self.performance_chart,
            ),
            1,
        )

        self.setCentralWidget(root)

    def _panel(self, title: str, content: QWidget, controls: list[QWidget] | None = None) -> QWidget:
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
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(header)
        header_layout.addStretch(1)
        for control in controls or []:
            header_layout.addWidget(control)

        layout.addLayout(header_layout)
        layout.addWidget(content, 1)
        return frame
