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
        self.setWindowTitle("OptimalDrilling Simulator")
        self.resize(1600, 860)
        self.setStyleSheet(
            """
            QMainWindow {
                background: #090d13;
            }
            QWidget#RootSurface {
                background: #090d13;
            }
            QFrame#SimulatorPanel {
                background: #111821;
                border: 1px solid #223041;
                border-radius: 8px;
            }
            QLabel#PanelTitle {
                color: #dce6f1;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0px;
            }
            QLabel#PanelBadge {
                color: #7f92a8;
                background: #0b1118;
                border: 1px solid #223041;
                border-radius: 4px;
                padding: 3px 7px;
                font-size: 10px;
                font-weight: 600;
            }
            """
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("RootSurface")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        self.side_view = SideViewWidget()
        self.performance_chart = Performance3DWidget()
        self.side_view.drilling_sample.connect(self.performance_chart.append_drilling_point)
        self.side_view.drilling_cycle_started.connect(self.performance_chart.on_drilling_cycle_started)

        root_layout.addWidget(self._panel("Rig Side View", self.side_view, badge="LIVE"), 1)
        root_layout.addWidget(
            self._panel(
                "Performance Dashboard",
                self.performance_chart,
                badge="MODEL",
            ),
            1,
        )

        self.setCentralWidget(root)

    def _panel(
        self,
        title: str,
        content: QWidget,
        controls: list[QWidget] | None = None,
        badge: str | None = None,
    ) -> QWidget:
        frame = QFrame(self)
        frame.setObjectName("SimulatorPanel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QLabel(title, frame)
        header.setObjectName("PanelTitle")
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(header)
        header_layout.addStretch(1)
        if badge:
            badge_label = QLabel(badge, frame)
            badge_label.setObjectName("PanelBadge")
            header_layout.addWidget(badge_label)
        for control in controls or []:
            header_layout.addWidget(control)

        layout.addLayout(header_layout)
        layout.addWidget(content, 1)
        return frame
