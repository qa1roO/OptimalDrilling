from pathlib import Path
import math

import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QBrush, QColor, QImage, QPen
from PySide6.QtWidgets import QGraphicsRectItem, QVBoxLayout, QWidget

from app.config import PanelLayout
from .rock_layer_generator_stub import generate_rock_layers_stub


class SideViewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout_cfg = PanelLayout()
        self.layers = generate_rock_layers_stub()
        self.rig_image = self._load_rig_image()
        self.depth_m = 0.0
        self.depth_dir = 1.0
        self._tick = 0.0
        self.setMinimumSize(760, 360)
        self._build_ui()
        self._start_depth_timer()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        canvas = pg.GraphicsLayoutWidget()
        canvas.setBackground("#fcfdff")
        layout.addWidget(canvas)

        rig_view = canvas.addViewBox(row=0, col=0, enableMouse=False)
        rig_view.setMenuEnabled(False)
        rig_view.setAspectLocked(True)
        rig_view.invertY(True)
        self._populate_rig_view(rig_view)

        section_view = canvas.addViewBox(row=1, col=0, enableMouse=False)
        section_view.setMenuEnabled(False)
        section_view.invertY(True)
        self._populate_section_view(section_view)

        canvas.ci.layout.setRowStretchFactor(0, 1)
        canvas.ci.layout.setRowStretchFactor(1, 1)

    def _load_rig_image(self) -> QImage | None:
        image_path = Path(self.layout_cfg.rig_image_path)
        if not image_path.is_absolute():
            image_path = Path(__file__).resolve().parents[2] / image_path
        if not image_path.exists():
            return None
        image = QImage(str(image_path))
        if image.isNull():
            return None
        return image

    def _populate_rig_view(self, view: pg.ViewBox) -> None:
        cfg = self.layout_cfg
        if self.rig_image is not None:
            image_item = pg.ImageItem(self.rig_image)
            view.addItem(image_item)
            view.setRange(
                QRectF(0, 0, self.rig_image.width(), self.rig_image.height()),
                padding=0.04,
            )
            self._add_pressure_labels(view, self.rig_image.width(), self.rig_image.height())
            return

        placeholder = QGraphicsRectItem(0, 0, cfg.rig_width, cfg.rig_height)
        placeholder.setBrush(QBrush(QColor("#eff4f9")))
        placeholder.setPen(QPen(QColor("#27323d"), 2))
        view.addItem(placeholder)

        label = pg.TextItem("Rig image missing", anchor=(0.5, 0.5), color="#1f2933")
        label.setPos(cfg.rig_width / 2, cfg.rig_height / 2)
        view.addItem(label)
        self._add_pressure_labels(view, cfg.rig_width, cfg.rig_height)
        view.setRange(QRectF(0, 0, cfg.rig_width, cfg.rig_height), padding=0.05)

    def _add_pressure_labels(self, view: pg.ViewBox, width: float, height: float) -> None:
        self.axial_label = pg.TextItem("Axial pressure: 0 kN", anchor=(0, 0), color="#1f2933")
        self.rot_label = pg.TextItem("Rotational pressure: 0 kN·m", anchor=(0, 0), color="#1f2933")
        self.axial_label.setPos(width * 0.05, height * 0.08)
        self.rot_label.setPos(width * 0.05, height * 0.16)
        view.addItem(self.axial_label)
        view.addItem(self.rot_label)

    def _populate_section_view(self, view: pg.ViewBox) -> None:
        cfg = self.layout_cfg
        w, h = cfg.section_width, cfg.section_height
        frame = QGraphicsRectItem(0, 0, w, h)
        frame.setBrush(QBrush(QColor("#eff4f9")))
        frame.setPen(QPen(QColor("#27323d"), 2))
        view.addItem(frame)

        total_thickness = sum(layer.thickness for layer in self.layers) or 1.0
        y_offset = 0.0
        for layer in self.layers:
            layer_h = (layer.thickness / total_thickness) * h
            rect = QGraphicsRectItem(0, y_offset, w, layer_h)
            rect.setBrush(QBrush(QColor(layer.color_hex)))
            rect.setPen(QPen(QColor("#576878"), 1))
            view.addItem(rect)

            label = pg.TextItem(layer.name, anchor=(0, 0), color="#1f2933")
            label.setPos(6, y_offset + min(12, max(layer_h - 14, 2)))
            view.addItem(label)
            y_offset += layer_h

        borehole_x = w * 0.55
        borehole = QGraphicsRectItem(borehole_x - 6, 4, 12, h - 8)
        borehole.setBrush(QBrush(QColor("#222f3a")))
        borehole.setPen(QPen(QColor("#222f3a"), 1))
        view.addItem(borehole)

        label = pg.TextItem("Borehole cut", anchor=(0, 0), color="#f0f7ff")
        label.setPos(6, 6)
        view.addItem(label)

        self.depth_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#ff8c42", width=2))
        self.depth_line.setPos(0)
        view.addItem(self.depth_line)

        self.depth_label = pg.TextItem("Depth: 0 m", anchor=(0, 0), color="#ff8c42")
        self.depth_label.setPos(6, 22)
        view.addItem(self.depth_label)

        self.section_height = h
        view.setRange(QRectF(0, 0, w, h), padding=0.04)

    def _start_depth_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._update_depth)
        self.timer.start()

    def _update_depth(self) -> None:
        if not hasattr(self, "section_height"):
            return

        max_depth = max(self.section_height - 10, 1)
        step = 1.6 * self.depth_dir
        self.depth_m += step
        if self.depth_m >= max_depth:
            self.depth_m = max_depth
            self.depth_dir = -1.0
        elif self.depth_m <= 0:
            self.depth_m = 0
            self.depth_dir = 1.0

        self._tick += 0.08
        axial = 420 + 40 * math.sin(self._tick)
        rot = 120 + 15 * math.cos(self._tick * 0.8)
        if hasattr(self, "axial_label"):
            self.axial_label.setText(f"Axial pressure: {axial:0.0f} kN")
        if hasattr(self, "rot_label"):
            self.rot_label.setText(f"Rotational pressure: {rot:0.0f} kN·m")

        if hasattr(self, "depth_line"):
            self.depth_line.setPos(self.depth_m)
        if hasattr(self, "depth_label"):
            self.depth_label.setText(f"Depth: {self.depth_m:0.0f} m")
            self.depth_label.setPos(6, min(self.depth_m + 6, max_depth - 8))
