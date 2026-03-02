import math

import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QVBoxLayout,
    QWidget,
)

from .rock_layer_generator_stub import generate_rock_layers_stub


class SideViewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layers = generate_rock_layers_stub()
        self.depth_m = 0.0
        self.depth_dir = 1.0
        self._tick = 0.0

        self.scene_w = 880.0
        self.scene_h = 700.0
        self.rig_top = 28.0
        self.deck_y = 230.0
        self.section_top = 360.0
        self.section_bottom = 680.0
        self.max_depth_m = 1800.0
        self.borehole_x = 350.0
        self.drill_top_y = 126.0

        self.setMinimumSize(760, 520)
        self._build_ui()
        self._start_timer()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        canvas = pg.GraphicsLayoutWidget()
        canvas.setBackground("#f5f7fb")
        layout.addWidget(canvas)

        self.view = canvas.addViewBox(row=0, col=0, enableMouse=False)
        self.view.setMenuEnabled(False)
        self.view.invertY(True)
        self.view.setRange(QRectF(0, 0, self.scene_w, self.scene_h), padding=0.02)

        self._draw_scene()

    def _draw_scene(self) -> None:
        self._draw_rig_platform()
        self._draw_mast()
        self._draw_power_units()
        self._draw_section_and_borehole()
        self._draw_measure_labels()
        self._update_kinematics()

    def _draw_rig_platform(self) -> None:
        deck = QGraphicsRectItem(120, self.deck_y, 620, 56)
        deck.setBrush(QBrush(QColor("#29354a")))
        deck.setPen(QPen(QColor("#1d2433"), 2))
        self.view.addItem(deck)

        skirt = QGraphicsRectItem(140, self.deck_y + 56, 580, 16)
        skirt.setBrush(QBrush(QColor("#202a39")))
        skirt.setPen(QPen(QColor("#202a39"), 1))
        self.view.addItem(skirt)

        for x_pos in (132, 724):
            support = QGraphicsRectItem(x_pos, self.deck_y + 56, 12, 34)
            support.setBrush(QBrush(QColor("#e8d06b")))
            support.setPen(QPen(QColor("#b89f42"), 1))
            self.view.addItem(support)

        for x_pos in (100, 470):
            track = QGraphicsRectItem(x_pos, self.deck_y + 76, 220, 34)
            track.setBrush(QBrush(QColor("#1d2433")))
            track.setPen(QPen(QColor("#0f1520"), 2))
            self.view.addItem(track)

            for idx in range(6):
                wheel = QGraphicsEllipseItem(x_pos + 12 + idx * 34, self.deck_y + 84, 18, 18)
                wheel.setBrush(QBrush(QColor("#313f56")))
                wheel.setPen(QPen(QColor("#121a28"), 1))
                self.view.addItem(wheel)

        for x_pos in range(145, 730, 48):
            rail = QGraphicsRectItem(x_pos, self.deck_y - 8, 32, 8)
            rail.setBrush(QBrush(QColor("#efdb84")))
            rail.setPen(QPen(QColor("#b89f42"), 1))
            self.view.addItem(rail)

    def _draw_mast(self) -> None:
        mast_x = self.borehole_x - 48
        mast_w = 96
        mast_h = self.deck_y - self.rig_top

        mast = QGraphicsRectItem(mast_x, self.rig_top, mast_w, mast_h)
        mast.setBrush(QBrush(QColor("#2a3550")))
        mast.setPen(QPen(QColor("#1a2233"), 3))
        self.view.addItem(mast)

        left_rail = QGraphicsRectItem(mast_x + 12, self.rig_top + 8, 9, mast_h - 16)
        left_rail.setBrush(QBrush(QColor("#101726")))
        left_rail.setPen(QPen(QColor("#101726"), 1))
        self.view.addItem(left_rail)

        right_rail = QGraphicsRectItem(mast_x + mast_w - 21, self.rig_top + 8, 9, mast_h - 16)
        right_rail.setBrush(QBrush(QColor("#101726")))
        right_rail.setPen(QPen(QColor("#101726"), 1))
        self.view.addItem(right_rail)

        pen = QPen(QColor("#566684"), 2)
        for y_pos in range(int(self.rig_top + 12), int(self.deck_y - 10), 28):
            brace_a = QGraphicsLineItem(mast_x + 22, y_pos, mast_x + mast_w - 22, y_pos + 20)
            brace_a.setPen(pen)
            self.view.addItem(brace_a)

            brace_b = QGraphicsLineItem(mast_x + mast_w - 22, y_pos, mast_x + 22, y_pos + 20)
            brace_b.setPen(pen)
            self.view.addItem(brace_b)

        self.carriage = QGraphicsRectItem(self.borehole_x - 34, 126, 68, 24)
        self.carriage.setBrush(QBrush(QColor("#f0d97a")))
        self.carriage.setPen(QPen(QColor("#9f8342"), 2))
        self.view.addItem(self.carriage)

        self.hydraulic_motor = QGraphicsRectItem(self.borehole_x - 16, 154, 32, 32)
        self.hydraulic_motor.setBrush(QBrush(QColor("#e8ce66")))
        self.hydraulic_motor.setPen(QPen(QColor("#9f8342"), 2))
        self.view.addItem(self.hydraulic_motor)

        self.motor_shaft = QGraphicsRectItem(self.borehole_x - 5, 186, 10, 8)
        self.motor_shaft.setBrush(QBrush(QColor("#d5bf63")))
        self.motor_shaft.setPen(QPen(QColor("#9f8342"), 1))
        self.view.addItem(self.motor_shaft)

    def _draw_power_units(self) -> None:
        engine_box = QGraphicsRectItem(430, 190, 190, 74)
        engine_box.setBrush(QBrush(QColor("#e8ce66")))
        engine_box.setPen(QPen(QColor("#9f8342"), 2))
        self.view.addItem(engine_box)

        cabin = QGraphicsRectItem(162, 216, 118, 52)
        cabin.setBrush(QBrush(QColor("#e8ce66")))
        cabin.setPen(QPen(QColor("#9f8342"), 2))
        self.view.addItem(cabin)

    def _draw_section_and_borehole(self) -> None:
        section_frame = QGraphicsRectItem(80, self.section_top, 760, self.section_bottom - self.section_top)
        section_frame.setBrush(QBrush(QColor("#ecf0f6")))
        section_frame.setPen(QPen(QColor("#4a586e"), 2))
        self.view.addItem(section_frame)

        total_thickness = sum(layer.thickness for layer in self.layers) or 1.0
        y_offset = self.section_top
        for layer in self.layers:
            layer_h = (layer.thickness / total_thickness) * (self.section_bottom - self.section_top)
            rect = QGraphicsRectItem(82, y_offset, 756, layer_h)
            rect.setBrush(QBrush(QColor(layer.color_hex)))
            rect.setPen(QPen(QColor("#5f6d83"), 1))
            self.view.addItem(rect)

            label = pg.TextItem(layer.name, anchor=(0, 0), color="#1f2933")
            label.setPos(90, y_offset + 6)
            self.view.addItem(label)
            y_offset += layer_h

        self.borehole = QGraphicsRectItem(
            self.borehole_x - 9,
            self.section_top + 4,
            18,
            (self.section_bottom - self.section_top) - 8,
        )
        self.borehole.setBrush(QBrush(QColor("#18202e")))
        self.borehole.setPen(QPen(QColor("#18202e"), 1))
        self.view.addItem(self.borehole)

        self.drill_string = QGraphicsRectItem(self.borehole_x - 1.5, self.drill_top_y, 3, 10)
        self.drill_string.setBrush(QBrush(QColor("#1a2333")))
        self.drill_string.setPen(QPen(QColor("#0f1520"), 1))
        self.view.addItem(self.drill_string)

        self.bit_body = QGraphicsRectItem(self.borehole_x - 6, self.section_top + 8, 12, 9)
        self.bit_body.setBrush(QBrush(QColor("#f1c15a")))
        self.bit_body.setPen(QPen(QColor("#a8782c"), 2))
        self.view.addItem(self.bit_body)

        self.bit_cone_left = QGraphicsEllipseItem(self.borehole_x - 9, self.section_top + 14, 6, 7)
        self.bit_cone_left.setBrush(QBrush(QColor("#ec8a45")))
        self.bit_cone_left.setPen(QPen(QColor("#a5542c"), 1))
        self.view.addItem(self.bit_cone_left)

        self.bit_cone_mid = QGraphicsEllipseItem(self.borehole_x - 3, self.section_top + 15, 6, 7)
        self.bit_cone_mid.setBrush(QBrush(QColor("#ec8a45")))
        self.bit_cone_mid.setPen(QPen(QColor("#a5542c"), 1))
        self.view.addItem(self.bit_cone_mid)

        self.bit_cone_right = QGraphicsEllipseItem(self.borehole_x + 3, self.section_top + 14, 6, 7)
        self.bit_cone_right.setBrush(QBrush(QColor("#ec8a45")))
        self.bit_cone_right.setPen(QPen(QColor("#a5542c"), 1))
        self.view.addItem(self.bit_cone_right)

        self.depth_line = QGraphicsLineItem(370, self.section_top + 10, 820, self.section_top + 10)
        self.depth_line.setPen(QPen(QColor("#ff8c42"), 2))
        self.view.addItem(self.depth_line)

        self.depth_text = pg.TextItem("Depth: 0 m", anchor=(0, 0), color="#ff8c42")
        self.depth_text.setPos(600, self.section_top + 14)
        self.view.addItem(self.depth_text)

    def _draw_measure_labels(self) -> None:
        self.axial_text = pg.TextItem("Axial pressure: 0 kN", anchor=(0, 0), color="#122238")
        self.axial_text.setPos(432, 72)
        self.view.addItem(self.axial_text)

        self.rot_text = pg.TextItem("Rotational pressure: 0 kN*m", anchor=(0, 0), color="#122238")
        self.rot_text.setPos(432, 100)
        self.view.addItem(self.rot_text)

    def _start_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._update_kinematics)
        self.timer.start()

    def _depth_to_y(self, depth_m: float) -> float:
        travel = (self.section_bottom - self.section_top) - 18
        return (self.section_top + 8) + travel * (depth_m / self.max_depth_m)

    def _update_kinematics(self) -> None:
        self._tick += 0.06
        step = 9.0 * self.depth_dir
        self.depth_m += step

        if self.depth_m >= self.max_depth_m:
            self.depth_m = self.max_depth_m
            self.depth_dir = -1.0
        elif self.depth_m <= 0.0:
            self.depth_m = 0.0
            self.depth_dir = 1.0

        motor_y = 154 + 4 * math.sin(self._tick * 1.4)
        self.hydraulic_motor.setRect(self.borehole_x - 16, motor_y, 32, 32)
        self.motor_shaft.setRect(self.borehole_x - 5, motor_y + 32, 10, 8)
        self.drill_top_y = motor_y + 40

        bit_y = self._depth_to_y(self.depth_m)
        string_h = max(bit_y - self.drill_top_y, 6)
        self.drill_string.setRect(self.borehole_x - 1.5, self.drill_top_y, 3, string_h)
        self._set_tricone_position(bit_y)
        self.depth_line.setLine(370, bit_y, 820, bit_y)
        self.depth_text.setText(f"Depth: {self.depth_m:0.0f} m")
        self.depth_text.setPos(600, min(bit_y + 6, self.section_bottom - 24))

        axial = 430 + 34 * math.sin(self._tick)
        rotational = 118 + 14 * math.cos(self._tick * 0.9)
        self.axial_text.setText(f"Axial pressure: {axial:0.0f} kN")
        self.rot_text.setText(f"Rotational pressure: {rotational:0.0f} kN*m")

    def _set_tricone_position(self, bit_center_y: float) -> None:
        body_y = bit_center_y - 10
        self.bit_body.setRect(self.borehole_x - 6, body_y, 12, 9)
        self.bit_cone_left.setRect(self.borehole_x - 9, body_y + 6, 6, 7)
        self.bit_cone_mid.setRect(self.borehole_x - 3, body_y + 7, 6, 7)
        self.bit_cone_right.setRect(self.borehole_x + 3, body_y + 6, 6, 7)
