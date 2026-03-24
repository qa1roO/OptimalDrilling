"""
SideViewWidget — боковой вид бурового станка PV-271.

Z-порядок отрисовки (снизу вверх):
  1. Геология (породный пирог)
  2. Пробуренная скважина (серый канал)
  3. Труба бурильной колонны
  4. Статичный станок (мачта, шасси, кабина)
  5. Каретка с мотором (движется по мачте)
  6. Долото (поверх всего — никогда не перекрывается)
  7. Лейблы / текст

Логика анимации:
  - depth_m растёт от 0 до MAX_DEPTH_M (только вниз)
  - bit_y = SECTION_TOP + SECTION_H * (depth_m / MAX_DEPTH_M)
  - Каретка: едет от CARRIAGE_TOP_Y до CARRIAGE_BOT_Y
  - Труба: верх = низ вала мотора, низ = bit_y (всегда жёстко)
  - Скважина: от SECTION_TOP до bit_y (серый прямоугольник)
  - При сбросе: только геология меняется, всё остальное — setRect в начало
"""

import math
import random

import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QVBoxLayout,
    QWidget,
)

from .rock_layer_generator_stub import generate_rock_layers

# ---------- Геометрия сцены ----------
SCENE_W = 900.0
SCENE_H = 980.0

MAST_X = 310.0
MAST_W = 60.0
MAST_TOP_Y = 20.0
MAST_BOT_Y = 540.0

CHASSIS_Y = MAST_BOT_Y + 46.0
CHASSIS_H = 56.0
TRACK_Y = CHASSIS_Y + CHASSIS_H
TRACK_H = 30.0

CARRIAGE_W = 52.0
CARRIAGE_H = 28.0
CARRIAGE_TOP_Y = MAST_TOP_Y + 10
CARRIAGE_BOT_Y = MAST_BOT_Y - 60
ROTARY_W = CARRIAGE_W + 16.0
ROTARY_HEAD_H = 16.0
ROTARY_NECK_W = 20.0
ROTARY_NECK_H = 14.0

SECTION_LEFT = 60.0
SECTION_TOP = TRACK_Y + TRACK_H
SECTION_RIGHT = 860.0
SECTION_BOTTOM = SCENE_H - 20.0
SECTION_W = SECTION_RIGHT - SECTION_LEFT
SECTION_H = SECTION_BOTTOM - SECTION_TOP

PIPE_X = MAST_X
PIPE_W = 5.0
BOREHOLE_W = 13.0
BIT_BODY_H = 8.0
BIT_CONE_H = 8.0
BIT_CONE_OFFSET_Y = 9.0

# Реалистичный рабочий диапазон одной буровой штанги.
MIN_DEPTH_M = 22.0
MAX_DEPTH_M = 27.0
DEPTH_STEP_M = 0.06

Z_GEOLOGY = -30.0
Z_GEOLOGY_LABEL = -25.0
Z_BOREHOLE = -10.0
Z_PIPE = -5.0
Z_RIG = 10.0
Z_CARRIAGE = 20.0
Z_BIT = 30.0
Z_OVERLAY = 40.0


class SideViewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tick = 0.0
        self.depth_m = 0.0
        self.target_depth_m = random.uniform(MIN_DEPTH_M, MAX_DEPTH_M)
        self.region, self.layers = generate_rock_layers()
        self._geology_items: list = []

        self.setMinimumSize(780, 580)
        self._build_ui()
        self._start_timer()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        canvas = pg.GraphicsLayoutWidget()
        canvas.setBackground("#e8ecf2")
        layout.addWidget(canvas)

        self.view = canvas.addViewBox(row=0, col=0, enableMouse=False)
        self.view.setMenuEnabled(False)
        self.view.invertY(True)
        self.view.setRange(QRectF(0, 0, SCENE_W, SCENE_H), padding=0.01)

        # Порядок вызовов = z-порядок
        self._draw_geology()          # 1. фон породы
        self._draw_borehole_pipe()    # 2. скважина + труба
        self._draw_rig_static()       # 3. станок (поверх породы)
        self._draw_carriage()         # 4. каретка (поверх мачты)
        self._draw_bit()              # 5. долото (самый верх)
        self._draw_depth_indicator()
        self._draw_labels()

    # ------------------------------------------------------------------ Геология

    def _add_item(self, item, z: float):
        item.setZValue(z)
        self.view.addItem(item)
        return item

    def _draw_geology(self) -> None:
        bg = QGraphicsRectItem(SECTION_LEFT, SECTION_TOP, SECTION_W, SECTION_H)
        bg.setBrush(QBrush(QColor("#d8dde6")))
        bg.setPen(QPen(QColor("#4a586e"), 2))
        self._add_item(bg, Z_GEOLOGY)
        self._geology_items.append(bg)

        total = sum(l.thickness for l in self.layers) or 1.0
        y = SECTION_TOP
        for layer in self.layers:
            h = (layer.thickness / total) * SECTION_H
            rect = QGraphicsRectItem(SECTION_LEFT + 2, y, SECTION_W - 4, h)
            rect.setBrush(QBrush(QColor(layer.color_hex)))
            rect.setPen(QPen(QColor("#5f6d83"), 1))
            self._add_item(rect, Z_GEOLOGY)
            self._geology_items.append(rect)

            lbl = pg.TextItem(layer.name, anchor=(0, 0), color="#1f2933")
            lbl.setPos(SECTION_LEFT + 8, y + 3)
            self._add_item(lbl, Z_GEOLOGY_LABEL)
            self._geology_items.append(lbl)
            y += h

    def _clear_geology(self) -> None:
        for item in self._geology_items:
            self.view.removeItem(item)
        self._geology_items.clear()

    # ------------------------------------------------------------------ Скважина + труба

    def _draw_borehole_pipe(self) -> None:
        bx = PIPE_X

        # Скважина — серый канал (h=0 в начале)
        self.drilled_hole = QGraphicsRectItem(
            bx - BOREHOLE_W / 2, SECTION_TOP, BOREHOLE_W, 0
        )
        self.drilled_hole.setBrush(QBrush(QColor("#4a5060")))
        self.drilled_hole.setPen(QPen(QColor("#4a5060"), 0))
        self._add_item(self.drilled_hole, Z_BOREHOLE)

        # Труба (h=0 в начале — растянется в update)
        self.drill_pipe = QGraphicsRectItem(bx - PIPE_W / 2, MAST_BOT_Y, PIPE_W, 0)
        self.drill_pipe.setBrush(QBrush(QColor("#b0bac8")))
        self.drill_pipe.setPen(QPen(QColor("#6a7888"), 1))
        self._add_item(self.drill_pipe, Z_PIPE)

        # Металлический блик на трубе
        self.pipe_shine = QGraphicsRectItem(bx - PIPE_W / 2, MAST_BOT_Y, 1.5, 0)
        self.pipe_shine.setBrush(QBrush(QColor("#dde4f0")))
        self.pipe_shine.setPen(QPen(QColor("#dde4f0"), 0))
        self._add_item(self.pipe_shine, Z_PIPE + 1)

    # ------------------------------------------------------------------ Станок (статика)

    def _draw_rig_static(self) -> None:
        self._draw_mast_structure()
        self._draw_mast_support()
        self._draw_chassis()
        self._draw_cabin()
        self._draw_engine_deck()

    def _draw_mast_structure(self) -> None:
        mx = MAST_X - MAST_W / 2
        mh = MAST_BOT_Y - MAST_TOP_Y

        # Фон мачты
        mast_bg = QGraphicsRectItem(mx, MAST_TOP_Y, MAST_W, mh)
        mast_bg.setBrush(QBrush(QColor("#3a4560")))
        mast_bg.setPen(QPen(QColor("#252e42"), 2))
        self._add_item(mast_bg, Z_RIG)

        # Направляющие рельсы
        for dx in (5, MAST_W - 13):
            rail = QGraphicsRectItem(mx + dx, MAST_TOP_Y + 4, 8, mh - 8)
            rail.setBrush(QBrush(QColor("#1e2636")))
            rail.setPen(QPen(QColor("#1e2636"), 1))
            self._add_item(rail, Z_RIG + 0.1)

        # Раскосы крест-накрест
        pen_b = QPen(QColor("#4e5f80"), 1)
        pen_h = QPen(QColor("#4e5f80"), 1)
        step = 30
        for y0 in range(int(MAST_TOP_Y + 8), int(MAST_BOT_Y - 10), step):
            y1 = y0 + step
            la = QGraphicsLineItem(mx + 13, y0, mx + MAST_W - 13, y1)
            la.setPen(pen_b)
            self._add_item(la, Z_RIG + 0.1)
            lb = QGraphicsLineItem(mx + MAST_W - 13, y0, mx + 13, y1)
            lb.setPen(pen_b)
            self._add_item(lb, Z_RIG + 0.1)
            hl = QGraphicsLineItem(mx + 13, y0, mx + MAST_W - 13, y0)
            hl.setPen(pen_h)
            self._add_item(hl, Z_RIG + 0.1)

        # Голова мачты
        head = QGraphicsRectItem(mx + 6, MAST_TOP_Y, MAST_W - 12, 22)
        head.setBrush(QBrush(QColor("#e8ce50")))
        head.setPen(QPen(QColor("#9f8c2a"), 2))
        self._add_item(head, Z_RIG + 0.2)

        # Шкив
        pulley = QGraphicsEllipseItem(MAST_X - 8, MAST_TOP_Y - 7, 16, 16)
        pulley.setBrush(QBrush(QColor("#2a3040")))
        pulley.setPen(QPen(QColor("#1a1f2e"), 2))
        self._add_item(pulley, Z_RIG + 0.3)

        # Тросы
        pen_c = QPen(QColor("#8899aa"), 1)
        for cx in (MAST_X - 4, MAST_X + 4):
            c = QGraphicsLineItem(cx, MAST_TOP_Y + 7, cx, MAST_BOT_Y)
            c.setPen(pen_c)
            self._add_item(c, Z_RIG + 0.2)

    def _draw_mast_support(self) -> None:
        pedestal_w = 30.0
        pedestal_x = MAST_X - pedestal_w / 2
        pedestal_h = CHASSIS_Y - MAST_BOT_Y + 8

        pedestal = QGraphicsRectItem(pedestal_x, MAST_BOT_Y - 4, pedestal_w, pedestal_h)
        pedestal.setBrush(QBrush(QColor("#55657d")))
        pedestal.setPen(QPen(QColor("#2b3446"), 2))
        self._add_item(pedestal, Z_RIG - 0.2)

        foot = QGraphicsRectItem(MAST_X - 52, CHASSIS_Y - 10, 104, 12)
        foot.setBrush(QBrush(QColor("#6f8099")))
        foot.setPen(QPen(QColor("#374255"), 2))
        self._add_item(foot, Z_RIG - 0.1)

        brace_pen = QPen(QColor("#425066"), 5)
        for x1, x2 in ((MAST_X - 12, MAST_X - 84), (MAST_X + 12, MAST_X + 84)):
            brace = QGraphicsLineItem(x1, MAST_BOT_Y + 8, x2, CHASSIS_Y + 4)
            brace.setPen(brace_pen)
            self._add_item(brace, Z_RIG - 0.1)

    def _draw_chassis(self) -> None:
        body = QGraphicsRectItem(120, CHASSIS_Y, 620, CHASSIS_H)
        body.setBrush(QBrush(QColor("#d4a800")))
        body.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(body, Z_RIG)

        for tx in (128, 476):
            track = QGraphicsRectItem(tx, TRACK_Y, 212, TRACK_H)
            track.setBrush(QBrush(QColor("#1a1f2a")))
            track.setPen(QPen(QColor("#0d1017"), 2))
            self._add_item(track, Z_RIG + 0.1)
            for i in range(7):
                w = QGraphicsEllipseItem(tx + 8 + i * 28, TRACK_Y + 5, 20, 20)
                w.setBrush(QBrush(QColor("#2e3545")))
                w.setPen(QPen(QColor("#0d1017"), 1))
                self._add_item(w, Z_RIG + 0.2)
            for wx in (tx + 3, tx + 188):
                gw = QGraphicsEllipseItem(wx, TRACK_Y + 1, 26, 26)
                gw.setBrush(QBrush(QColor("#d4a800")))
                gw.setPen(QPen(QColor("#8a6c00"), 2))
                self._add_item(gw, Z_RIG + 0.3)

    def _draw_cabin(self) -> None:
        cab_y = MAST_BOT_Y + 6
        cabin = QGraphicsRectItem(128, cab_y, 130, 42)
        cabin.setBrush(QBrush(QColor("#d4a800")))
        cabin.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(cabin, Z_RIG + 0.3)

        win = QGraphicsRectItem(138, cab_y + 5, 68, 26)
        win.setBrush(QBrush(QColor("#a8d8c8")))
        win.setPen(QPen(QColor("#4a8a7a"), 1))
        self._add_item(win, Z_RIG + 0.4)

        lbl = pg.TextItem("PV-271", anchor=(0.5, 0), color="#1a1f2a")
        lbl.setPos(193, cab_y + 50)
        self._add_item(lbl, Z_RIG + 0.5)

    def _draw_engine_deck(self) -> None:
        deck_y = MAST_BOT_Y + 8
        deck = QGraphicsRectItem(378, deck_y, 362, 40)
        deck.setBrush(QBrush(QColor("#d4a800")))
        deck.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(deck, Z_RIG + 0.3)

        eng = QGraphicsRectItem(398, deck_y + 5, 160, 26)
        eng.setBrush(QBrush(QColor("#b08c00")))
        eng.setPen(QPen(QColor("#6a5200"), 2))
        self._add_item(eng, Z_RIG + 0.4)

        flt = QGraphicsEllipseItem(570, deck_y + 3, 30, 30)
        flt.setBrush(QBrush(QColor("#c09800")))
        flt.setPen(QPen(QColor("#6a5200"), 2))
        self._add_item(flt, Z_RIG + 0.5)

        for i in range(4):
            rung = QGraphicsLineItem(700, deck_y + 3 + i * 9, 720, deck_y + 3 + i * 9)
            rung.setPen(QPen(QColor("#8a6c00"), 2))
            self._add_item(rung, Z_RIG + 0.4)
        for lx in (700, 720):
            side = QGraphicsLineItem(lx, deck_y + 3, lx, deck_y + 38)
            side.setPen(QPen(QColor("#8a6c00"), 2))
            self._add_item(side, Z_RIG + 0.4)

    # ------------------------------------------------------------------ Каретка (поверх мачты)

    def _draw_carriage(self) -> None:
        cx = MAST_X - CARRIAGE_W / 2

        self.carriage = QGraphicsRectItem(cx, CARRIAGE_TOP_Y, CARRIAGE_W, CARRIAGE_H)
        self.carriage.setBrush(QBrush(QColor("#f0d840")))
        self.carriage.setPen(QPen(QColor("#9f8c2a"), 2))
        self._add_item(self.carriage, Z_CARRIAGE)

        self.motor = QGraphicsPathItem()
        self.motor.setBrush(QBrush(QColor("#c8a800")))
        self.motor.setPen(QPen(QColor("#7a6400"), 2))
        self._add_item(self.motor, Z_CARRIAGE + 0.1)

        self.motor_highlight = QGraphicsPathItem()
        self.motor_highlight.setBrush(QBrush(QColor("#f3e37a")))
        self.motor_highlight.setPen(QPen(QColor("#f3e37a"), 0))
        self._add_item(self.motor_highlight, Z_CARRIAGE + 0.15)

        self._set_rotary_geometry(MAST_X, CARRIAGE_TOP_Y + CARRIAGE_H)

        # Вал — соединяет мотор с трубой, растягивается динамически
        shaft_y = CARRIAGE_TOP_Y + CARRIAGE_H + ROTARY_HEAD_H + ROTARY_NECK_H
        self.motor_shaft = QGraphicsRectItem(MAST_X - 3, shaft_y, 6, 6)
        self.motor_shaft.setBrush(QBrush(QColor("#a08000")))
        self.motor_shaft.setPen(QPen(QColor("#604800"), 1))
        self._add_item(self.motor_shaft, Z_CARRIAGE + 0.2)

    # ------------------------------------------------------------------ Долото (поверх всего)

    def _draw_bit(self) -> None:
        bx = PIPE_X

        self.bit_body = QGraphicsRectItem(bx - 7, SECTION_TOP, 14, BIT_BODY_H)
        self.bit_body.setBrush(QBrush(QColor("#f1c15a")))
        self.bit_body.setPen(QPen(QColor("#a8782c"), 2))
        self._add_item(self.bit_body, Z_BIT)

        self.bit_cone_l = QGraphicsEllipseItem(bx - 9, SECTION_TOP + 7, 6, BIT_CONE_H)
        self.bit_cone_l.setBrush(QBrush(QColor("#ec8a45")))
        self.bit_cone_l.setPen(QPen(QColor("#a5542c"), 1))
        self._add_item(self.bit_cone_l, Z_BIT + 0.1)

        self.bit_cone_m = QGraphicsEllipseItem(bx - 3, SECTION_TOP + BIT_CONE_OFFSET_Y, 6, BIT_CONE_H)
        self.bit_cone_m.setBrush(QBrush(QColor("#ec8a45")))
        self.bit_cone_m.setPen(QPen(QColor("#a5542c"), 1))
        self._add_item(self.bit_cone_m, Z_BIT + 0.1)

        self.bit_cone_r = QGraphicsEllipseItem(bx + 3, SECTION_TOP + 7, 6, BIT_CONE_H)
        self.bit_cone_r.setBrush(QBrush(QColor("#ec8a45")))
        self.bit_cone_r.setPen(QPen(QColor("#a5542c"), 1))
        self._add_item(self.bit_cone_r, Z_BIT + 0.1)

    def _rotary_paths(self, center_x: float, top_y: float) -> tuple[QPainterPath, QPainterPath]:
        left = center_x - ROTARY_W / 2
        right = center_x + ROTARY_W / 2
        notch_w = 10.0
        notch_d = 5.0
        shoulder_y = top_y + ROTARY_HEAD_H
        neck_left = center_x - ROTARY_NECK_W / 2
        neck_right = center_x + ROTARY_NECK_W / 2
        neck_bottom = shoulder_y + ROTARY_NECK_H

        body = QPainterPath()
        body.moveTo(left, shoulder_y)
        body.lineTo(left, top_y + notch_d)
        body.lineTo(left + notch_w * 0.6, top_y + notch_d)
        body.lineTo(left + notch_w * 0.6, top_y)
        body.lineTo(left + notch_w * 1.4, top_y)
        body.lineTo(left + notch_w * 1.4, top_y + notch_d)
        body.lineTo(center_x - notch_w * 0.5, top_y + notch_d)
        body.lineTo(center_x - notch_w * 0.5, top_y)
        body.lineTo(center_x + notch_w * 0.5, top_y)
        body.lineTo(center_x + notch_w * 0.5, top_y + notch_d)
        body.lineTo(right - notch_w * 1.4, top_y + notch_d)
        body.lineTo(right - notch_w * 1.4, top_y)
        body.lineTo(right - notch_w * 0.6, top_y)
        body.lineTo(right - notch_w * 0.6, top_y + notch_d)
        body.lineTo(right, top_y + notch_d)
        body.lineTo(right, shoulder_y)
        body.lineTo(neck_right, shoulder_y)
        body.lineTo(neck_right, neck_bottom)
        body.lineTo(neck_left, neck_bottom)
        body.lineTo(neck_left, shoulder_y)
        body.closeSubpath()

        hl = QPainterPath()
        hl.moveTo(left + 5, shoulder_y - 2)
        hl.lineTo(left + 5, top_y + notch_d + 2)
        hl.lineTo(left + notch_w * 0.9, top_y + notch_d + 2)
        hl.lineTo(left + notch_w * 0.9, top_y + 2)
        hl.lineTo(left + notch_w * 1.15, top_y + 2)
        hl.lineTo(left + notch_w * 1.15, top_y + notch_d + 2)
        hl.lineTo(center_x - notch_w * 0.75, top_y + notch_d + 2)
        hl.lineTo(center_x - notch_w * 0.75, shoulder_y - 2)
        hl.lineTo(center_x - 3, shoulder_y - 2)
        hl.lineTo(center_x - 3, neck_bottom - 3)
        hl.lineTo(center_x - ROTARY_NECK_W / 4, neck_bottom - 3)
        hl.lineTo(center_x - ROTARY_NECK_W / 4, shoulder_y + 3)
        hl.lineTo(left + 5, shoulder_y + 3)
        hl.closeSubpath()
        return body, hl

    def _set_rotary_geometry(self, center_x: float, top_y: float) -> None:
        body_path, hl_path = self._rotary_paths(center_x, top_y)
        self.motor.setPath(body_path)
        self.motor_highlight.setPath(hl_path)

    # ------------------------------------------------------------------ Индикатор глубины

    def _draw_depth_indicator(self) -> None:
        self.depth_line = QGraphicsLineItem(
            PIPE_X + 18, SECTION_TOP + BIT_BODY_H / 2, PIPE_X + 94, SECTION_TOP + BIT_BODY_H / 2
        )
        depth_pen = QPen(QColor("#00ff26"), 2)
        depth_pen.setStyle(Qt.PenStyle.DashLine)
        self.depth_line.setPen(depth_pen)
        self._add_item(self.depth_line, Z_OVERLAY)

        self.depth_text = pg.TextItem("0 m", anchor=(0, 0.5), color="#00ff26")
        self.depth_text.setFont(QFont("Segoe UI", 8))
        self.depth_text.setPos(PIPE_X + 100, SECTION_TOP + BIT_BODY_H / 2)
        self._add_item(self.depth_text, Z_OVERLAY)

    # ------------------------------------------------------------------ Лейблы

    def _draw_labels(self) -> None:
        info_x = SCENE_W - 18
        self.region_text = pg.TextItem(f"Region: {self.region}", anchor=(1, 0), color="#1a2030")
        self.region_text.setPos(info_x, 24)
        self._add_item(self.region_text, Z_OVERLAY)

        self.axial_text = pg.TextItem("Axial pressure: 0 kN", anchor=(1, 0), color="#1a2030")
        self.axial_text.setPos(info_x, 48)
        self._add_item(self.axial_text, Z_OVERLAY)

        self.rot_text = pg.TextItem("Rotational pressure: 0 kN·m", anchor=(0, 0), color="#1a2030")
        self.rot_text.setAnchor((1, 0))
        self.rot_text.setPos(info_x, 72)
        self._add_item(self.rot_text, Z_OVERLAY)

    # ------------------------------------------------------------------ Таймер

    def _start_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    # ------------------------------------------------------------------ Кинематика

    def _bit_y(self) -> float:
        return SECTION_TOP + SECTION_H * (self.depth_m / self.target_depth_m)

    def _bit_tip_y(self) -> float:
        return self._bit_y() + BIT_CONE_OFFSET_Y + BIT_CONE_H

    def _car_y(self) -> float:
        t = (self.depth_m / self.target_depth_m)
        return CARRIAGE_TOP_Y + (CARRIAGE_BOT_Y - CARRIAGE_TOP_Y) * t

    def _on_tick(self) -> None:
        self._tick += 0.06
        self.depth_m = min(self.depth_m + DEPTH_STEP_M, self.target_depth_m)

        if self._bit_tip_y() >= SECTION_BOTTOM:
            self.depth_m = 0.0
            self.target_depth_m = random.uniform(MIN_DEPTH_M, MAX_DEPTH_M)
            self.region, self.layers = generate_rock_layers()
            self.region_text.setText(f"Region: {self.region}")
            self._clear_geology()
            self._draw_geology()
            self._reset()
            return

        self._sync()

        axial = 430 + 34 * math.sin(self._tick)
        rot = 118 + 14 * math.cos(self._tick * 0.9)
        self.axial_text.setText(f"Axial pressure: {axial:0.0f} kN")
        self.rot_text.setText(f"Rotational pressure: {rot:0.0f} kN·m")

    def _sync(self) -> None:
        """Синхронизирует все подвижные элементы с текущей depth_m."""
        bx = PIPE_X
        bit_y = self._bit_y()
        car_y = self._car_y()

        # 1. Каретка + мотор (жёстко как один блок)
        self.carriage.setRect(bx - CARRIAGE_W / 2, car_y, CARRIAGE_W, CARRIAGE_H)
        motor_y = car_y + CARRIAGE_H
        self._set_rotary_geometry(bx, motor_y)

        # 2. Вал мотора: от низа мотора до верха трубы в породе
        shaft_top = motor_y + ROTARY_HEAD_H + ROTARY_NECK_H
        # Труба начинается там, где кончается вал — то есть вал это и есть верхняя часть трубы
        # shaft растягивается от мотора до section_top (над породой труба видна как вал)
        shaft_h = max(SECTION_TOP - shaft_top, 1)
        self.motor_shaft.setRect(bx - 3, shaft_top, 6, shaft_h)

        # 3. Труба в породе: от section_top до bit_y
        pipe_top = SECTION_TOP
        pipe_h = max(bit_y - pipe_top, 0)
        self.drill_pipe.setRect(bx - PIPE_W / 2, pipe_top, PIPE_W, pipe_h)
        self.pipe_shine.setRect(bx - PIPE_W / 2, pipe_top, 1.5, pipe_h)

        # 4. Скважина: от section_top до bit_y
        self.drilled_hole.setRect(bx - BOREHOLE_W / 2, SECTION_TOP, BOREHOLE_W, pipe_h)

        # 5. Долото строго на конце трубы
        self._place_bit(bit_y)

        # 6. Глубина
        indicator_y = bit_y + BIT_BODY_H * 0.5
        self.depth_line.setLine(bx + 18, indicator_y, bx + 94, indicator_y)
        self.depth_text.setText(f"{self.depth_m:0.1f} m")
        self.depth_text.setPos(bx + 100, min(indicator_y, SECTION_BOTTOM - 12))

    def _reset(self) -> None:
        """Возвращает все динамические элементы в начальное положение."""
        bx = PIPE_X

        self.carriage.setRect(bx - CARRIAGE_W / 2, CARRIAGE_TOP_Y, CARRIAGE_W, CARRIAGE_H)
        self._set_rotary_geometry(bx, CARRIAGE_TOP_Y + CARRIAGE_H)
        self.motor_shaft.setRect(bx - 3, CARRIAGE_TOP_Y + CARRIAGE_H + ROTARY_HEAD_H + ROTARY_NECK_H, 6, max(SECTION_TOP - (CARRIAGE_TOP_Y + CARRIAGE_H + ROTARY_HEAD_H + ROTARY_NECK_H), 1))

        self.drill_pipe.setRect(bx - PIPE_W / 2, SECTION_TOP, PIPE_W, 0)
        self.pipe_shine.setRect(bx - PIPE_W / 2, SECTION_TOP, 1.5, 0)
        self.drilled_hole.setRect(bx - BOREHOLE_W / 2, SECTION_TOP, BOREHOLE_W, 0)

        self._place_bit(SECTION_TOP)

        self.depth_line.setLine(PIPE_X + 18, SECTION_TOP + BIT_BODY_H / 2, PIPE_X + 94, SECTION_TOP + BIT_BODY_H / 2)
        self.depth_text.setText("0.0 m")
        self.depth_text.setPos(PIPE_X + 100, SECTION_TOP + BIT_BODY_H / 2)

    def _place_bit(self, bit_y: float) -> None:
        bx = PIPE_X
        self.bit_body.setRect(bx - 7, bit_y, 14, BIT_BODY_H)
        self.bit_cone_l.setRect(bx - 9, bit_y + 7, 6, BIT_CONE_H)
        self.bit_cone_m.setRect(bx - 3, bit_y + BIT_CONE_OFFSET_Y, 6, BIT_CONE_H)
        self.bit_cone_r.setRect(bx + 3, bit_y + 7, 6, BIT_CONE_H)
