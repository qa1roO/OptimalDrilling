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
SCENE_H = 1220.0

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

SECTION_LEFT = 0.0
SECTION_TOP = TRACK_Y + TRACK_H
SECTION_RIGHT = SCENE_W
SECTION_BOTTOM = SCENE_H - 20.0
SECTION_W = SECTION_RIGHT - SECTION_LEFT
SECTION_H = SECTION_BOTTOM - SECTION_TOP

PIPE_X = MAST_X
PIPE_W = 5.0
BOREHOLE_W = 13.0
BIT_BODY_H = 8.0
BIT_CONE_H = 8.0
BIT_CONE_OFFSET_Y = 9.0
PIPE_SPIN_MARKS = 14
PIPE_SPIN_SPACING = 24.0
ROT_SPEED_BASE_RPM = 92.0
ROT_SPEED_SWING_RPM = 14.0

# Реалистичный рабочий диапазон одной буровой штанги.
MIN_DEPTH_M = 20.0
MAX_DEPTH_M = 40.0
DEPTH_STEP_M = 0.06
TIMER_INTERVAL_MS = 40

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
        self.rot_speed_rpm = ROT_SPEED_BASE_RPM
        self.spin_angle_rad = 0.0
        self._pending_geology_reset = False
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
            rect = QGraphicsRectItem(SECTION_LEFT, y, SECTION_W, h)
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

        self.pipe_rot_shadow = QGraphicsRectItem(bx - PIPE_W / 2, MAST_BOT_Y, 1.5, 0)
        self.pipe_rot_shadow.setBrush(QBrush(QColor(88, 97, 110, 110)))
        self.pipe_rot_shadow.setPen(QPen(QColor(88, 97, 110, 0), 0))
        self._add_item(self.pipe_rot_shadow, Z_PIPE + 1)

        self.pipe_spin_marks: list[QGraphicsLineItem] = []
        spin_pen = QPen(QColor("#eef4fb"), 1.2)
        for _ in range(PIPE_SPIN_MARKS):
            mark = QGraphicsLineItem()
            mark.setPen(spin_pen)
            self._add_item(mark, Z_PIPE + 2)
            self.pipe_spin_marks.append(mark)

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
        self.pulley = QGraphicsEllipseItem(MAST_X - 8, MAST_TOP_Y - 7, 16, 16)
        self.pulley.setBrush(QBrush(QColor("#2a3040")))
        self.pulley.setPen(QPen(QColor("#1a1f2e"), 2))
        self._add_item(self.pulley, Z_RIG + 0.3)

        self.pulley_spokes: list[QGraphicsLineItem] = []
        spoke_pen = QPen(QColor("#b9c4d1"), 1.5)
        for _ in range(4):
            spoke = QGraphicsLineItem()
            spoke.setPen(spoke_pen)
            self._add_item(spoke, Z_RIG + 0.4)
            self.pulley_spokes.append(spoke)
        self._update_pulley_rotation()

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
        body = QGraphicsRectItem(115, CHASSIS_Y, 625, CHASSIS_H)
        body.setBrush(QBrush(QColor("#495468")))
        body.setPen(QPen(QColor("#252e3d"), 2))
        self._add_item(body, Z_RIG)

        mast_pass = QGraphicsRectItem(130, CHASSIS_Y + 55, 230, 30)
        mast_pass.setBrush(QBrush(QColor("#262d39")))
        mast_pass.setPen(QPen(QColor("#161b23"), 2))
        self._add_item(mast_pass, Z_RIG + 0.15)

        track_x = 388
        track_w = 218
        track_y = TRACK_Y - 5
        sprocket_d = 38
        idler_d = 24
        frame_h = 34
        belt_h = 10

        track_body = QGraphicsRectItem(track_x + 5, track_y + 20, track_w - 28, frame_h - 29)
        track_body.setBrush(QBrush(QColor("#e0b400")))
        track_body.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(track_body, Z_RIG + 0.15)

        upper_belt = QGraphicsRectItem(track_x + 8, track_y + 2, track_w - 34, belt_h)
        upper_belt.setBrush(QBrush(QColor("#222933")))
        upper_belt.setPen(QPen(QColor("#141920"), 1))
        self._add_item(upper_belt, Z_RIG + 0.20)

        lower_belt = QGraphicsRectItem(track_x + 12, track_y + 25, track_w - 36, belt_h)
        lower_belt.setBrush(QBrush(QColor("#222933")))
        lower_belt.setPen(QPen(QColor("#141920"), 1))
        self._add_item(lower_belt, Z_RIG + 0.20)

        left_belt = QGraphicsEllipseItem(track_x - 2, track_y + 10, idler_d, idler_d)
        left_belt.setBrush(QBrush(QColor("#222933")))
        left_belt.setPen(QPen(QColor("#141920"), 1))
        self._add_item(left_belt, Z_RIG + 0.20)

        right_belt = QGraphicsEllipseItem(track_x + track_w - sprocket_d, track_y + 1, sprocket_d, sprocket_d)
        right_belt.setBrush(QBrush(QColor("#222933")))
        right_belt.setPen(QPen(QColor("#141920"), 1))
        self._add_item(right_belt, Z_RIG + 0.20)

        left_wheel = QGraphicsEllipseItem(track_x, track_y + 12, idler_d - 4, idler_d - 4)
        left_wheel.setBrush(QBrush(QColor("#d9ad00")))
        left_wheel.setPen(QPen(QColor("#846400"), 2))
        self._add_item(left_wheel, Z_RIG + 0.24)

        for tx in range(track_x + 30, track_x + track_w - 56, 20):
            roller = QGraphicsEllipseItem(tx, track_y + 20, 10, 10)
            roller.setBrush(QBrush(QColor("#667387")))
            roller.setPen(QPen(QColor("#2c3442"), 1))
            self._add_item(roller, Z_RIG + 0.25)

        carrier = QGraphicsRectItem(track_x + 38, track_y + 13, 122, 4)
        carrier.setBrush(QBrush(QColor("#586476")))
        carrier.setPen(QPen(QColor("#313a48"), 1))
        self._add_item(carrier, Z_RIG + 0.19)

        for tx in range(track_x + 10, track_x + track_w - 28, 18):
            top_tooth = QGraphicsRectItem(tx, track_y - 2, 12, 3)
            top_tooth.setBrush(QBrush(QColor("#141920")))
            top_tooth.setPen(QPen(QColor("#141920"), 1))
            self._add_item(top_tooth, Z_RIG + 0.23)

        for tx in range(track_x + 32, track_x + track_w - 50, 18):
            shoe = QGraphicsRectItem(tx, track_y + 30, 12, 4)
            shoe.setBrush(QBrush(QColor("#37404e")))
            shoe.setPen(QPen(QColor("#171c24"), 1))
            self._add_item(shoe, Z_RIG + 0.23)

        sprocket = QGraphicsEllipseItem(track_x + track_w - sprocket_d + 4, track_y + 5, sprocket_d - 8, sprocket_d - 8)
        sprocket.setBrush(QBrush(QColor("#e0b400")))
        sprocket.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(sprocket, Z_RIG + 0.26)

        hub = QGraphicsEllipseItem(track_x + track_w - 24, track_y + 15, 10, 10)
        hub.setBrush(QBrush(QColor("#505b6b")))
        hub.setPen(QPen(QColor("#283243"), 1))
        self._add_item(hub, Z_RIG + 0.27)

        leg_top_y = CHASSIS_Y + CHASSIS_H
        self._draw_stabilizer_leg(340, leg_top_y, SECTION_TOP)
        self._draw_stabilizer_leg(720, leg_top_y, SECTION_TOP)

    def _draw_stabilizer_leg(self, center_x: float, top_y: float, ground_y: float) -> None:
        foot_h = 8.0
        leg_h = max(ground_y - top_y - foot_h, 16.0)

        leg = QGraphicsRectItem(center_x - 7, top_y, 14, leg_h)
        leg.setBrush(QBrush(QColor("#e0b400")))
        leg.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(leg, Z_RIG + 0.25)

        foot = QGraphicsPathItem()
        fp = QPainterPath()
        foot_y = top_y + leg_h
        fp.moveTo(center_x - 22, foot_y)
        fp.lineTo(center_x + 22, foot_y)
        fp.lineTo(center_x + 14, foot_y + foot_h)
        fp.lineTo(center_x - 14, foot_y + foot_h)
        fp.closeSubpath()
        foot.setPath(fp)
        foot.setBrush(QBrush(QColor("#75664a")))
        foot.setPen(QPen(QColor("#4a4030"), 2))
        self._add_item(foot, Z_RIG + 0.24)

        for dx in (-16, -6, 6, 16):
            rib = QGraphicsLineItem(center_x, foot_y, center_x + dx, foot_y + foot_h)
            rib.setPen(QPen(QColor("#4a4030"), 1))
            self._add_item(rib, Z_RIG + 0.26)

    def _draw_cabin(self) -> None:
        cab_y = MAST_BOT_Y - 30

        roof = QGraphicsRectItem(90, cab_y + 0, 150, 12)
        roof.setBrush(QBrush(QColor("#d9ca8f")))
        roof.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(roof, Z_RIG + 0.31)

        cabin = QGraphicsPathItem()
        path = QPainterPath()
        path.moveTo(114, cab_y + 86)
        path.lineTo(100, cab_y + 14)
        path.lineTo(128, cab_y + 14)
        path.lineTo(236, cab_y + 14)
        path.lineTo(236, cab_y + 86)
        path.closeSubpath()
        cabin.setPath(path)
        cabin.setBrush(QBrush(QColor("#d8b347")))
        cabin.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(cabin, Z_RIG + 0.3)

        front_win = QGraphicsPathItem()
        front_path = QPainterPath()
        front_path.moveTo(124, cab_y + 58)
        front_path.lineTo(124, cab_y + 28)
        front_path.lineTo(137, cab_y + 20)
        front_path.lineTo(157, cab_y + 20)
        front_path.lineTo(155, cab_y + 58)
        front_path.closeSubpath()
        front_win.setPath(front_path)
        front_win.setBrush(QBrush(QColor("#d9e3e7")))
        front_win.setPen(QPen(QColor("#51596a"), 2))
        self._add_item(front_win, Z_RIG + 0.4)

        center_win = QGraphicsRectItem(158, cab_y + 20, 24, 38)
        center_win.setBrush(QBrush(QColor("#d9e3e7")))
        center_win.setPen(QPen(QColor("#51596a"), 2))
        self._add_item(center_win, Z_RIG + 0.4)

        side_win = QGraphicsRectItem(186, cab_y + 20, 40, 38)
        side_win.setBrush(QBrush(QColor("#d9e3e7")))
        side_win.setPen(QPen(QColor("#51596a"), 2))
        self._add_item(side_win, Z_RIG + 0.4)

        lower_panel = QGraphicsRectItem(120, cab_y + 60, 112, 22)
        lower_panel.setBrush(QBrush(QColor("#d1aa40")))
        lower_panel.setPen(QPen(QColor("#8a6c00"), 1))
        self._add_item(lower_panel, Z_RIG + 0.35)

        lbl = pg.TextItem("PV-271", anchor=(0.5, 0), color="#4a1d1f")
        lbl.setPos(176, cab_y + 52)
        self._add_item(lbl, Z_RIG + 0.5)

        cab_ladder = QGraphicsLineItem(113, cab_y + 70, 113, SECTION_TOP - 6)
        cab_ladder.setPen(QPen(QColor("#d4b100"), 2))
        self._add_item(cab_ladder, Z_RIG + 0.34)

    def _draw_engine_deck(self) -> None:
        rail_base_y = MAST_BOT_Y + 48
        rail_color = QColor("#c29300")
        rail_dark = QColor("#7b5d00")

        for cx in (340, 720):
            cap = QGraphicsRectItem(cx - 8, rail_base_y - 12, 16, 12)
            cap.setBrush(QBrush(QColor("#e0b400")))
            cap.setPen(QPen(QColor("#8a6c00"), 2))
            self._add_item(cap, Z_RIG + 0.42)

        rail_posts = (232, 340, 500, 610, 720)
        rail_top_y = rail_base_y - 22
        rail_mid_y = rail_base_y - 10
        for px in rail_posts:
            post = QGraphicsLineItem(px, rail_top_y, px, rail_base_y)
            post.setPen(QPen(rail_color, 2))
            self._add_item(post, Z_RIG + 0.43)

        top_rail = QGraphicsLineItem(232, rail_top_y, 720, rail_top_y)
        top_rail.setPen(QPen(rail_color, 2))
        self._add_item(top_rail, Z_RIG + 0.44)

        mid_rail = QGraphicsLineItem(232, rail_mid_y, 720, rail_mid_y)
        mid_rail.setPen(QPen(rail_color, 2))
        self._add_item(mid_rail, Z_RIG + 0.44)

        stair_top_x = 426
        stair_top_y = rail_base_y - 2
        stair_bottom_x = 464
        stair_bottom_y = rail_base_y + 56

        left_stringer = QGraphicsLineItem(stair_top_x, stair_top_y, stair_bottom_x, stair_bottom_y)
        left_stringer.setPen(QPen(rail_color, 3))
        self._add_item(left_stringer, Z_RIG + 0.45)

        right_stringer = QGraphicsLineItem(stair_top_x + 14, stair_top_y, stair_bottom_x + 14, stair_bottom_y)
        right_stringer.setPen(QPen(rail_color, 3))
        self._add_item(right_stringer, Z_RIG + 0.45)

        for i in range(6):
            t = (i + 0.6) / 6.6
            lx = stair_top_x + (stair_bottom_x - stair_top_x) * t
            ly = stair_top_y + (stair_bottom_y - stair_top_y) * t
            rx = stair_top_x + 14 + (stair_bottom_x - stair_top_x) * t
            ry = stair_top_y + (stair_bottom_y - stair_top_y) * t
            step = QGraphicsLineItem(lx, ly, rx, ry)
            step.setPen(QPen(QColor("#e7c640"), 2))
            self._add_item(step, Z_RIG + 0.46)

        stair_post_top = QGraphicsLineItem(stair_top_x + 14, rail_top_y, stair_top_x + 14, stair_top_y)
        stair_post_top.setPen(QPen(rail_color, 2))
        self._add_item(stair_post_top, Z_RIG + 0.45)

        deck_y = rail_base_y - 18
        deck_h = 14
        deck = QGraphicsRectItem(474, deck_y, 214, deck_h)
        deck.setBrush(QBrush(QColor("#d1a200")))
        deck.setPen(QPen(QColor("#896700"), 2))
        self._add_item(deck, Z_RIG + 0.41)

        module_y = deck_y - 18

        tank = QGraphicsRectItem(500, module_y - 12, 70, 16)
        tank.setBrush(QBrush(QColor("#d9b100")))
        tank.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(tank, Z_RIG + 0.46)

        tank_head = QGraphicsEllipseItem(494, module_y - 12, 18, 16)
        tank_head.setBrush(QBrush(QColor("#d9b100")))
        tank_head.setPen(QPen(QColor("#8a6c00"), 2))
        self._add_item(tank_head, Z_RIG + 0.46)

        for tx in (524, 548):
            band = QGraphicsLineItem(tx, module_y - 12, tx, module_y + 4)
            band.setPen(QPen(QColor("#8a6c00"), 2))
            self._add_item(band, Z_RIG + 0.47)

        for px in (516, 548):
            post = QGraphicsLineItem(px, module_y + 4, px, deck_y + deck_h)
            post.setPen(QPen(QColor("#8a6c00"), 2))
            self._add_item(post, Z_RIG + 0.45)

        elbow = QGraphicsPathItem()
        ep = QPainterPath()
        ep.moveTo(570, module_y - 4)
        ep.cubicTo(582, module_y - 4, 584, module_y + 4, 586, module_y + 9)
        ep.cubicTo(588, module_y + 13, 594, module_y + 13, 597, module_y + 11)
        elbow.setPath(ep)
        elbow.setPen(QPen(QColor("#3d434c"), 3))
        self._add_item(elbow, Z_RIG + 0.47)

        for cx in (520, 542, 564):
            unit = QGraphicsEllipseItem(cx, deck_y + 4, 18, 18)
            unit.setBrush(QBrush(QColor("#4e5766")))
            unit.setPen(QPen(QColor("#2d3440"), 2))
            self._add_item(unit, Z_RIG + 0.46)

        unit_base = QGraphicsRectItem(512, deck_y + deck_h - 2, 74, 8)
        unit_base.setBrush(QBrush(QColor("#3a414e")))
        unit_base.setPen(QPen(QColor("#242a34"), 1))
        self._add_item(unit_base, Z_RIG + 0.45)

        generator = QGraphicsRectItem(606, deck_y - 14, 48, 24)
        generator.setBrush(QBrush(QColor("#cf9f00")))
        generator.setPen(QPen(QColor("#876600"), 2))
        self._add_item(generator, Z_RIG + 0.46)

        gen_top = QGraphicsRectItem(612, deck_y - 22, 36, 10)
        gen_top.setBrush(QBrush(QColor("#d8ad14")))
        gen_top.setPen(QPen(QColor("#876600"), 1))
        self._add_item(gen_top, Z_RIG + 0.47)

        top_bar = QGraphicsLineItem(602, deck_y - 20, 660, deck_y - 20)
        top_bar.setPen(QPen(rail_color, 2))
        self._add_item(top_bar, Z_RIG + 0.46)
        for px in (606, 620, 634, 648):
            guard = QGraphicsLineItem(px, deck_y - 20, px, deck_y)
            guard.setPen(QPen(rail_dark, 1))
            self._add_item(guard, Z_RIG + 0.46)

        hose_1 = QGraphicsPathItem()
        hp1 = QPainterPath()
        hp1.moveTo(584, deck_y + 14)
        hp1.cubicTo(596, deck_y + 16, 612, deck_y - 4, 626, deck_y - 2)
        hose_1.setPath(hp1)
        hose_1.setPen(QPen(QColor("#353b45"), 3))
        self._add_item(hose_1, Z_RIG + 0.47)

        hose_2 = QGraphicsPathItem()
        hp2 = QPainterPath()
        hp2.moveTo(590, deck_y + 20)
        hp2.cubicTo(604, deck_y + 26, 620, deck_y + 10, 644, deck_y + 8)
        hose_2.setPath(hp2)
        hose_2.setPen(QPen(QColor("#353b45"), 3))
        self._add_item(hose_2, Z_RIG + 0.47)

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

    def _update_pulley_rotation(self) -> None:
        cx = MAST_X
        cy = MAST_TOP_Y + 1
        radius = 6.0
        angle = self.spin_angle_rad
        for idx, spoke in enumerate(self.pulley_spokes):
            a = angle + idx * (math.pi / 2)
            dx = math.cos(a) * radius
            dy = math.sin(a) * radius
            spoke.setLine(cx - dx, cy - dy, cx + dx, cy + dy)

    def _update_pipe_rotation(self, pipe_top: float, pipe_h: float) -> None:
        for idx, mark in enumerate(self.pipe_spin_marks):
            mark.setLine(0, 0, 0, 0)

        if pipe_h <= 0:
            self.pipe_shine.setRect(PIPE_X - PIPE_W / 2, pipe_top, 0, 0)
            self.pipe_rot_shadow.setRect(PIPE_X - PIPE_W / 2, pipe_top, 0, 0)
            return

        glow_phase = 0.5 + 0.5 * math.sin(self.spin_angle_rad)
        glow_w = 1.2
        glow_x = PIPE_X - PIPE_W / 2 + 0.4 + glow_phase * max(PIPE_W - glow_w - 0.8, 0)
        shadow_phase = 0.5 + 0.5 * math.sin(self.spin_angle_rad + math.pi)
        shadow_w = 1.6
        shadow_x = PIPE_X - PIPE_W / 2 + shadow_phase * max(PIPE_W - shadow_w, 0)

        self.pipe_shine.setRect(glow_x, pipe_top, glow_w, pipe_h)
        self.pipe_rot_shadow.setRect(shadow_x, pipe_top, shadow_w, pipe_h)

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
        self.rot_text.setText("Rotation speed: 0 rpm")
        self.rot_text.setPos(info_x, 72)
        self._add_item(self.rot_text, Z_OVERLAY)

    # ------------------------------------------------------------------ Таймер

    def _start_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(TIMER_INTERVAL_MS)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    # ------------------------------------------------------------------ Кинематика

    def _bit_y(self) -> float:
        return SECTION_TOP + SECTION_H * min(self.depth_m / MAX_DEPTH_M, 1.0)

    def _bit_tip_y(self) -> float:
        return self._bit_y() + BIT_CONE_OFFSET_Y + BIT_CONE_H

    def _target_bottom_y(self) -> float:
        return SECTION_TOP + SECTION_H * (self.target_depth_m / MAX_DEPTH_M)

    def _car_y(self) -> float:
        t = min(self.depth_m / MAX_DEPTH_M, 1.0)
        return CARRIAGE_TOP_Y + (CARRIAGE_BOT_Y - CARRIAGE_TOP_Y) * t

    def _on_tick(self) -> None:
        if self._pending_geology_reset:
            self._pending_geology_reset = False
            self.depth_m = 0.0
            self.target_depth_m = random.uniform(MIN_DEPTH_M, MAX_DEPTH_M)
            self.region, self.layers = generate_rock_layers()
            self.region_text.setText(f"Region: {self.region}")
            self._clear_geology()
            self._draw_geology()
            self._reset()
            return

        self._tick += 0.06
        self.rot_speed_rpm = ROT_SPEED_BASE_RPM + ROT_SPEED_SWING_RPM * math.cos(self._tick * 1.2)
        self.spin_angle_rad += 2 * math.pi * (self.rot_speed_rpm / 60.0) * (TIMER_INTERVAL_MS / 1000.0)
        self.depth_m = min(self.depth_m + DEPTH_STEP_M, self.target_depth_m)

        if self._bit_tip_y() >= self._target_bottom_y():
            self._pending_geology_reset = True

        self._sync()

        axial = 430 + 34 * math.sin(self._tick)
        rot = self.rot_speed_rpm
        self.axial_text.setText(f"Axial pressure: {axial:0.0f} kN")
        self.rot_text.setText(f"Rotational pressure: {rot:0.0f} kN·m")
        self.rot_text.setText(f"Rotation speed: {self.rot_speed_rpm:0.0f} rpm")

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
        self._update_pulley_rotation()

        # 3. Труба в породе: от section_top до bit_y
        pipe_top = SECTION_TOP
        pipe_h = max(bit_y - pipe_top, 0)
        self.drill_pipe.setRect(bx - PIPE_W / 2, pipe_top, PIPE_W, pipe_h)
        self.pipe_shine.setRect(bx - PIPE_W / 2, pipe_top, 1.5, pipe_h)
        self._update_pipe_rotation(pipe_top, pipe_h)

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
        self._update_pulley_rotation()

        self.drill_pipe.setRect(bx - PIPE_W / 2, SECTION_TOP, PIPE_W, 0)
        self.pipe_shine.setRect(bx - PIPE_W / 2, SECTION_TOP, 1.5, 0)
        self.drilled_hole.setRect(bx - BOREHOLE_W / 2, SECTION_TOP, BOREHOLE_W, 0)
        self._update_pipe_rotation(SECTION_TOP, 0)

        self._place_bit(SECTION_TOP)

        self.depth_line.setLine(PIPE_X + 18, SECTION_TOP + BIT_BODY_H / 2, PIPE_X + 94, SECTION_TOP + BIT_BODY_H / 2)
        self.depth_text.setText("0.0 m")
        self.depth_text.setPos(PIPE_X + 100, SECTION_TOP + BIT_BODY_H / 2)

    def _place_bit(self, bit_y: float) -> None:
        bx = PIPE_X
        roll = math.sin(self.spin_angle_rad) * 1.2
        self.bit_body.setRect(bx - 7, bit_y, 14, BIT_BODY_H)
        self.bit_cone_l.setRect(bx - 9 - roll, bit_y + 7, 6, BIT_CONE_H)
        self.bit_cone_m.setRect(bx - 3, bit_y + BIT_CONE_OFFSET_Y + abs(roll) * 0.4, 6, BIT_CONE_H)
        self.bit_cone_r.setRect(bx + 3 + roll, bit_y + 7, 6, BIT_CONE_H)
