from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.data import (
    ClusterProfile,
    DrillingPoint,
    load_cluster_profiles,
)

try:
    import numpy as np
    import pyqtgraph.opengl as gl

    OPENGL_READY = True
except Exception:  # pragma: no cover
    OPENGL_READY = False


class Performance3DWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(760, 360)
        self.axis_size_x = 360.0
        self.axis_size_y = 230.0
        self.axis_size_z = 130.0
        self.speed_scale = 25.0
        self.time_scale = 1.0
        self.rotation_scale = 1.35
        self._plot_items: list = []
        self._live_points: list[DrillingPoint] = []
        self._live_line = None
        self._live_scatter = None
        self._mode = "live"
        self._live_segment_index = 0
        self.axis_description = "OX=time, OY=rotation x1.35, OZ=speed"
        self._cluster_profiles: dict[int, ClusterProfile] = {}
        self._active_profile: ClusterProfile | None = None
        self._active_cluster_id: int | None = None
        self._segment_start_time_s: float | None = None
        self._graph_start_time_s: float | None = None
        self._transition_duration_s = 10.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.status_label = QLabel(self)
        self.status_label.setStyleSheet(
            "background: #121820; color: #c8d2dc; font-size: 12px; padding: 6px 8px;"
            "border-radius: 6px;"
        )
        layout.addWidget(self.status_label)

        if OPENGL_READY:
            self._axis_label_items: list = []
            self.view = self._build_gl_view()
            layout.addWidget(self.view, 1)
            self.show_default_mode()
        else:
            warning = QLabel(
                "OpenGL backend is unavailable.\n"
                "Install dependencies from requirements.txt.",
                self,
            )
            warning.setStyleSheet("padding: 16px; color: #8a1f1f;")
            layout.addWidget(warning, 1)

    def show_default_mode(self) -> None:
        self._mode = "live"
        self.reset_live_mode()

    def reset_live_mode(self) -> None:
        if not OPENGL_READY:
            return

        self._mode = "live"
        self._live_segment_index = 0
        self._live_points.clear()
        self._active_profile = None
        self._active_cluster_id = None
        self._segment_start_time_s = None
        self._graph_start_time_s = None
        self._live_line = None
        self._live_scatter = None
        self._clear_plot_items()
        self.status_label.setText(
            f"3D graph: {self.axis_description}. Waiting until the bit touches rock."
        )

    def on_drilling_cycle_started(self) -> None:
        if self._mode == "live":
            self.reset_live_mode()

    def append_drilling_point(
        self,
        time_s: float,
        rotation: float,
        speed: float,
        cluster_id: int | None = None,
    ) -> None:
        if not OPENGL_READY or self._mode != "live":
            return

        if cluster_id is None:
            self._live_points.append(
                DrillingPoint(time_s=time_s, rotation=rotation, speed=speed, source="model")
            )
        else:
            if self._active_cluster_id != cluster_id:
                self._start_layer_segment(cluster_id=cluster_id, time_s=time_s)
            self._live_points.append(self._interpolated_profile_point(time_s))

        if self._live_line is None or self._live_scatter is None:
            self._create_live_items()

        data = self._points_to_array(self._live_points)
        self._live_line.setData(pos=data)
        self._live_scatter.setData(pos=data)
        current = self._live_points[-1]
        self.status_label.setText(
            f"Live drilling: {self.axis_description}. "
            f"cluster={current.cluster_id}, t={time_s:0.1f}s, "
            f"rotation={current.rotation:0.0f} rpm, speed={current.speed:0.4f} m/s."
        )

    def _build_gl_view(self) -> QWidget:
        view = gl.GLViewWidget()
        view.opts["distance"] = 380
        view.opts["elevation"] = 24
        view.opts["azimuth"] = -42
        view.setBackgroundColor((12, 16, 22, 255))

        axis = gl.GLAxisItem()
        axis.setSize(x=self.axis_size_x, y=self.axis_size_y, z=self.axis_size_z)
        view.addItem(axis)

        grid_xy = gl.GLGridItem()
        grid_xy.setColor((80, 92, 108, 85))
        grid_xy.setSize(x=380, y=240)
        grid_xy.setSpacing(x=50, y=30)
        grid_xy.translate(190, 120, 0)
        view.addItem(grid_xy)

        grid_yz = gl.GLGridItem()
        grid_yz.setColor((80, 92, 108, 65))
        grid_yz.setSize(x=240, y=130)
        grid_yz.setSpacing(x=30, y=20)
        grid_yz.rotate(90, 0, 1, 0)
        grid_yz.translate(0, 120, 65)
        view.addItem(grid_yz)

        self._add_axis_labels(view)
        return view

    def _add_axis_labels(self, view) -> None:
        label_font = QFont("Segoe UI", 10)
        tick_font = QFont("Segoe UI", 8)
        label_color = (225, 234, 244, 255)
        tick_color = (150, 164, 180, 255)

        labels = [
            ("Time, s", (self.axis_size_x + 12, -8, 0), label_font, label_color),
            ("Rotation, rpm", (-14, self.axis_size_y + 14, 0), label_font, label_color),
            ("Speed", (-12, -12, self.axis_size_z + 12), label_font, label_color),
        ]

        for text, pos, font, color in labels:
            self._add_text_item(view, text=text, pos=pos, font=font, color=color)

        for seconds in range(0, 361, 60):
            x = seconds * self.time_scale
            self._add_text_item(
                view,
                text=str(seconds),
                pos=(x, -10, 0),
                font=tick_font,
                color=tick_color,
                alignment=Qt.AlignHCenter | Qt.AlignTop,
            )

        for rotation in range(0, 181, 30):
            y = rotation * self.rotation_scale
            self._add_text_item(
                view,
                text=str(rotation),
                pos=(-10, y, 0),
                font=tick_font,
                color=tick_color,
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            )

        for speed in (0, 1, 2, 3, 4, 5):
            z = speed * self.speed_scale
            self._add_text_item(
                view,
                text=str(speed),
                pos=(-8, -8, z),
                font=tick_font,
                color=tick_color,
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            )

    def _add_text_item(
        self,
        view,
        text: str,
        pos: tuple[float, float, float],
        font: QFont,
        color: tuple[int, int, int, int],
        alignment=Qt.AlignLeft | Qt.AlignBottom,
    ) -> None:
        item = gl.GLTextItem(pos=pos, text=text, color=color, font=font, alignment=alignment)
        view.addItem(item)
        self._axis_label_items.append(item)

    def _create_live_items(self) -> None:
        if not self._live_points:
            return

        data = self._points_to_array(self._live_points)
        color = self._live_color()
        self._live_line = gl.GLLinePlotItem(
            pos=data,
            color=color,
            width=3,
            antialias=True,
            mode="line_strip",
        )
        self._live_scatter = gl.GLScatterPlotItem(
            pos=data,
            color=color,
            size=6,
            pxMode=True,
        )
        self.view.addItem(self._live_line)
        self.view.addItem(self._live_scatter)
        self._plot_items.extend([self._live_line, self._live_scatter])

    def _start_layer_segment(self, cluster_id: int, time_s: float) -> None:
        self._ensure_cluster_profiles()
        previous_point = self._live_points[-1] if self._live_points else None
        self._active_profile = self._cluster_profiles.get(cluster_id)
        self._active_cluster_id = cluster_id
        self._segment_start_time_s = time_s
        if self._graph_start_time_s is None:
            self._graph_start_time_s = time_s
        self._live_points.append(self._segment_start_point(time_s, previous_point))
        self._add_target_marker(time_s)
        self._live_segment_index += 1

    def _add_target_marker(self, segment_start_time_s: float) -> None:
        if self._active_profile is None:
            return

        target = DrillingPoint(
            time_s=segment_start_time_s + self._transition_duration_s,
            rotation=self._active_profile.optimal_rotation,
            speed=self._active_profile.optimal_speed,
            source="target",
            cluster_id=self._active_profile.cluster_id,
        )
        self._target_scatter = gl.GLScatterPlotItem(
            pos=self._points_to_array([target], first_time=self._graph_start_time_s),
            color=(1.0, 0.10, 0.08, 1.0),
            size=13,
            pxMode=True,
        )
        self.view.addItem(self._target_scatter)
        self._plot_items.append(self._target_scatter)

    def _clear_plot_items(self) -> None:
        for item in self._plot_items:
            self.view.removeItem(item)
        self._plot_items.clear()
        self._live_line = None
        self._live_scatter = None

    def _points_to_array(self, points: list[DrillingPoint], first_time: float | None = None):
        if first_time is None:
            first_time = points[0].time_s
        # OpenGL coordinates: OX=time, OY=rotation, OZ=speed.
        return np.array(
            [
                [
                    (point.time_s - first_time) * self.time_scale,
                    point.rotation * self.rotation_scale,
                    point.speed * self.speed_scale,
                ]
                for point in points
            ],
            dtype=float,
        )

    def _live_color(self) -> tuple[float, float, float, float]:
        colors = (
            (0.42, 0.62, 0.82, 1.0),
            (0.34, 0.68, 0.52, 1.0),
            (0.78, 0.50, 0.30, 1.0),
            (0.58, 0.48, 0.78, 1.0),
            (0.38, 0.66, 0.70, 1.0),
        )
        return colors[self._live_segment_index % len(colors)]

    def _ensure_cluster_profiles(self) -> None:
        if self._cluster_profiles:
            return
        self._cluster_profiles = load_cluster_profiles()

    def _profile_start_point(self, time_s: float) -> DrillingPoint:
        if self._active_profile is None:
            return DrillingPoint(time_s=time_s, rotation=0.0, speed=0.0, cluster_id=self._active_cluster_id)
        return DrillingPoint(
            time_s=time_s,
            rotation=self._active_profile.avg_rotation,
            speed=self._active_profile.avg_speed,
            source="cluster-average",
            cluster_id=self._active_profile.cluster_id,
        )

    def _segment_start_point(
        self,
        time_s: float,
        previous_point: DrillingPoint | None,
    ) -> DrillingPoint:
        if previous_point is None:
            return self._profile_start_point(time_s)
        return DrillingPoint(
            time_s=time_s,
            rotation=previous_point.rotation,
            speed=previous_point.speed,
            source="layer-transition",
            cluster_id=self._active_cluster_id,
        )

    def _interpolated_profile_point(self, time_s: float) -> DrillingPoint:
        assert self._active_profile is not None
        assert self._segment_start_time_s is not None
        start_point = self._segment_start_value()
        k = min(max((time_s - self._segment_start_time_s) / self._transition_duration_s, 0.0), 1.0)
        smooth_k = k * k * (3.0 - 2.0 * k)
        return DrillingPoint(
            time_s=time_s,
            rotation=_lerp(start_point.rotation, self._active_profile.optimal_rotation, smooth_k),
            speed=_lerp(start_point.speed, self._active_profile.optimal_speed, smooth_k),
            source="model",
            cluster_id=self._active_profile.cluster_id,
        )

    def _segment_start_value(self) -> DrillingPoint:
        assert self._segment_start_time_s is not None
        for point in reversed(self._live_points):
            if point.time_s == self._segment_start_time_s:
                return point
        return self._profile_start_point(self._segment_start_time_s)


def _lerp(start: float, end: float, k: float) -> float:
    return start + (end - start) * k


Performance3DWidgetStub = Performance3DWidget
