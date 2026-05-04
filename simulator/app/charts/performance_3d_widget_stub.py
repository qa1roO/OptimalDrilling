from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.data import (
    ClusterProfile,
    DrillingPoint,
    load_cluster_profiles,
)

try:
    import numpy as np
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    CHARTS_READY = True
except Exception:  # pragma: no cover
    CHARTS_READY = False


class Performance3DWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(760, 360)
        self.speed_scale_3d = 1800.0
        self.rotation_scale_3d = 1.0
        self.surface_size_x = 170.0
        self.surface_size_y = 120.0
        self.surface_height = 42.0
        self._plot_items: list = []
        self._surface_items: list = []
        self._live_points: list[DrillingPoint] = []
        self._surface_segment_points: list[DrillingPoint] = []
        self._mode = "live"
        self._live_segment_index = 0
        self.axis_description = "2D: time/rotation + time/speed. 3D: cluster surface."
        self._cluster_profiles: dict[int, ClusterProfile] = {}
        self._active_profile: ClusterProfile | None = None
        self._active_cluster_id: int | None = None
        self._segment_start_time_s: float | None = None
        self._graph_start_time_s: float | None = None
        self._transition_duration_s = 10.0
        self._fast_speed_drop_ratio = 0.12
        self._initial_speed_drop_k = 0.72
        self._surface_rotation_center = 100.0
        self._surface_speed_center = 0.02
        self._rotation_span = 44.0
        self._speed_span = 0.035

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.status_label = QLabel(self)
        self.status_label.setStyleSheet(
            "background: #121820; color: #c8d2dc; font-size: 12px; padding: 6px 8px;"
            "border-radius: 6px;"
        )
        layout.addWidget(self.status_label)

        if CHARTS_READY:
            self._build_views(layout)
            self.show_default_mode()
        else:
            warning = QLabel(
                "Chart backend is unavailable.\n"
                "Install dependencies from requirements.txt.",
                self,
            )
            warning.setStyleSheet("padding: 16px; color: #8a1f1f;")
            layout.addWidget(warning, 1)

    def show_default_mode(self) -> None:
        self._mode = "live"
        self.reset_live_mode()

    def reset_live_mode(self) -> None:
        if not CHARTS_READY:
            return

        self._mode = "live"
        self._live_segment_index = 0
        self._live_points.clear()
        self._surface_segment_points.clear()
        self._active_profile = None
        self._active_cluster_id = None
        self._segment_start_time_s = None
        self._graph_start_time_s = None
        self._clear_plot_items()
        self._clear_surface_items()
        self.rotation_curve.setData([], [])
        self.rotation_target_line.setValue(0)
        self.rotation_target_line.hide()
        self.speed_curve.setData([], [])
        self.speed_target_line.setValue(0)
        self.speed_target_line.hide()
        self.status_label.setText(
            f"Live charts: {self.axis_description}. Waiting until the bit touches rock."
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
        if not CHARTS_READY or self._mode != "live":
            return

        if cluster_id is None:
            point = DrillingPoint(time_s=time_s, rotation=rotation, speed=speed, source="model")
        else:
            if self._active_cluster_id != cluster_id:
                self._start_layer_segment(cluster_id=cluster_id, time_s=time_s)
            point = self._interpolated_profile_point(time_s)

        self._live_points.append(point)
        self._surface_segment_points.append(point)
        self._update_live_views(time_s)

        current = self._live_points[-1]
        self.status_label.setText(
            f"Live drilling: {self.axis_description} "
            f"cluster={current.cluster_id}, t={time_s:0.1f}s, "
            f"rotation={current.rotation:0.0f} rpm, speed={current.speed:0.4f} m/s."
        )

    def _build_views(self, layout: QVBoxLayout) -> None:
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(8)
        layout.addLayout(content, 1)

        plots = QWidget(self)
        plots_layout = QVBoxLayout(plots)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(8)

        self.rotation_plot = self._build_2d_plot("Rotation over time", "rotation, rpm")
        self.rotation_curve = self.rotation_plot.plot(
            pen=pg.mkPen("#74a6d8", width=2),
            symbol="o",
            symbolSize=4,
            symbolBrush="#74a6d8",
        )
        self.rotation_target_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("#ff3b30", width=1.5, style=Qt.DashLine),
        )
        self.rotation_plot.addItem(self.rotation_target_line)
        self.rotation_target_line.hide()

        self.speed_plot = self._build_2d_plot("Speed over time", "speed, m/s")
        self.speed_curve = self.speed_plot.plot(
            pen=pg.mkPen("#57b37e", width=2),
            symbol="o",
            symbolSize=4,
            symbolBrush="#57b37e",
        )
        self.speed_target_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("#ff3b30", width=1.5, style=Qt.DashLine),
        )
        self.speed_plot.addItem(self.speed_target_line)
        self.speed_target_line.hide()

        plots_layout.addWidget(self.rotation_plot, 1)
        plots_layout.addWidget(self.speed_plot, 1)
        content.addWidget(plots, 1)

        self.surface_view = self._build_surface_view()
        content.addWidget(self.surface_view, 1)

    def _build_2d_plot(self, title: str, left_label: str):
        plot = pg.PlotWidget()
        plot.setBackground("#121820")
        plot.setTitle(title, color="#dce6f1", size="10pt")
        plot.setLabel("bottom", "time", units="s", color="#b7c4d2")
        plot.setLabel("left", left_label, color="#b7c4d2")
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.getAxis("bottom").setPen(pg.mkPen("#5f7082"))
        plot.getAxis("left").setPen(pg.mkPen("#5f7082"))
        return plot

    def _build_surface_view(self) -> QWidget:
        view = gl.GLViewWidget()
        view.opts["distance"] = 235
        view.opts["elevation"] = 27
        view.opts["azimuth"] = -44
        view.setBackgroundColor((12, 16, 22, 255))

        axis = gl.GLAxisItem()
        axis.setSize(x=self.surface_size_x, y=self.surface_size_y, z=self.surface_height)
        view.addItem(axis)

        grid = gl.GLGridItem()
        grid.setColor((80, 92, 108, 80))
        grid.setSize(x=self.surface_size_x, y=self.surface_size_y)
        grid.setSpacing(x=20, y=15)
        view.addItem(grid)

        self._add_text_item(view, "rotation", (self.surface_size_x / 2 + 10, -8, 0))
        self._add_text_item(view, "speed", (-12, self.surface_size_y / 2 + 12, 0))
        self._add_text_item(view, "surface", (-10, -10, self.surface_height + 8))
        return view

    def _add_text_item(self, view, text: str, pos: tuple[float, float, float]) -> None:
        item = gl.GLTextItem(
            pos=pos,
            text=text,
            color=(225, 234, 244, 255),
            font=QFont("Segoe UI", 9),
            alignment=Qt.AlignLeft | Qt.AlignBottom,
        )
        view.addItem(item)

    def _start_layer_segment(self, cluster_id: int, time_s: float) -> None:
        self._ensure_cluster_profiles()
        previous_point = self._live_points[-1] if self._live_points else None
        self._active_profile = self._cluster_profiles.get(cluster_id)
        self._active_cluster_id = cluster_id
        self._segment_start_time_s = time_s
        if self._graph_start_time_s is None:
            self._graph_start_time_s = time_s

        start_point = self._segment_start_point(time_s, previous_point)
        self._live_points.append(start_point)
        self._surface_segment_points = [start_point]
        self._draw_cluster_surface()
        self._set_2d_targets()
        self._live_segment_index += 1

    def _draw_cluster_surface(self) -> None:
        self._clear_surface_items()
        if self._active_profile is None:
            return

        self._surface_rotation_center = self._active_profile.optimal_rotation
        self._surface_speed_center = self._active_profile.optimal_speed
        self._rotation_span = max(abs(self._active_profile.optimal_rotation - self._active_profile.avg_rotation) * 3.0, 34.0)
        self._speed_span = max(abs(self._active_profile.optimal_speed - self._active_profile.avg_speed) * 3.0, 0.018)

        x = np.linspace(-self.surface_size_x / 2, self.surface_size_x / 2, 36)
        y = np.linspace(-self.surface_size_y / 2, self.surface_size_y / 2, 30)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        z = self.surface_height * (0.88 - 0.38 * ((xx / (self.surface_size_x / 2)) ** 2 + (yy / (self.surface_size_y / 2)) ** 2))
        z += 4.0 * np.sin((xx + self._active_profile.cluster_id * 8.0) / 24.0)
        z += 2.5 * np.cos((yy - self._active_profile.cluster_id * 5.0) / 18.0)
        z = np.clip(z, 3.0, self.surface_height)

        surface = gl.GLSurfacePlotItem(
            x=x,
            y=y,
            z=z,
            shader="shaded",
            color=(0.18, 0.42, 0.62, 0.72),
        )
        self.surface_view.addItem(surface)
        self._surface_items.append(surface)

        self._add_target_marker()

    def _add_target_marker(self) -> None:
        if self._active_profile is None:
            return

        target = DrillingPoint(
            time_s=(self._segment_start_time_s or 0.0) + self._transition_duration_s,
            rotation=self._active_profile.optimal_rotation,
            speed=self._active_profile.optimal_speed,
            source="target",
            cluster_id=self._active_profile.cluster_id,
        )
        marker = gl.GLScatterPlotItem(
            pos=self._surface_points_to_array([target], lift=3.5),
            color=(1.0, 0.10, 0.08, 1.0),
            size=14,
            pxMode=True,
        )
        self.surface_view.addItem(marker)
        self._surface_items.append(marker)

    def _set_2d_targets(self) -> None:
        if self._active_profile is None:
            return
        self.rotation_target_line.setValue(self._active_profile.optimal_rotation)
        self.rotation_target_line.show()
        self.speed_target_line.setValue(self._active_profile.optimal_speed)
        self.speed_target_line.show()

    def _update_live_views(self, time_s: float) -> None:
        first_time = self._graph_start_time_s
        if first_time is None and self._live_points:
            first_time = self._live_points[0].time_s
        if first_time is None:
            return

        xs = [point.time_s - first_time for point in self._live_points]
        rotations = [point.rotation for point in self._live_points]
        speeds = [point.speed for point in self._live_points]
        self.rotation_curve.setData(xs, rotations)
        self.speed_curve.setData(xs, speeds)

        self._clear_plot_items()
        if self._surface_segment_points:
            data = self._surface_points_to_array(self._surface_segment_points, lift=1.8)
            color = self._live_color()
            line = gl.GLLinePlotItem(
                pos=data,
                color=color,
                width=4,
                antialias=True,
                mode="line_strip",
            )
            scatter = gl.GLScatterPlotItem(
                pos=data,
                color=color,
                size=6,
                pxMode=True,
            )
            self.surface_view.addItem(line)
            self.surface_view.addItem(scatter)
            self._plot_items.extend([line, scatter])

        self.rotation_plot.setXRange(max(0.0, time_s - first_time - 90.0), max(90.0, time_s - first_time + 8.0), padding=0)
        self.speed_plot.setXRange(max(0.0, time_s - first_time - 90.0), max(90.0, time_s - first_time + 8.0), padding=0)

    def _surface_points_to_array(self, points: list[DrillingPoint], lift: float = 0.0):
        return np.array(
            [
                [
                    self._rotation_to_surface_x(point.rotation),
                    self._speed_to_surface_y(point.speed),
                    self._surface_z(point.rotation, point.speed) + lift,
                ]
                for point in points
            ],
            dtype=float,
        )

    def _rotation_to_surface_x(self, rotation: float) -> float:
        return ((rotation - self._surface_rotation_center) / self._rotation_span) * self.surface_size_x

    def _speed_to_surface_y(self, speed: float) -> float:
        return ((speed - self._surface_speed_center) / self._speed_span) * self.surface_size_y

    def _surface_z(self, rotation: float, speed: float) -> float:
        x = self._rotation_to_surface_x(rotation)
        y = self._speed_to_surface_y(speed)
        z = self.surface_height * (0.88 - 0.38 * ((x / (self.surface_size_x / 2)) ** 2 + (y / (self.surface_size_y / 2)) ** 2))
        if self._active_profile is not None:
            z += 4.0 * np.sin((x + self._active_profile.cluster_id * 8.0) / 24.0)
            z += 2.5 * np.cos((y - self._active_profile.cluster_id * 5.0) / 18.0)
        return float(np.clip(z, 3.0, self.surface_height))

    def _clear_plot_items(self) -> None:
        for item in self._plot_items:
            self.surface_view.removeItem(item)
        self._plot_items.clear()

    def _clear_surface_items(self) -> None:
        for item in self._surface_items:
            self.surface_view.removeItem(item)
        self._surface_items.clear()

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
        speed_k = smooth_k
        if self._active_profile.optimal_speed < start_point.speed:
            fast_k = min(k / self._fast_speed_drop_ratio, 1.0)
            speed_k = max(self._initial_speed_drop_k, fast_k)
        return DrillingPoint(
            time_s=time_s,
            rotation=_lerp(start_point.rotation, self._active_profile.optimal_rotation, smooth_k),
            speed=_lerp(start_point.speed, self._active_profile.optimal_speed, speed_k),
            source="model",
            cluster_id=self._active_profile.cluster_id,
        )

    def _segment_start_value(self) -> DrillingPoint:
        assert self._segment_start_time_s is not None
        for point in reversed(self._surface_segment_points):
            if point.time_s == self._segment_start_time_s:
                return point
        return self._profile_start_point(self._segment_start_time_s)


def _lerp(start: float, end: float, k: float) -> float:
    return start + (end - start) * k


Performance3DWidgetStub = Performance3DWidget
