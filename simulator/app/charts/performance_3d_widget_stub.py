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
        self.depth_axis_max_m = 42.0
        self.axial_effort_span = 180.0
        self.rotation_effort_span = 130.0
        self._plot_items: list = []
        self._surface_items: list = []
        self._live_points: list[DrillingPoint] = []
        self._live_depths_m: list[float] = []
        self._surface_segment_points: list[DrillingPoint] = []
        self._mode = "live"
        self._live_segment_index = 0
        self.axis_description = "2D: depth/rotation + depth/speed. 3D: cluster surface."
        self._cluster_profiles: dict[int, ClusterProfile] = {}
        self._active_profile: ClusterProfile | None = None
        self._active_cluster_id: int | None = None
        self._segment_start_time_s: float | None = None
        self._graph_start_time_s: float | None = None
        self._graph_start_depth_m: float | None = None
        self._transition_duration_s = 10.0
        self._fast_speed_drop_ratio = 0.12
        self._initial_speed_drop_k = 0.72
        self._surface_rotation_center = 100.0
        self._surface_speed_center = 0.02
        self._surface_axial_center = 430.0
        self._surface_rotation_effort_center = 230.0
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
        self._live_depths_m.clear()
        self._surface_segment_points.clear()
        self._active_profile = None
        self._active_cluster_id = None
        self._segment_start_time_s = None
        self._graph_start_time_s = None
        self._graph_start_depth_m = None
        self._clear_plot_items()
        self._clear_surface_items()
        self.rotation_curve.setData([], [])
        self.rotation_plot.setXRange(0.0, self.depth_axis_max_m, padding=0)
        self.rotation_target_line.setValue(0)
        self.rotation_target_line.hide()
        self.speed_curve.setData([], [])
        self.speed_plot.setXRange(0.0, self.depth_axis_max_m, padding=0)
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
        depth_m: float | None = None,
        cluster_id: int | None = None,
    ) -> None:
        if not CHARTS_READY or self._mode != "live":
            return
        if cluster_id is None and isinstance(depth_m, int):
            cluster_id = depth_m
            depth_m = None
        if depth_m is None:
            depth_m = self._live_depths_m[-1] if self._live_depths_m else 0.0

        if cluster_id is None:
            point = DrillingPoint(time_s=time_s, rotation=rotation, speed=speed, source="model")
        else:
            if self._active_cluster_id != cluster_id:
                self._start_layer_segment(cluster_id=cluster_id, time_s=time_s, depth_m=depth_m)
            point = self._interpolated_profile_point(time_s)

        self._live_points.append(point)
        self._live_depths_m.append(depth_m)
        self._surface_segment_points.append(point)
        self._update_live_views(depth_m)

        current = self._live_points[-1]
        self.status_label.setText(
            f"Live drilling: {self.axis_description} "
            f"cluster={current.cluster_id}, depth={depth_m:0.1f}m, "
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

        self.rotation_plot = self._build_2d_plot("Rotation by depth", "rotation, rpm")
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

        self.speed_plot = self._build_2d_plot("Speed by depth", "speed, m/s")
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
        plot.setLabel("bottom", "depth", units="m", color="#b7c4d2")
        plot.setLabel("left", left_label, color="#b7c4d2")
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.setXRange(0.0, self.depth_axis_max_m, padding=0)
        plot.setLimits(xMin=0.0, xMax=self.depth_axis_max_m)
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

        self._add_3d_axis_labels(view)
        return view

    def _add_3d_axis_labels(self, view) -> None:
        self._add_text_item(view, "Axial effort, kN", (self.surface_size_x / 2 + 10, -8, 0))
        self._add_text_item(view, "Rotation effort, kN*m", (-12, self.surface_size_y / 2 + 12, 0))
        self._add_text_item(view, "Drilling speed, m/s", (-10, -10, self.surface_height + 8))

        for effort in range(340, 521, 60):
            x = self._axial_effort_to_surface_x(float(effort))
            self._add_text_item(view, str(effort), (x, -10, 0), font_size=7)
        for effort in range(170, 291, 40):
            y = self._rotation_effort_to_surface_y(float(effort))
            self._add_text_item(view, str(effort), (-10, y, 0), font_size=7)
        for speed in (0.00, 0.01, 0.02, 0.03, 0.04):
            z = self._speed_to_surface_z(speed)
            self._add_text_item(view, f"{speed:0.2f}", (-8, -8, z), font_size=7)

    def _add_text_item(self, view, text: str, pos: tuple[float, float, float], font_size: int = 9) -> None:
        item = gl.GLTextItem(
            pos=pos,
            text=text,
            color=(225, 234, 244, 255),
            font=QFont("Segoe UI", font_size),
            alignment=Qt.AlignLeft | Qt.AlignBottom,
        )
        view.addItem(item)

    def _start_layer_segment(self, cluster_id: int, time_s: float, depth_m: float) -> None:
        self._ensure_cluster_profiles()
        previous_point = self._live_points[-1] if self._live_points else None
        self._active_profile = self._cluster_profiles.get(cluster_id)
        self._active_cluster_id = cluster_id
        self._segment_start_time_s = time_s
        if self._graph_start_time_s is None:
            self._graph_start_time_s = time_s
        if self._graph_start_depth_m is None:
            self._graph_start_depth_m = depth_m

        start_point = self._segment_start_point(time_s, previous_point)
        self._live_points.append(start_point)
        self._live_depths_m.append(depth_m)
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
        self._surface_axial_center = 430.0
        self._surface_rotation_effort_center = 230.0
        self._rotation_span = max(abs(self._active_profile.optimal_rotation - self._active_profile.avg_rotation) * 3.0, 34.0)
        self._speed_span = max(abs(self._active_profile.optimal_speed - self._active_profile.avg_speed) * 3.0, 0.018)

        x = np.linspace(-self.surface_size_x / 2, self.surface_size_x / 2, 34)
        y = np.linspace(-self.surface_size_y / 2, self.surface_size_y / 2, 28)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        z = self._surface_z_from_xy(xx, yy)

        surface = gl.GLSurfacePlotItem(
            x=x,
            y=y,
            z=z,
            shader="shaded",
            color=(0.18, 0.42, 0.62, 0.72),
        )
        self.surface_view.addItem(surface)
        self._surface_items.append(surface)

        self._add_surface_wire_grid(x, y)
        self._add_start_marker()
        self._add_target_marker()

    def _add_surface_wire_grid(self, x_values, y_values) -> None:
        grid_color = (0.74, 0.86, 0.96, 0.34)
        for x in x_values[::4]:
            y_line = y_values
            x_line = np.full_like(y_line, x)
            z_line = self._surface_z_from_xy(x_line, y_line) + 0.35
            item = gl.GLLinePlotItem(
                pos=np.column_stack([x_line, y_line, z_line]),
                color=grid_color,
                width=1,
                antialias=True,
                mode="line_strip",
            )
            self.surface_view.addItem(item)
            self._surface_items.append(item)
        for y in y_values[::4]:
            x_line = x_values
            y_line = np.full_like(x_line, y)
            z_line = self._surface_z_from_xy(x_line, y_line) + 0.35
            item = gl.GLLinePlotItem(
                pos=np.column_stack([x_line, y_line, z_line]),
                color=grid_color,
                width=1,
                antialias=True,
                mode="line_strip",
            )
            self.surface_view.addItem(item)
            self._surface_items.append(item)

    def _add_start_marker(self) -> None:
        if not self._surface_segment_points:
            return
        marker = gl.GLScatterPlotItem(
            pos=self._surface_points_to_array([self._surface_segment_points[0]], lift=3.0),
            color=(0.20, 0.78, 1.0, 1.0),
            size=12,
            pxMode=True,
        )
        self.surface_view.addItem(marker)
        self._surface_items.append(marker)

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

    def _update_live_views(self, depth_m: float) -> None:
        first_depth = self._graph_start_depth_m
        if first_depth is None and self._live_depths_m:
            first_depth = self._live_depths_m[0]
        if first_depth is None:
            return

        xs = [depth - first_depth for depth in self._live_depths_m]
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

        self.rotation_plot.setXRange(0.0, self.depth_axis_max_m, padding=0)
        self.speed_plot.setXRange(0.0, self.depth_axis_max_m, padding=0)

    def _surface_points_to_array(self, points: list[DrillingPoint], lift: float = 0.0):
        return np.array(
            [
                [
                    self._axial_effort_to_surface_x(self._point_axial_effort(point)),
                    self._rotation_effort_to_surface_y(self._point_rotation_effort(point)),
                    self._speed_to_surface_z(point.speed) + lift,
                ]
                for point in points
            ],
            dtype=float,
        )

    def _point_axial_effort(self, point: DrillingPoint) -> float:
        speed_term = (self._surface_speed_center - point.speed) / max(self._speed_span, 0.001)
        rotation_term = (point.rotation - self._surface_rotation_center) / max(self._rotation_span, 1.0)
        return self._surface_axial_center + speed_term * 42.0 + rotation_term * 18.0

    def _point_rotation_effort(self, point: DrillingPoint) -> float:
        rotation_term = (point.rotation - self._surface_rotation_center) / max(self._rotation_span, 1.0)
        speed_term = (point.speed - self._surface_speed_center) / max(self._speed_span, 0.001)
        return self._surface_rotation_effort_center + rotation_term * 36.0 + speed_term * 10.0

    def _axial_effort_to_surface_x(self, axial_effort: float) -> float:
        return ((axial_effort - self._surface_axial_center) / self.axial_effort_span) * self.surface_size_x

    def _rotation_effort_to_surface_y(self, rotation_effort: float) -> float:
        return ((rotation_effort - self._surface_rotation_effort_center) / self.rotation_effort_span) * self.surface_size_y

    def _speed_to_surface_z(self, speed: float) -> float:
        return float(np.clip((speed / 0.04) * self.surface_height, 3.0, self.surface_height))

    def _surface_z_from_xy(self, x, y):
        z = self.surface_height * (0.80 - 0.33 * ((x / (self.surface_size_x / 2)) ** 2 + (y / (self.surface_size_y / 2)) ** 2))
        if self._active_profile is not None:
            z += 4.0 * np.sin((x + self._active_profile.cluster_id * 8.0) / 24.0)
            z += 2.5 * np.cos((y - self._active_profile.cluster_id * 5.0) / 18.0)
        return np.clip(z, 3.0, self.surface_height)

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
