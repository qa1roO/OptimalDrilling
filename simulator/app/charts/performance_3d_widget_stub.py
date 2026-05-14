import json
import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.data import (
    AdvisoryEngine,
    ClusterProfile,
    DrillingPoint,
    load_cluster_profiles,
)
from app.data.notebook_surfaces import load_notebook_surfaces

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
        self._plot_items: list = []
        self._surface_items: list = []
        self._marker_items: list = []
        self._advisory_current_marker = None
        self._advisory_recommended_marker = None
        self._advisory_link_item = None
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
        self._segment_start_depth_m: float | None = None
        self._segment_start_point_value: DrillingPoint | None = None
        self._active_speed_target: float | None = None
        self._graph_start_time_s: float | None = None
        self._graph_start_depth_m: float | None = None
        self._transition_duration_s = 10.0
        self._transition_depth_m = 3.0
        self._active_rotation_transition_depth_m = self._transition_depth_m
        self._previous_rotation_drop_rpm = 0.0
        self._rotation_growth_base_slowdown = 1.35
        self._rotation_drop_slowdown_per_10rpm = 0.28
        self._live_speed_min = 0.003
        self._live_speed_max = 0.035
        self._initial_rotation_rpm = 100.0
        self._fast_speed_drop_ratio = 0.12
        self._initial_speed_drop_k = 0.72
        self._surface_rotation_center = 100.0
        self._surface_speed_center = 0.02
        self._rotation_span = 44.0
        self._speed_span = 0.035
        self._advisory_engine: AdvisoryEngine | None = None
        self._advisory_error = ""
        self._current_surface_energy_type: str | None = None
        self._current_surface_source: str | None = None
        self._cached_surface_by_energy_type: dict[str, dict] = {}
        self._notebook_surfaces: dict[str, dict] = {}
        self._advisory_surface_bounds: dict[str, float] = {}
        self._advisory_surface_scene: dict[str, object] = {}
        self._advisory_surface_ranges: dict[str, dict[str, float]] = {}
        self._speed_smoothing_window = 35
        self._rotation_smoothing_window = 9
        self._depth_bin_size_m = 0.2
        self._chart_update_index = 0
        self._chart_update_stride = 3
        self._advisory_update_index = 0
        self._advisory_update_stride = 10
        self._last_advisory_compute_energy_type: str | None = None

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
            self._init_advisory_engine()
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
        self._segment_start_depth_m = None
        self._segment_start_point_value = None
        self._active_speed_target = None
        self._active_rotation_transition_depth_m = self._transition_depth_m
        self._previous_rotation_drop_rpm = 0.0
        self._current_surface_energy_type = None
        self._current_surface_source = None
        self._cached_surface_by_energy_type.clear()
        self._advisory_surface_bounds.clear()
        self._advisory_surface_scene.clear()
        self._chart_update_index = 0
        self._advisory_update_index = 0
        self._last_advisory_compute_energy_type = None
        if self._advisory_engine is not None:
            self._advisory_engine.reset()
        self._graph_start_time_s = None
        self._graph_start_depth_m = None
        self._clear_plot_items()
        self._clear_surface_items()
        self._clear_marker_items()
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
        if self._mode in {"live", "advisory"}:
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
                self._start_layer_segment(
                    cluster_id=cluster_id,
                    time_s=time_s,
                    depth_m=depth_m,
                    live_speed=speed,
                )
            point = self._interpolated_profile_point(
                time_s=time_s,
                depth_m=depth_m,
                live_speed=speed,
            )

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

    def append_advisory_telemetry(self, telemetry_row: dict, depth_m: float) -> None:
        if not CHARTS_READY:
            return
        if self._mode != "advisory":
            self.reset_live_mode()
            self._mode = "advisory"

        depth_m = self._telemetry_depth_m(telemetry_row, depth_m)
        if self._graph_start_depth_m is None:
            self._graph_start_depth_m = depth_m

        point = DrillingPoint(
            time_s=float(telemetry_row.get("elapsed_time_s", len(self._live_points))),
            rotation=_to_float(telemetry_row.get("rotation")),
            speed=_to_float(telemetry_row.get("speed")),
            source="replay",
            cluster_id=_energy_type_id(str(telemetry_row.get("rock_energy_type_final", ""))),
        )
        self._live_points.append(point)
        self._live_depths_m.append(depth_m)
        self._surface_segment_points.append(point)
        self._update_live_views(depth_m)

        if self._advisory_engine is None:
            self._draw_advisory_preview_surface_if_needed(telemetry_row)
            self._update_advisory_preview_marker(telemetry_row)
            surface_status = self._surface_status_name(self._current_surface_source)
            self.status_label.setText(
                f"Advisory: unavailable, fallback mode. Surface: {surface_status}. "
                f"{self._advisory_error or 'Check ML artifacts and dependencies.'}"
            )
            return

        should_compute = self._should_compute_advisory(str(telemetry_row.get("rock_energy_type_final", "")))
        recommendation = self._advisory_engine.update(telemetry_row, compute=should_compute)
        if recommendation is None:
            energy_type = str(telemetry_row.get("rock_energy_type_final", "unknown"))
            self._draw_advisory_preview_surface_if_needed(telemetry_row)
            self._update_advisory_preview_marker(telemetry_row)
            self.rotation_target_line.hide()
            self.speed_target_line.hide()
            surface_status = self._surface_status_name(self._current_surface_source)
            self.status_label.setText(
                "Advisory: warming up buffer, showing current pressure position "
                f"({len(self._advisory_engine.rows)}/{self._advisory_engine.min_buffer_size}), "
                f"energy={energy_type}, depth={depth_m:0.1f}m | Surface: {surface_status}."
            )
            return
        if not should_compute:
            return
        self._last_advisory_compute_energy_type = str(recommendation["energy_type"])

        energy_type = str(recommendation["energy_type"])
        self._set_advisory_targets(recommendation)
        surface = self._get_surface_for_energy_type(energy_type, recommendation)
        surface_source = str(surface.get("source", "advisory"))
        if (
            energy_type != self._current_surface_energy_type
            or self._current_surface_source != surface_source
        ):
            self._draw_advisory_surface(
                surface,
                energy_type=energy_type,
                source=surface_source,
            )
        self._update_advisory_markers(recommendation)
        current = recommendation["current"]
        recommended = recommendation["recommended"]
        surface_status = self._surface_status_name(surface_source)
        self.status_label.setText(
            f"Energy: {energy_type} | depth={depth_m:0.1f}m | Surface: {surface_status} | "
            f"current p_ax={current['pressure_axis']:0.0f}, p_rot={current['pressure_rotation']:0.0f}, "
            f"speed={current['speed']:0.4f} m/s | recommended p_ax={recommended['pressure_axis']:0.0f}, "
            f"p_rot={recommended['pressure_rotation']:0.0f} | uplift={recommended['predicted_uplift_pct']:+0.2f}% | "
            f"delta p_ax={recommended['delta_pressure_axis_pct']:+0.1f}% | "
            f"delta p_rot={recommended['delta_pressure_rotation_pct']:+0.1f}%."
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
        plot.hideButtons()
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

        self._add_text_item(view, "pressure axis", (self.surface_size_x / 2 + 10, -8, 0))
        self._add_text_item(view, "pressure rotation", (-12, self.surface_size_y / 2 + 12, 0))
        self._add_text_item(view, "speed", (-10, -10, self.surface_height + 8))
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

    def _start_layer_segment(
        self,
        cluster_id: int,
        time_s: float,
        depth_m: float,
        live_speed: float,
    ) -> None:
        self._ensure_cluster_profiles()
        previous_point = self._live_points[-1] if self._live_points else None
        previous_drop = self._current_segment_rotation_drop()
        self._active_profile = self._cluster_profiles.get(cluster_id)
        self._active_cluster_id = cluster_id
        self._segment_start_time_s = time_s
        self._segment_start_depth_m = depth_m
        if self._graph_start_time_s is None:
            self._graph_start_time_s = time_s
        if self._graph_start_depth_m is None:
            self._graph_start_depth_m = depth_m

        start_point = self._segment_start_point(time_s, previous_point, live_speed)
        self._segment_start_point_value = start_point
        self._active_speed_target = self._target_live_speed(start_point.speed)
        self._previous_rotation_drop_rpm = previous_drop
        self._active_rotation_transition_depth_m = self._rotation_transition_depth(start_point.rotation)
        self._surface_segment_points = [start_point]
        self._draw_cluster_surface()
        self._set_2d_targets()
        self._live_segment_index += 1

    def _draw_cluster_surface(self) -> None:
        self._clear_surface_items()
        if self._active_profile is None:
            return

        self._surface_rotation_center = self._active_profile.optimal_rotation
        self._surface_speed_center = self._active_speed_target or self._active_profile.optimal_speed
        self._rotation_span = max(abs(self._active_profile.optimal_rotation - self._active_profile.avg_rotation) * 3.0, 34.0)
        start_speed = self._segment_start_point_value.speed if self._segment_start_point_value is not None else self._surface_speed_center
        self._speed_span = max(abs(self._surface_speed_center - start_speed) * 3.0, 0.018)

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
            speed=self._active_speed_target or self._active_profile.optimal_speed,
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
        if self._active_speed_target is not None:
            self.speed_target_line.setValue(self._active_speed_target)
            self.speed_target_line.show()
        else:
            self.speed_target_line.hide()

    def _update_live_views(self, depth_m: float) -> None:
        self._chart_update_index += 1
        if self._chart_update_index % self._chart_update_stride and len(self._live_points) > 1:
            return

        first_depth = self._graph_start_depth_m
        if first_depth is None and self._live_depths_m:
            first_depth = self._live_depths_m[0]
        if first_depth is None:
            return

        if self._mode == "advisory":
            xs = np.asarray(self._live_depths_m, dtype=float)
        else:
            xs = np.asarray([depth - first_depth for depth in self._live_depths_m], dtype=float)
        rotations = np.asarray([point.rotation for point in self._live_points], dtype=float)
        speeds = np.asarray([point.speed for point in self._live_points], dtype=float)
        clipped_rotations = self._clip_series(rotations, low_quantile=0.02, high_quantile=0.98)
        clipped_speeds = self._clip_series(speeds, low_quantile=0.03, high_quantile=0.97)
        rotation_x, binned_rotations = self._depth_binned_curve(
            xs,
            clipped_rotations,
            bin_size=self._depth_bin_size_m,
        )
        speed_x, binned_speeds = self._depth_binned_curve(
            xs,
            clipped_speeds,
            bin_size=self._depth_bin_size_m,
        )
        self.rotation_curve.setData(
            rotation_x,
            self._rolling_mean(
                binned_rotations,
                self._rotation_smoothing_window,
            ),
        )
        self.speed_curve.setData(
            speed_x,
            self._rolling_mean(
                binned_speeds,
                self._speed_smoothing_window,
            ),
        )

        self._clear_plot_items()
        if self._surface_segment_points and self._mode != "advisory":
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

        visible_x = np.concatenate([rotation_x, speed_x]) if len(rotation_x) or len(speed_x) else xs
        x_max = max(self.depth_axis_max_m, float(np.nanmax(visible_x)) * 1.05 if len(visible_x) else 0.0)
        self.rotation_plot.setLimits(xMin=0.0, xMax=x_max)
        self.speed_plot.setLimits(xMin=0.0, xMax=x_max)
        self.rotation_plot.setXRange(0.0, x_max, padding=0)
        self.speed_plot.setXRange(0.0, x_max, padding=0)

    def _init_advisory_engine(self) -> None:
        artifact_dir = (
            Path(__file__).resolve().parents[1]
            / "ml_artifacts"
            / "drilling_advisory_light_penalty_artifacts"
        )
        repository_root = Path(__file__).resolve().parents[3]
        self._notebook_surfaces = load_notebook_surfaces(repository_root)
        ranges_path = artifact_dir / "surface_ranges_by_energy_type.json"
        if ranges_path.exists():
            with ranges_path.open("r", encoding="utf-8") as file:
                self._advisory_surface_ranges = json.load(file)
        try:
            self._advisory_engine = AdvisoryEngine(artifact_dir)
            self._advisory_surface_ranges = dict(self._advisory_engine.surface_ranges)
            self._advisory_error = ""
        except Exception as exc:
            self._advisory_engine = None
            self._advisory_error = f"{type(exc).__name__}: {exc}"

    def _set_advisory_targets(self, recommendation: dict) -> None:
        recommended = recommendation["recommended"]
        self.rotation_target_line.setValue(recommended["predicted_target_rotation"])
        self.rotation_target_line.show()
        self.speed_target_line.setValue(recommended["predicted_target_speed"])
        self.speed_target_line.show()

    def _telemetry_depth_m(self, telemetry_row: dict, fallback_depth_m: float) -> float:
        try:
            if fallback_depth_m is not None and math.isfinite(float(fallback_depth_m)):
                return float(fallback_depth_m)
        except (TypeError, ValueError):
            pass

        for column in ("depth_m", "depth"):
            value = telemetry_row.get(column)
            if value in (None, ""):
                continue
            depth_m = _to_float(value, math.nan)
            if math.isfinite(depth_m) and depth_m >= 0:
                return depth_m
        return 0.0

    def _should_compute_advisory(self, energy_type: str) -> bool:
        self._advisory_update_index += 1
        if self._advisory_engine is None:
            return False
        if len(self._advisory_engine.rows) + 1 <= self._advisory_engine.min_buffer_size:
            return True
        if self._advisory_engine.get_recommendation() is None:
            return True
        if energy_type != self._last_advisory_compute_energy_type:
            return True
        return self._advisory_update_index % self._advisory_update_stride == 0

    def _get_surface_for_energy_type(self, energy_type: str, recommendation: dict) -> dict:
        if energy_type in self._cached_surface_by_energy_type:
            return self._cached_surface_by_energy_type[energy_type]

        notebook_surface = self._notebook_surfaces.get(energy_type)
        if notebook_surface is not None:
            self._cached_surface_by_energy_type[energy_type] = notebook_surface
            return notebook_surface

        fallback_surface = dict(recommendation["surface"])
        fallback_surface["source"] = "advisory_fallback"
        self._cached_surface_by_energy_type[energy_type] = fallback_surface
        return fallback_surface

    def _draw_advisory_surface(self, surface: dict, energy_type: str, source: str) -> None:
        self._clear_surface_items()
        pressure_axis = np.asarray(surface["x"], dtype=float)
        pressure_rotation = np.asarray(surface["y"], dtype=float)
        speed = np.asarray(surface["z"], dtype=float)
        if pressure_axis.ndim != 2 or pressure_rotation.ndim != 2 or speed.ndim != 2:
            return

        x_axis = np.linspace(-self.surface_size_x / 2, self.surface_size_x / 2, pressure_axis.shape[0])
        y_axis = np.linspace(-self.surface_size_y / 2, self.surface_size_y / 2, pressure_axis.shape[1])
        z_values = self._advisory_z_values(speed)
        self._advisory_surface_bounds = {
            "x_min": float(np.nanmin(pressure_axis)),
            "x_max": float(np.nanmax(pressure_axis)),
            "y_min": float(np.nanmin(pressure_rotation)),
            "y_max": float(np.nanmax(pressure_rotation)),
            "z_min": float(np.nanmin(speed)),
            "z_max": float(np.nanmax(speed)),
        }
        self._advisory_surface_scene = {
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z": z_values,
        }

        surface_item = gl.GLSurfacePlotItem(
            x=x_axis,
            y=y_axis,
            z=z_values,
            shader="shaded",
            color=(0.18, 0.48, 0.64, 0.78),
        )
        self.surface_view.addItem(surface_item)
        self._surface_items.append(surface_item)
        self._current_surface_energy_type = energy_type
        self._current_surface_source = source

    def _update_advisory_markers(self, recommendation: dict) -> None:
        if not self._advisory_surface_bounds:
            return
        current_pos = self._advisory_point_to_scene(recommendation["current"], "predicted_target_speed")
        recommended_pos = self._advisory_point_to_scene(recommendation["recommended"], "predicted_target_speed")
        current_array = np.asarray([current_pos], dtype=float)
        recommended_array = np.asarray([recommended_pos], dtype=float)
        link_array = np.asarray([current_pos, recommended_pos], dtype=float)

        if self._advisory_current_marker is None:
            self._advisory_current_marker = gl.GLScatterPlotItem(
                pos=current_array,
                color=(0.95, 0.95, 0.95, 1.0),
                size=11,
                pxMode=True,
            )
            self.surface_view.addItem(self._advisory_current_marker)
            self._marker_items.append(self._advisory_current_marker)
        else:
            self._advisory_current_marker.setData(pos=current_array)

        if self._advisory_recommended_marker is None:
            self._advisory_recommended_marker = gl.GLScatterPlotItem(
                pos=recommended_array,
                color=(1.0, 0.12, 0.08, 1.0),
                size=14,
                pxMode=True,
            )
            self.surface_view.addItem(self._advisory_recommended_marker)
            self._marker_items.append(self._advisory_recommended_marker)
        else:
            self._advisory_recommended_marker.setData(pos=recommended_array)

        if self._advisory_link_item is None:
            self._advisory_link_item = gl.GLLinePlotItem(
                pos=link_array,
                color=(1.0, 1.0, 1.0, 0.95),
                width=3,
                antialias=True,
                mode="line_strip",
            )
            self.surface_view.addItem(self._advisory_link_item)
            self._marker_items.append(self._advisory_link_item)
        else:
            self._advisory_link_item.setData(pos=link_array)

    def _draw_advisory_preview_surface_if_needed(self, telemetry_row: dict) -> None:
        energy_type = str(telemetry_row.get("rock_energy_type_final", "unknown"))
        notebook_surface = self._cached_surface_by_energy_type.get(energy_type)
        if notebook_surface is None:
            notebook_surface = self._notebook_surfaces.get(energy_type)
            if notebook_surface is not None:
                self._cached_surface_by_energy_type[energy_type] = notebook_surface
        if notebook_surface is not None:
            source = str(notebook_surface.get("source", "notebook_plotly_surface"))
            if (
                energy_type == self._current_surface_energy_type
                and self._current_surface_source == source
            ):
                return
            self._draw_advisory_surface(
                notebook_surface,
                energy_type=energy_type,
                source=source,
            )
            return
        if (
            energy_type == self._current_surface_energy_type
            and self._current_surface_source == "advisory_fallback"
        ):
            return
        self._draw_advisory_preview_surface(telemetry_row)

    def _draw_advisory_preview_surface(self, telemetry_row: dict) -> None:
        self._clear_surface_items()
        energy_type = str(telemetry_row.get("rock_energy_type_final", "unknown"))
        p_ax = _to_float(telemetry_row.get("pressure_axis"))
        p_rot = _to_float(telemetry_row.get("pressure_rotation"))
        speed = _to_float(telemetry_row.get("speed"))
        ranges = self._surface_ranges_for_energy(energy_type, p_ax, p_rot, speed)
        self._advisory_surface_bounds = ranges

        x_axis = np.linspace(-self.surface_size_x / 2, self.surface_size_x / 2, 24)
        y_axis = np.linspace(-self.surface_size_y / 2, self.surface_size_y / 2, 24)
        xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
        median_z = _normalize_to_height(
            ranges["speed_median"],
            ranges["z_min"],
            ranges["z_max"],
            self.surface_height,
        )
        z_values = median_z + 1.6 * (xx / max(self.surface_size_x / 2, 1.0)) - 1.1 * (
            yy / max(self.surface_size_y / 2, 1.0)
        )
        z_values = np.clip(z_values, 3.0, self.surface_height)
        self._advisory_surface_scene = {
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z": z_values,
        }

        surface_item = gl.GLSurfacePlotItem(
            x=x_axis,
            y=y_axis,
            z=z_values,
            shader="shaded",
            color=(0.16, 0.34, 0.48, 0.56),
        )
        self.surface_view.addItem(surface_item)
        self._surface_items.append(surface_item)
        self._current_surface_energy_type = energy_type
        self._current_surface_source = "advisory_fallback"

    def _update_advisory_preview_marker(self, telemetry_row: dict) -> None:
        if not self._advisory_surface_bounds:
            return
        if self._advisory_recommended_marker is not None or self._advisory_link_item is not None:
            self._clear_marker_items()
        p_ax = _to_float(telemetry_row.get("pressure_axis"))
        p_rot = _to_float(telemetry_row.get("pressure_rotation"))
        speed = _to_float(telemetry_row.get("speed"))

        current_point = {
            "pressure_axis": p_ax,
            "pressure_rotation": p_rot,
            "speed": speed,
        }
        current_array = np.asarray([self._advisory_point_to_scene(current_point, "speed")], dtype=float)
        if self._advisory_current_marker is None:
            self._advisory_current_marker = gl.GLScatterPlotItem(
                pos=current_array,
                color=(0.95, 0.95, 0.95, 1.0),
                size=12,
                pxMode=True,
            )
            self.surface_view.addItem(self._advisory_current_marker)
            self._marker_items.append(self._advisory_current_marker)
        else:
            self._advisory_current_marker.setData(pos=current_array)

    def _surface_ranges_for_energy(
        self,
        energy_type: str,
        pressure_axis: float,
        pressure_rotation: float,
        speed: float,
    ) -> dict[str, float]:
        ranges = self._advisory_surface_ranges.get(energy_type, {})
        x_min = float(ranges.get("pressure_axis_q05", pressure_axis * 0.92))
        x_max = float(ranges.get("pressure_axis_q95", pressure_axis * 1.08))
        y_min = float(ranges.get("pressure_rotation_q05", pressure_rotation * 0.92))
        y_max = float(ranges.get("pressure_rotation_q95", pressure_rotation * 1.08))
        speed_median = float(ranges.get("speed_median", speed))
        z_min = min(speed, speed_median) * 0.82
        z_max = max(speed, speed_median) * 1.18
        if abs(z_max - z_min) < 1e-9:
            z_min = max(speed_median * 0.85, 0.0)
            z_max = max(speed_median * 1.15, z_min + 0.001)
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "speed_median": speed_median,
        }

    def _advisory_z_values(self, speed: np.ndarray):
        z_min = float(np.nanmin(speed))
        z_max = float(np.nanmax(speed))
        if abs(z_max - z_min) < 1e-12:
            return np.full_like(speed, self.surface_height * 0.5, dtype=float)
        normalized = (speed - z_min) / (z_max - z_min)
        return 4.0 + normalized * (self.surface_height - 8.0)

    def _advisory_point_to_scene(self, point: dict, speed_key: str) -> tuple[float, float, float]:
        bounds = self._advisory_surface_bounds
        pressure_axis = _clamp(
            float(point["pressure_axis"]),
            bounds["x_min"],
            bounds["x_max"],
        )
        pressure_rotation = _clamp(
            float(point["pressure_rotation"]),
            bounds["y_min"],
            bounds["y_max"],
        )
        x = _normalize_to_span(
            pressure_axis,
            bounds["x_min"],
            bounds["x_max"],
            self.surface_size_x,
        )
        y = _normalize_to_span(
            pressure_rotation,
            bounds["y_min"],
            bounds["y_max"],
            self.surface_size_y,
        )
        z = self._surface_scene_z_at(pressure_axis, pressure_rotation)
        return (x, y, z + 2.5)

    def _surface_scene_z_at(self, pressure_axis: float, pressure_rotation: float) -> float:
        if not self._advisory_surface_scene:
            return self.surface_height * 0.5

        z_values = np.asarray(self._advisory_surface_scene["z"], dtype=float)
        if z_values.ndim != 2 or z_values.size == 0:
            return self.surface_height * 0.5

        bounds = self._advisory_surface_bounds
        x_span = max(bounds["x_max"] - bounds["x_min"], 1e-12)
        y_span = max(bounds["y_max"] - bounds["y_min"], 1e-12)
        x_pos = (pressure_axis - bounds["x_min"]) / x_span * (z_values.shape[0] - 1)
        y_pos = (pressure_rotation - bounds["y_min"]) / y_span * (z_values.shape[1] - 1)
        x_pos = float(np.clip(x_pos, 0.0, z_values.shape[0] - 1))
        y_pos = float(np.clip(y_pos, 0.0, z_values.shape[1] - 1))
        x0 = int(np.floor(x_pos))
        y0 = int(np.floor(y_pos))
        x1 = min(x0 + 1, z_values.shape[0] - 1)
        y1 = min(y0 + 1, z_values.shape[1] - 1)
        tx = x_pos - x0
        ty = y_pos - y0
        z00 = z_values[x0, y0]
        z10 = z_values[x1, y0]
        z01 = z_values[x0, y1]
        z11 = z_values[x1, y1]
        return float(
            (1.0 - tx) * (1.0 - ty) * z00
            + tx * (1.0 - ty) * z10
            + (1.0 - tx) * ty * z01
            + tx * ty * z11
        )

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

    def _clear_marker_items(self) -> None:
        for item in self._marker_items:
            self.surface_view.removeItem(item)
        self._marker_items.clear()
        self._advisory_current_marker = None
        self._advisory_recommended_marker = None
        self._advisory_link_item = None

    def _rolling_mean(self, values, window: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if len(array) < 2 or window <= 1:
            return array
        finite = np.isfinite(array)
        safe = np.where(finite, array, 0.0)
        counts = np.cumsum(finite.astype(float))
        totals = np.cumsum(safe)
        if len(array) > window:
            counts[window:] = counts[window:] - counts[:-window]
            totals[window:] = totals[window:] - totals[:-window]
        return totals / np.maximum(counts, 1.0)

    def _clip_series(
        self,
        values: list[float],
        low_quantile: float,
        high_quantile: float,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if len(finite) < 8:
            return array
        low = float(np.nanquantile(finite, low_quantile))
        high = float(np.nanquantile(finite, high_quantile))
        if low >= high:
            return array
        return np.clip(array, low, high)

    def _depth_binned_curve(
        self,
        depths,
        values,
        bin_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        depths = np.asarray(depths, dtype=float)
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(depths) & np.isfinite(values)
        depths = depths[mask]
        values = values[mask]
        if len(depths) < 2 or bin_size <= 0:
            return depths, values

        bins = np.floor(depths / bin_size) * bin_size
        unique_bins = np.unique(bins)
        xs: list[float] = []
        ys: list[float] = []
        for depth_bin in unique_bins:
            part = values[bins == depth_bin]
            if len(part) == 0:
                continue
            xs.append(float(depth_bin + bin_size / 2.0))
            ys.append(float(np.nanmedian(part)))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    def _surface_status_name(self, source: str | None) -> str:
        if source == "notebook_plotly_surface":
            return "notebook/canonical"
        return "live advisory fallback"

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
            return DrillingPoint(
                time_s=time_s,
                rotation=self._initial_rotation_rpm,
                speed=0.0,
                cluster_id=self._active_cluster_id,
            )
        return DrillingPoint(
            time_s=time_s,
            rotation=self._initial_rotation_rpm,
            speed=self._active_profile.avg_speed,
            source="cluster-average",
            cluster_id=self._active_profile.cluster_id,
        )

    def _segment_start_point(
        self,
        time_s: float,
        previous_point: DrillingPoint | None,
        live_speed: float | None = None,
    ) -> DrillingPoint:
        if previous_point is None:
            point = self._profile_start_point(time_s)
            if live_speed is None:
                return point
            return DrillingPoint(
                time_s=point.time_s,
                rotation=point.rotation,
                speed=live_speed,
                source=point.source,
                cluster_id=point.cluster_id,
            )
        return DrillingPoint(
            time_s=time_s,
            rotation=previous_point.rotation,
            speed=live_speed if live_speed is not None else previous_point.speed,
            source="layer-transition",
            cluster_id=self._active_cluster_id,
        )

    def _interpolated_profile_point(self, time_s: float, depth_m: float, live_speed: float) -> DrillingPoint:
        assert self._active_profile is not None
        assert self._segment_start_time_s is not None
        start_point = self._segment_start_value()
        if self._segment_start_depth_m is not None:
            speed_k = (depth_m - self._segment_start_depth_m) / self._transition_depth_m
            rotation_k = (depth_m - self._segment_start_depth_m) / self._active_rotation_transition_depth_m
        else:
            speed_k = (time_s - self._segment_start_time_s) / self._transition_duration_s
            rotation_k = speed_k
        speed_k = min(max(speed_k, 0.0), 1.0)
        rotation_k = min(max(rotation_k, 0.0), 1.0)
        speed_smooth_k = speed_k * speed_k * (3.0 - 2.0 * speed_k)
        rotation_smooth_k = rotation_k * rotation_k * (3.0 - 2.0 * rotation_k)
        target_speed = self._active_speed_target
        if target_speed is None:
            target_speed = live_speed
        return DrillingPoint(
            time_s=time_s,
            rotation=_lerp(start_point.rotation, self._active_profile.optimal_rotation, rotation_smooth_k),
            speed=_lerp(start_point.speed, target_speed, speed_smooth_k),
            source="model",
            cluster_id=self._active_profile.cluster_id,
        )

    def _segment_start_value(self) -> DrillingPoint:
        assert self._segment_start_time_s is not None
        if self._segment_start_point_value is not None:
            return self._segment_start_point_value
        return self._profile_start_point(self._segment_start_time_s)

    def _target_live_speed(self, start_speed: float) -> float:
        if self._active_profile is None or self._active_profile.avg_speed <= 0:
            return _clamp(start_speed, self._live_speed_min, self._live_speed_max)
        model_ratio = self._active_profile.optimal_speed / self._active_profile.avg_speed
        model_ratio = _clamp(model_ratio, 1.08, 1.35)
        return _clamp(start_speed * model_ratio, self._live_speed_min, self._live_speed_max)

    def _current_segment_rotation_drop(self) -> float:
        if self._segment_start_point_value is None or not self._surface_segment_points:
            return 0.0
        min_rotation = min(point.rotation for point in self._surface_segment_points)
        return max(self._segment_start_point_value.rotation - min_rotation, 0.0)

    def _rotation_transition_depth(self, start_rotation: float) -> float:
        if self._active_profile is None:
            return self._transition_depth_m
        if self._active_profile.optimal_rotation <= start_rotation:
            return self._transition_depth_m

        drop_factor = min(self._previous_rotation_drop_rpm, 35.0) / 10.0
        slowdown = self._rotation_growth_base_slowdown + drop_factor * self._rotation_drop_slowdown_per_10rpm
        return self._transition_depth_m * slowdown


def _lerp(start: float, end: float, k: float) -> float:
    return start + (end - start) * k


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _energy_type_id(energy_type: str) -> int:
    mapping = {
        "soft_low_energy": 0,
        "medium_low_energy": 1,
        "medium_high_energy": 2,
        "hard_high_energy": 3,
    }
    return mapping.get(energy_type, -1)


def _normalize_to_span(value: float, minimum: float, maximum: float, span: float) -> float:
    if abs(maximum - minimum) < 1e-12:
        return 0.0
    return ((value - minimum) / (maximum - minimum) - 0.5) * span


def _normalize_to_height(value: float, minimum: float, maximum: float, height: float) -> float:
    if abs(maximum - minimum) < 1e-12:
        return height * 0.5
    return 4.0 + ((value - minimum) / (maximum - minimum)) * (height - 8.0)


Performance3DWidgetStub = Performance3DWidget
