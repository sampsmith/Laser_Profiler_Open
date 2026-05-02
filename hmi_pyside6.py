from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import (
    apply_calibration,
    build_layered_cloud_from_depth_rows,
    build_single_line_cloud,
    centroids_to_depth,
    draw_overlay,
    extract_green_centroids,
    load_calibration,
    read_image,
)
from calibration_window import CalibrationWindow


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content_widget: QWidget, expanded: bool = True) -> None:
        super().__init__()
        self.toggle_button = QToolButton(text=title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_button.clicked.connect(self._on_toggled)

        self.content_widget = content_widget
        self.content_widget.setVisible(expanded)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle_button)
        layout.addWidget(divider)
        layout.addWidget(self.content_widget)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content_widget.setVisible(checked)


class LaserPrototypeQt(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Laser Prototype HMI (PySide6)")
        self.resize(1500, 900)

        self.image_path = ""
        self.bgr = None
        self.depths = None
        self.raw_depths = None
        self.cloud = None
        self.layered_cloud = None
        self.sequence_depth_rows: list[np.ndarray] = []
        self.sequence_frame_positions_mm = np.array([], dtype=np.float32)
        self.depth_map = None
        self.display_depths = None
        self.filtered_cloud = None
        self.filtered_layered_cloud = None
        self.filtered_depth_map = None
        self.sequence_paths: list[str] = []
        self.calibration = None
        self.calibration_meta: dict[str, float] = {}
        self.caliper_mode = False
        self.caliper_target = "profile"
        self.caliper_points: list[tuple[float, float]] = []
        self.caliper_line: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.depth_map_caliper_line: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.z_ground_offset_mm = 0.0

        self.inputs: dict[str, QLineEdit] = {}
        self.roi_inputs: dict[str, QLineEdit] = {}
        self.profile_source_mode = "Current image"
        self.sequence_profile_index = 0
        self.calibration_window: CalibrationWindow | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        # Left: scrollable controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(320)
        left_panel = QWidget()
        left_scroll.setWidget(left_panel)
        left_layout = QVBoxLayout(left_panel)
        splitter.addWidget(left_scroll)

        # Right: tabbed views
        self.right_tabs = QTabWidget()
        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])

        self.image_fig = Figure(figsize=(7, 5))
        self.ax_image = self.image_fig.add_subplot(111)
        self.image_canvas = FigureCanvas(self.image_fig)
        self.right_tabs.addTab(self.image_canvas, "Mask + Overlay")

        self.profile_fig = Figure(figsize=(7, 5))
        self.ax_profile = self.profile_fig.add_subplot(111)
        self.profile_canvas = FigureCanvas(self.profile_fig)
        self.profile_canvas.mpl_connect("motion_notify_event", self._on_profile_hover)
        self.profile_canvas.mpl_connect("button_press_event", self._on_profile_click)
        self.right_tabs.addTab(self.profile_canvas, "Depth Profile")

        self.cloud_fig = Figure(figsize=(7, 5))
        self.ax3d = self.cloud_fig.add_subplot(111, projection="3d")
        self.cloud_canvas = FigureCanvas(self.cloud_fig)
        self.right_tabs.addTab(self.cloud_canvas, "3D Cloud")

        self.depth_map_fig = Figure(figsize=(7, 5))
        self.ax_depth_map = self.depth_map_fig.add_subplot(111)
        self.depth_map_canvas = FigureCanvas(self.depth_map_fig)
        self.depth_map_canvas.mpl_connect("button_press_event", self._on_depth_map_click)
        self.depth_map_colorbar = None
        self.right_tabs.addTab(self.depth_map_canvas, "Depth Map")
        self.filter_tab = QWidget()
        self.right_tabs.addTab(self.filter_tab, "Filtering Studio")
        self._build_filtering_tab()

        self._build_toolbar()

        # Simplified workflow controls.
        workflow_panel = QWidget()
        workflow_layout = QVBoxLayout(workflow_panel)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        for label, cb in [
            ("Load Image", self.load_image),
            ("Process Current Image", self.process_current),
            ("Load Calibration JSON", self.load_cal_json),
            ("Open Calibration Window", self.open_calibration_window),
        ]:
            b = QPushButton(label)
            b.clicked.connect(cb)
            workflow_layout.addWidget(b)
        left_layout.addWidget(CollapsibleSection("Workflow", workflow_panel, expanded=True))

        status_panel = QWidget()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.path_label = QLabel("No image loaded")
        self.path_label.setWordWrap(True)
        self.seq_label = QLabel("No sequence loaded")
        self.calibration_label = QLabel("Calibration: none loaded")
        self.depth_hover_label = QLabel("Depth @ cursor: -")
        self.caliper_label = QLabel("Calipers: off")
        self.profile_source_combo = QComboBox()
        self.profile_source_combo.addItems(["Current image", "Sequence frame"])
        self.profile_source_combo.currentTextChanged.connect(self._on_profile_source_changed)
        self.sequence_profile_spin = QSpinBox()
        self.sequence_profile_spin.setMinimum(0)
        self.sequence_profile_spin.setMaximum(0)
        self.sequence_profile_spin.setEnabled(False)
        self.sequence_profile_spin.valueChanged.connect(self._on_sequence_profile_index_changed)
        self.profile_visual_scale = QLineEdit("1.0")
        self.apply_profile_view_btn = QPushButton("Apply Profile View Scale")
        self.apply_profile_view_btn.clicked.connect(self.refresh_profile_view)
        self.cloud_visual_z_scale = QLineEdit("1.0")
        self.cloud_point_size = QLineEdit("3.0")
        self.apply_cloud_view_btn = QPushButton("Apply 3D Cloud Z Scale")
        self.apply_cloud_view_btn.clicked.connect(self.refresh_profile_view)
        self.z_ground_combo = QComboBox()
        self.z_ground_combo.addItems(
            ["Absolute Z (sensor / calibrated mm)", "Ground Z (subtract min → Z=0 at lowest point)"]
        )
        self.z_ground_combo.setCurrentIndex(1)
        self.z_ground_combo.currentIndexChanged.connect(self.refresh_profile_view)
        self.z_ground_status_label = QLabel("Z offset: 0.000 mm")
        self.manual_positions_input: QLineEdit | None = None
        self.filter_status_label = QLabel("Filter: inactive")
        self.sequence_spacing_label = QLabel("Sequence spacing: uniform step")
        self.sequence_output_mode = QComboBox()
        self.depth_map_style = QComboBox()
        self.depth_map_show_colorbar = QCheckBox("Show depth-map colorbar (slower)")
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.path_label)
        status_layout.addWidget(self.seq_label)
        status_layout.addWidget(self.calibration_label)
        status_layout.addWidget(self.depth_hover_label)
        status_layout.addWidget(self.caliper_label)
        status_layout.addWidget(QLabel("Depth profile source"))
        status_layout.addWidget(self.profile_source_combo)
        status_layout.addWidget(QLabel("Sequence profile frame index"))
        status_layout.addWidget(self.sequence_profile_spin)
        status_layout.addWidget(QLabel("Profile visual Z scale (display only)"))
        status_layout.addWidget(self.profile_visual_scale)
        status_layout.addWidget(self.apply_profile_view_btn)
        status_layout.addWidget(QLabel("3D cloud visual Z scale (display only)"))
        status_layout.addWidget(self.cloud_visual_z_scale)
        status_layout.addWidget(QLabel("3D cloud point size"))
        status_layout.addWidget(self.cloud_point_size)
        status_layout.addWidget(self.apply_cloud_view_btn)
        status_layout.addWidget(QLabel("Z reference (point cloud / profile / depth map)"))
        status_layout.addWidget(self.z_ground_combo)
        status_layout.addWidget(self.z_ground_status_label)
        status_layout.addWidget(self.filter_status_label)
        self.depth_map_style.addItems(["Grayscale (fast)", "Heat map (fast)"])
        self.depth_map_style.currentIndexChanged.connect(self._update_depth_map_plot)
        self.depth_map_show_colorbar.setChecked(False)
        self.depth_map_show_colorbar.stateChanged.connect(self._update_depth_map_plot)
        status_layout.addWidget(QLabel("Depth map style"))
        status_layout.addWidget(self.depth_map_style)
        status_layout.addWidget(self.depth_map_show_colorbar)
        status_layout.addWidget(self.status_label)
        left_layout.addWidget(CollapsibleSection("Status + View", status_panel, expanded=True))

        # Processing params
        params_panel = QWidget()
        params_form = QFormLayout(params_panel)
        defaults = {
            "h_min": "50", "h_max": "80", "s_min": "80", "s_max": "255", "v_min": "80", "v_max": "255",
            "blur_kernel": "5", "median_kernel": "0", "morph_open": "0", "morph_close": "0",
            "min_blob_area": "0", "centroid_smooth_window": "0",
            "laser_angle_deg": "30", "mm_per_pixel": "0.1", "zero_row": "0.0", "frame_step_mm": "1.0",
        }
        for k, v in defaults.items():
            e = QLineEdit(v)
            self.inputs[k] = e
            params_form.addRow(k, e)
        self.scan_axis = QComboBox()
        self.scan_axis.addItems(["y", "x"])
        params_form.addRow("scan_axis", self.scan_axis)
        left_layout.addWidget(CollapsibleSection("Parameters", params_panel, expanded=False))

        process_panel = QWidget()
        process_layout = QVBoxLayout(process_panel)
        process_layout.setContentsMargins(0, 0, 0, 0)
        self.sequence_spacing_mode = QComboBox()
        self.sequence_spacing_mode.addItems(["Uniform step (frame_step_mm)", "Manual frame positions (mm)"])
        process_layout.addWidget(QLabel("Sequence spacing mode"))
        process_layout.addWidget(self.sequence_spacing_mode)
        self.sequence_output_mode.addItems(["Depth map + 3D cloud", "Depth map only (fast)"])
        process_layout.addWidget(QLabel("Sequence output mode"))
        process_layout.addWidget(self.sequence_output_mode)
        self.manual_positions_input = QLineEdit("")
        self.manual_positions_input.setPlaceholderText("0, 1.0, 2.1, 3.1, ...")
        process_layout.addWidget(QLabel("Manual frame positions [mm] (comma-separated, one per image)"))
        process_layout.addWidget(self.manual_positions_input)
        self.autofill_positions_btn = QPushButton("Auto-fill Manual Positions from frame_step_mm")
        self.autofill_positions_btn.clicked.connect(self.autofill_manual_positions)
        process_layout.addWidget(self.autofill_positions_btn)
        process_layout.addWidget(self.sequence_spacing_label)
        for label, cb in [
            ("Load Sequence Images", self.load_sequence_images),
            ("Build Layered Cloud", self.build_layered_cloud),
            ("Save Depth Map Image", self.save_depth_map_image),
            ("Save Depth Map CSV", self.save_depth_map_csv),
            ("Save Layered Cloud CSV", self.save_layered_cloud_csv),
            ("Save Current Cloud CSV", self.save_cloud_csv),
            ("Start Profile Calipers", self.start_profile_calipers),
            ("Clear Profile Calipers", self.clear_profile_calipers),
        ]:
            b = QPushButton(label)
            b.clicked.connect(cb)
            process_layout.addWidget(b)
        left_layout.addWidget(CollapsibleSection("Sequence Process", process_panel, expanded=False))
        left_layout.addStretch(1)

    def _build_filtering_tab(self) -> None:
        layout = QVBoxLayout(self.filter_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Filtering Studio")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(title)

        hint = QLabel("Use ROI/Z limits for profile, cloud, and depth map. Leave fields blank to disable limits.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        bounds_box = QGroupBox("Axis Bounds (mm)")
        bounds_grid = QGridLayout(bounds_box)
        bounds_grid.setHorizontalSpacing(12)
        bounds_grid.setVerticalSpacing(8)
        filter_defaults = {
            "x_min": "", "x_max": "",
            "y_min": "", "y_max": "",
            "z_min": "", "z_max": "",
        }
        labels = [
            ("X min", "x_min", 0, 0), ("X max", "x_max", 0, 2),
            ("Y min", "y_min", 1, 0), ("Y max", "y_max", 1, 2),
            ("Z min", "z_min", 2, 0), ("Z max", "z_max", 2, 2),
        ]
        for label, key, row, col in labels:
            entry = QLineEdit(filter_defaults[key])
            entry.setPlaceholderText("disabled")
            self.roi_inputs[key] = entry
            bounds_grid.addWidget(QLabel(label), row, col)
            bounds_grid.addWidget(entry, row, col + 1)
        layout.addWidget(bounds_box)

        actions_row = QHBoxLayout()
        self.apply_filter_btn = QPushButton("Apply Filters")
        self.apply_filter_btn.clicked.connect(self.apply_filters)
        self.clear_filter_btn = QPushButton("Clear Filters")
        self.clear_filter_btn.clicked.connect(self.clear_filters)
        self.filter_auto_apply = QCheckBox("Live update as I type")
        self.filter_auto_apply.setChecked(True)
        actions_row.addWidget(self.apply_filter_btn)
        actions_row.addWidget(self.clear_filter_btn)
        actions_row.addWidget(self.filter_auto_apply)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        # Debounce live-edit updates so we don't re-render on every keystroke
        # while the user is still typing a number like "-1.2".
        self._filter_live_timer = QTimer(self)
        self._filter_live_timer.setSingleShot(True)
        self._filter_live_timer.setInterval(120)
        self._filter_live_timer.timeout.connect(self._apply_filters_if_valid)

        for entry in self.roi_inputs.values():
            entry.textEdited.connect(self._on_filter_field_edited)
            entry.editingFinished.connect(self._on_filter_field_edited)

        layout.addWidget(QLabel("Tip: switch to Depth Map tab while tuning filters for fastest feedback."))
        layout.addStretch(1)

    def _on_filter_field_edited(self, *_args) -> None:
        if getattr(self, "filter_auto_apply", None) is None:
            return
        if not self.filter_auto_apply.isChecked():
            return
        self._filter_live_timer.start()

    def _apply_filters_if_valid(self) -> None:
        # While the user is mid-typing ("-", "1.", "1e") the float parse throws.
        # Silently skip until the field is a valid number rather than popping an
        # error dialog every few keystrokes.
        try:
            self._normalized_roi_limits()
        except ValueError:
            return
        self.apply_filters()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.caliper_toggle_action = QAction("Calipers", self)
        self.caliper_toggle_action.setCheckable(True)
        self.caliper_toggle_action.toggled.connect(self._on_toolbar_calipers_toggled)
        toolbar.addAction(self.caliper_toggle_action)

        clear_action = QAction("Clear Calipers", self)
        clear_action.triggered.connect(self.clear_profile_calipers)
        toolbar.addAction(clear_action)

    def _f(self, key: str) -> float:
        return float(self.inputs[key].text().strip())

    def _i(self, key: str) -> int:
        return int(float(self.inputs[key].text().strip()))

    def _extract(self, bgr: np.ndarray):
        return extract_green_centroids(
            bgr,
            self._i("h_min"), self._i("h_max"),
            self._i("s_min"), self._i("s_max"),
            self._i("v_min"), self._i("v_max"),
            self._i("blur_kernel"),
            self._i("median_kernel"),
            self._i("morph_open"),
            self._i("morph_close"),
            self._i("min_blob_area"),
            self._i("centroid_smooth_window"),
        )

    def _optional_float(self, entry: QLineEdit) -> float | None:
        text = entry.text().strip()
        if not text:
            return None
        return float(text)

    def _roi_limits(self) -> dict[str, float | None]:
        return {
            key: self._optional_float(widget)
            for key, widget in self.roi_inputs.items()
        }

    def _normalized_roi_limits(self) -> dict[str, float | None]:
        limits = self._roi_limits()
        for axis in ("x", "y", "z"):
            lo_key = f"{axis}_min"
            hi_key = f"{axis}_max"
            lo = limits[lo_key]
            hi = limits[hi_key]
            if lo is not None and hi is not None and lo > hi:
                limits[lo_key], limits[hi_key] = hi, lo
        return limits

    def _has_active_limits(self, limits: dict[str, float | None]) -> bool:
        return any(v is not None for v in limits.values())

    def _apply_cloud_roi_filter(self, cloud: np.ndarray) -> np.ndarray:
        if cloud is None or cloud.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        limits = self._normalized_roi_limits()
        mask = np.isfinite(cloud).all(axis=1)
        x = cloud[:, 0]
        y = cloud[:, 1]
        # Compare Z in the same frame the user sees on screen (grounded if enabled).
        z_off = self._get_z_ground_offset_mm()
        z_disp = cloud[:, 2] - z_off
        if limits["x_min"] is not None:
            mask &= x >= float(limits["x_min"])
        if limits["x_max"] is not None:
            mask &= x <= float(limits["x_max"])
        if limits["y_min"] is not None:
            mask &= y >= float(limits["y_min"])
        if limits["y_max"] is not None:
            mask &= y <= float(limits["y_max"])
        if limits["z_min"] is not None:
            mask &= z_disp >= float(limits["z_min"])
        if limits["z_max"] is not None:
            mask &= z_disp <= float(limits["z_max"])
        return cloud[mask]

    def _apply_profile_roi_filter(self, depths: np.ndarray | None, profile_y_mm: float = 0.0) -> np.ndarray | None:
        if depths is None:
            return None
        out = depths.astype(np.float64, copy=True)
        limits = self._normalized_roi_limits()
        mm_per_pixel = self._f("mm_per_pixel")
        cols_mm = np.arange(out.shape[0], dtype=np.float64) * mm_per_pixel
        if limits["x_min"] is not None:
            out[cols_mm < float(limits["x_min"])] = np.nan
        if limits["x_max"] is not None:
            out[cols_mm > float(limits["x_max"])] = np.nan
        # Compare Z in the same frame shown on the profile (grounded if enabled).
        z_off = self._get_z_ground_offset_mm()
        if limits["z_min"] is not None:
            out[(out - z_off) < float(limits["z_min"])] = np.nan
        if limits["z_max"] is not None:
            out[(out - z_off) > float(limits["z_max"])] = np.nan
        y_min = limits["y_min"]
        y_max = limits["y_max"]
        if (y_min is not None and profile_y_mm < float(y_min)) or (y_max is not None and profile_y_mm > float(y_max)):
            out[:] = np.nan
        return out

    def _profile_frame_position_mm(self) -> float:
        if self.profile_source_combo.currentText() == "Sequence frame" and self.sequence_frame_positions_mm.size > 0:
            idx = int(self.sequence_profile_spin.value())
            idx = max(0, min(idx, self.sequence_frame_positions_mm.shape[0] - 1))
            return float(self.sequence_frame_positions_mm[idx])
        return 0.0

    def _active_profile_depths(self) -> np.ndarray | None:
        if self.profile_source_combo.currentText() == "Sequence frame":
            if not self.sequence_depth_rows:
                return None
            idx = int(self.sequence_profile_spin.value())
            idx = max(0, min(idx, len(self.sequence_depth_rows) - 1))
            return self.sequence_depth_rows[idx]
        return self.depths

    def _sync_sequence_profile_controls(self) -> None:
        has_sequence = len(self.sequence_depth_rows) > 0
        self.sequence_profile_spin.setEnabled(has_sequence)
        if has_sequence:
            self.sequence_profile_spin.setMaximum(len(self.sequence_depth_rows) - 1)
            if self.sequence_profile_spin.value() > self.sequence_profile_spin.maximum():
                self.sequence_profile_spin.setValue(self.sequence_profile_spin.maximum())
        else:
            self.sequence_profile_spin.setMaximum(0)
            self.sequence_profile_spin.setValue(0)

    def _on_profile_source_changed(self, mode: str) -> None:
        self.profile_source_mode = mode
        self.apply_filters()

    def _on_sequence_profile_index_changed(self, _: int) -> None:
        if self.profile_source_combo.currentText() == "Sequence frame":
            self.apply_filters()

    def _active_raw_cloud(self) -> np.ndarray:
        if self.layered_cloud is not None and self.layered_cloud.size > 0:
            return self.layered_cloud
        if self.cloud is not None and self.cloud.size > 0:
            return self.cloud
        return np.zeros((0, 3), dtype=np.float32)

    def _z_grounding_enabled(self) -> bool:
        if not hasattr(self, "z_ground_combo"):
            return False
        return self.z_ground_combo.currentIndex() == 1

    def _compute_min_z_reference_mm(self) -> float:
        """Smallest finite Z in the current dataset (for grounding to zero)."""
        raw = self._active_raw_cloud()
        if raw.size > 0:
            z = raw[:, 2]
            fin = z[np.isfinite(z)]
            if fin.size > 0:
                return float(np.nanmin(fin))
        if self.depth_map is not None and self.depth_map.size > 0:
            dm = self.depth_map
            fin = dm[np.isfinite(dm)]
            if fin.size > 0:
                return float(np.nanmin(fin))
        if self.depths is not None:
            d = self.depths
            fin = d[np.isfinite(d)]
            if fin.size > 0:
                return float(np.nanmin(fin))
        return 0.0

    def _get_z_ground_offset_mm(self) -> float:
        if not self._z_grounding_enabled():
            return 0.0
        return self._compute_min_z_reference_mm()

    def _ground_cloud_for_display(self, cloud: np.ndarray) -> np.ndarray:
        off = self._get_z_ground_offset_mm()
        if off == 0.0 or cloud.size == 0:
            return cloud
        out = cloud.astype(np.float64, copy=True)
        out[:, 2] = out[:, 2] - off
        return out.astype(np.float32, copy=False)

    def _ground_depths_for_display(self, depths: np.ndarray | None) -> np.ndarray | None:
        if depths is None:
            return None
        off = self._get_z_ground_offset_mm()
        if off == 0.0:
            return depths
        out = depths.astype(np.float64, copy=True)
        m = np.isfinite(out)
        out[m] = out[m] - off
        return out

    def _ground_depth_map_for_display(self, depth_map: np.ndarray) -> np.ndarray:
        off = self._get_z_ground_offset_mm()
        if off == 0.0:
            return depth_map
        out = depth_map.astype(np.float64, copy=True)
        m = np.isfinite(out)
        out[m] = out[m] - off
        return out.astype(np.float32, copy=False)

    def _active_filtered_cloud(self) -> np.ndarray:
        if self.layered_cloud is not None and self.filtered_layered_cloud is not None:
            return self.filtered_layered_cloud
        if self.cloud is not None and self.filtered_cloud is not None:
            return self.filtered_cloud
        return np.zeros((0, 3), dtype=np.float32)

    def _parse_manual_positions(self, expected_count: int) -> np.ndarray:
        if self.manual_positions_input is None:
            raise ValueError("Manual positions input is not initialized.")
        raw = self.manual_positions_input.text().strip()
        if not raw:
            raise ValueError("Manual positions are empty. Enter comma-separated values.")
        tokens = [part.strip() for part in raw.split(",") if part.strip()]
        if len(tokens) != expected_count:
            raise ValueError(
                f"Manual positions count mismatch: expected {expected_count}, got {len(tokens)}."
            )
        vals = np.array([float(tok) for tok in tokens], dtype=np.float32)
        if not np.all(np.isfinite(vals)):
            raise ValueError("Manual positions must be finite numbers.")
        if np.any(np.diff(vals) < 0):
            raise ValueError("Manual positions must be non-decreasing.")
        return vals

    def _build_layered_cloud_with_positions(
        self,
        depth_rows: list[np.ndarray],
        mm_per_pixel: float,
        frame_positions_mm: np.ndarray,
        scan_axis: str,
    ) -> np.ndarray:
        if scan_axis not in ("x", "y"):
            raise ValueError("scan_axis must be 'x' or 'y'")
        if len(depth_rows) != frame_positions_mm.shape[0]:
            raise ValueError("Frame positions length must match number of frames.")
        points = []
        for depths, pos_mm in zip(depth_rows, frame_positions_mm):
            valid = ~np.isnan(depths)
            cols = np.where(valid)[0].astype(np.float32)
            z = depths[valid].astype(np.float32)
            if cols.size == 0:
                continue
            if scan_axis == "y":
                x = cols * mm_per_pixel
                y = np.full_like(x, float(pos_mm), dtype=np.float32)
            else:
                x = np.full_like(cols, float(pos_mm), dtype=np.float32)
                y = cols * mm_per_pixel
            points.append(np.column_stack([x, y, z]))
        if not points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(points).astype(np.float32)

    def _center_to_edge_axis(self, centers: np.ndarray) -> np.ndarray:
        if centers.size == 0:
            return np.array([0.0, 1.0], dtype=np.float32)
        if centers.size == 1:
            c = float(centers[0])
            return np.array([c - 0.5, c + 0.5], dtype=np.float32)
        mids = 0.5 * (centers[:-1] + centers[1:])
        first = centers[0] - (mids[0] - centers[0])
        last = centers[-1] + (centers[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]]).astype(np.float32)

    def _build_depth_map(self, depth_rows: list[np.ndarray]) -> np.ndarray:
        if not depth_rows:
            return np.zeros((0, 0), dtype=np.float32)
        widths = [int(r.shape[0]) for r in depth_rows]
        max_w = max(widths)
        out = np.full((len(depth_rows), max_w), np.nan, dtype=np.float32)
        for i, row in enumerate(depth_rows):
            out[i, : row.shape[0]] = row.astype(np.float32, copy=False)
        return out

    def _filter_depth_map(self) -> np.ndarray | None:
        if self.depth_map is None or self.depth_map.size == 0:
            return None
        out = self.depth_map.copy()
        limits = self._normalized_roi_limits()
        mm_per_pixel = self._f("mm_per_pixel")
        x_mm = np.arange(out.shape[1], dtype=np.float32) * mm_per_pixel
        y_mm = self.sequence_frame_positions_mm.astype(np.float32, copy=False)
        if limits["x_min"] is not None:
            out[:, x_mm < float(limits["x_min"])] = np.nan
        if limits["x_max"] is not None:
            out[:, x_mm > float(limits["x_max"])] = np.nan
        if limits["y_min"] is not None and y_mm.size == out.shape[0]:
            out[y_mm < float(limits["y_min"]), :] = np.nan
        if limits["y_max"] is not None and y_mm.size == out.shape[0]:
            out[y_mm > float(limits["y_max"]), :] = np.nan
        # Z limits are compared in the same frame the user sees on the depth map
        # (grounded to zero at the lowest point if grounding is enabled). NaNs
        # stay NaN because NaN compares False, so they are never re-assigned.
        z_off = self._get_z_ground_offset_mm()
        if limits["z_min"] is not None:
            out[(out - z_off) < float(limits["z_min"])] = np.nan
        if limits["z_max"] is not None:
            out[(out - z_off) > float(limits["z_max"])] = np.nan
        return out

    def _clear_depth_map_colorbar(self) -> None:
        if self.depth_map_colorbar is None:
            return
        cb = self.depth_map_colorbar
        self.depth_map_colorbar = None
        try:
            cb.remove()
            return
        except Exception:
            pass
        # Fallback for matplotlib edge-cases where Colorbar.remove() fails.
        try:
            if getattr(cb, "ax", None) is not None:
                cb.ax.remove()
        except Exception:
            pass

    def _update_depth_map_plot(self) -> None:
        self._clear_depth_map_colorbar()
        self.ax_depth_map.clear()
        style = self.depth_map_style.currentText() if hasattr(self, "depth_map_style") else "Grayscale (fast)"
        title_suffix = "Grayscale" if "Grayscale" in style else "Heat Map"
        self.ax_depth_map.set_title(f"Sequence Depth Map ({title_suffix})")
        self.ax_depth_map.set_xlabel("X (mm)")
        self.ax_depth_map.set_ylabel("Scan position (mm)")
        depth_map = self.filtered_depth_map if self.filtered_depth_map is not None else self.depth_map
        if depth_map is None or depth_map.size == 0:
            self.ax_depth_map.text(0.5, 0.5, "No depth map built", ha="center", va="center", transform=self.ax_depth_map.transAxes)
            self.depth_map_canvas.draw_idle()
            return
        depth_map = self._ground_depth_map_for_display(depth_map)
        mm_per_pixel = self._f("mm_per_pixel")
        x_centers = np.arange(depth_map.shape[1], dtype=np.float32) * mm_per_pixel
        y_centers = self.sequence_frame_positions_mm.astype(np.float32, copy=False)
        x_edges = self._center_to_edge_axis(x_centers)
        y_edges = self._center_to_edge_axis(y_centers)
        data = np.ma.masked_invalid(depth_map)
        finite = depth_map[np.isfinite(depth_map)]
        cmap_name = "gray" if "Grayscale" in style else "inferno"
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="black")
        if finite.size > 0:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if abs(vmax - vmin) < 1e-9:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        mesh = self.ax_depth_map.pcolormesh(
            x_edges,
            y_edges,
            data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if self.depth_map_show_colorbar.isChecked():
            self.depth_map_colorbar = self.depth_map_fig.colorbar(mesh, ax=self.ax_depth_map, pad=0.02)
            self.depth_map_colorbar.set_label("Z (mm, relative)" if self._z_grounding_enabled() else "Z (mm)")
        if self.depth_map_caliper_line is not None:
            (p1, p2) = self.depth_map_caliper_line
            self.ax_depth_map.plot([p1[0], p2[0]], [p1[1], p2[1]], color="orange", linewidth=2.0)
            self.ax_depth_map.scatter([p1[0], p2[0]], [p1[1], p2[1]], color="orange", s=20)
        limits = self._normalized_roi_limits()
        x_lo = float(x_edges[0])
        x_hi = float(x_edges[-1])
        y_lo = float(y_edges[0])
        y_hi = float(y_edges[-1])
        x_min = limits["x_min"] if limits["x_min"] is not None else x_lo
        x_max = limits["x_max"] if limits["x_max"] is not None else x_hi
        y_min = limits["y_min"] if limits["y_min"] is not None else y_lo
        y_max = limits["y_max"] if limits["y_max"] is not None else y_hi
        x_min = max(x_lo, min(float(x_min), x_hi))
        x_max = max(x_lo, min(float(x_max), x_hi))
        y_min = max(y_lo, min(float(y_min), y_hi))
        y_max = max(y_lo, min(float(y_max), y_hi))
        if x_min >= x_max:
            x_min, x_max = x_lo, x_hi
        if y_min >= y_max:
            y_min, y_max = y_lo, y_hi
        self.ax_depth_map.set_xlim(x_min, x_max)
        # Keep first frame visually at the top, like image row ordering.
        self.ax_depth_map.set_ylim(y_max, y_min)
        self.ax_depth_map.set_aspect("auto")
        self.depth_map_canvas.draw_idle()

    def _active_measurement_target(self) -> str | None:
        tab_title = self.right_tabs.tabText(self.right_tabs.currentIndex())
        if tab_title == "Depth Profile":
            return "profile"
        if tab_title == "Depth Map":
            return "depth_map"
        return None

    def _on_toolbar_calipers_toggled(self, checked: bool) -> None:
        if checked:
            self.start_profile_calipers()
        else:
            if self.caliper_mode:
                self.caliper_mode = False
                self.caliper_points = []
                self.caliper_label.setText("Calipers: off")
                self.status_label.setText("Calipers disabled.")

    def _set_caliper_toggle(self, checked: bool) -> None:
        self.caliper_toggle_action.blockSignals(True)
        self.caliper_toggle_action.setChecked(checked)
        self.caliper_toggle_action.blockSignals(False)

    def autofill_manual_positions(self) -> None:
        if self.manual_positions_input is None:
            return
        try:
            frame_count = len(self.sequence_paths)
            if frame_count == 0:
                self.manual_positions_input.setText("")
                self.sequence_spacing_label.setText("Sequence spacing: load sequence first")
                self.status_label.setText("No sequence loaded for auto-fill.")
                return
            step = self._f("frame_step_mm")
            if step <= 0:
                raise ValueError("frame_step_mm must be > 0")
            positions = [i * step for i in range(frame_count)]
            self.manual_positions_input.setText(", ".join(f"{v:.4f}" for v in positions))
            self.sequence_spacing_label.setText(
                f"Sequence spacing: manual positions auto-filled ({frame_count} frames)"
            )
            self.status_label.setText("Manual frame positions auto-filled from frame_step_mm.")
        except Exception as e:
            QMessageBox.warning(self, "Auto-fill Error", str(e))

    def apply_filters(self) -> None:
        try:
            raw_profile_depths = self._active_profile_depths()
            profile_y_mm = self._profile_frame_position_mm()
            limits = self._normalized_roi_limits()
            if not self._has_active_limits(limits):
                self.display_depths = raw_profile_depths
                self.filtered_cloud = self.cloud
                self.filtered_layered_cloud = self.layered_cloud
                self.filtered_depth_map = self.depth_map
                self._update_plots(self._active_raw_cloud())
                self._update_depth_map_plot()
                self.filter_status_label.setText("Filter: inactive")
                self.status_label.setText("ROI/Z filters inactive.")
                return
            self.display_depths = self._apply_profile_roi_filter(raw_profile_depths, profile_y_mm)
            self.filtered_cloud = self._apply_cloud_roi_filter(self.cloud)
            self.filtered_layered_cloud = self._apply_cloud_roi_filter(self.layered_cloud)
            self.filtered_depth_map = self._filter_depth_map()
            raw_active = self._active_raw_cloud()
            filtered_active = self._active_filtered_cloud()
            self._update_plots(filtered_active)
            self._update_depth_map_plot()
            status_parts: list[str] = []
            if raw_active.shape[0] > 0:
                status_parts.append(
                    f"{filtered_active.shape[0]}/{raw_active.shape[0]} cloud points"
                )
            if self.depth_map is not None and self.depth_map.size > 0:
                raw_px = int(np.isfinite(self.depth_map).sum())
                filt_map = self.filtered_depth_map if self.filtered_depth_map is not None else self.depth_map
                filt_px = int(np.isfinite(filt_map).sum())
                status_parts.append(f"depth map {filt_px}/{raw_px} px")
            if status_parts:
                self.filter_status_label.setText("Filter active: " + ", ".join(status_parts))
            else:
                self.filter_status_label.setText("Filter active")
            # Do NOT silently revert to the unfiltered data if everything is masked;
            # that hides the filter's effect and makes "Apply" look broken.
            if (
                self.depth_map is not None
                and self.depth_map.size > 0
                and self.filtered_depth_map is not None
                and np.isfinite(self.filtered_depth_map).sum() == 0
            ):
                self.status_label.setText(
                    "ROI/Z filters applied — all depth-map pixels removed. Loosen the limits to see data."
                )
            else:
                self.status_label.setText("ROI/Z filters applied.")
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", str(e))

    def clear_filters(self) -> None:
        for entry in self.roi_inputs.values():
            entry.clear()
        self.display_depths = self._active_profile_depths()
        self.filtered_cloud = self.cloud
        self.filtered_layered_cloud = self.layered_cloud
        self.filtered_depth_map = self.depth_map
        self._update_plots(self._active_raw_cloud())
        self._update_depth_map_plot()
        self.filter_status_label.setText("Filter: inactive")
        self.status_label.setText("ROI/Z filters cleared.")

    def _process_depth(self, bgr: np.ndarray):
        cents, mask = self._extract(bgr)
        if self.calibration:
            # Keep raw depth basis aligned with active geometric parameters.
            raw = centroids_to_depth(
                cents,
                self._f("laser_angle_deg"),
                self._f("mm_per_pixel"),
                self._f("zero_row"),
            )
            depths = apply_calibration(raw, self.calibration)
        else:
            raw = centroids_to_depth(cents, self._f("laser_angle_deg"), self._f("mm_per_pixel"), self._f("zero_row"))
            depths = raw
        return cents, mask, raw, depths

    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
        if not path:
            return
        try:
            self.bgr = read_image(path)
            self.image_path = path
            self.path_label.setText(path)
            if self._f("zero_row") <= 0:
                self.inputs["zero_row"].setText(f"{self.bgr.shape[0] * 0.5:.3f}")
            self.status_label.setText("Image loaded. Click Process.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def process_current(self) -> None:
        if self.bgr is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        try:
            cents, mask, raw, depths = self._process_depth(self.bgr)
            overlay = draw_overlay(self.bgr, cents)
            self.raw_depths = raw
            self.depths = depths
            self.cloud = build_single_line_cloud(depths, self._f("mm_per_pixel"))
            self.profile_source_combo.setCurrentText("Current image")
            self._update_preview(mask, overlay)
            self.apply_filters()
            self.status_label.setText(f"Processed: valid {int(np.isfinite(depths).sum())}/{len(depths)}, points {self.cloud.shape[0]}")
        except Exception as e:
            QMessageBox.critical(self, "Process Error", str(e))

    def load_sequence_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select sequence images", "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
        if not paths:
            return
        self.sequence_paths = sorted(paths)
        self.sequence_depth_rows = []
        self.sequence_frame_positions_mm = np.array([], dtype=np.float32)
        self._sync_sequence_profile_controls()
        self.seq_label.setText(f"Sequence loaded: {len(self.sequence_paths)} images")
        self.autofill_manual_positions()
        self.status_label.setText("Sequence loaded. Click Build Layered Cloud.")

    def build_layered_cloud(self) -> None:
        if not self.sequence_paths:
            QMessageBox.warning(self, "No Sequence", "Load sequence images first.")
            return
        # Ensure sequence processing always uses calibration geometric metadata when available.
        if self.calibration_meta:
            if "mm_per_pixel" in self.calibration_meta:
                self.inputs["mm_per_pixel"].setText(f"{float(self.calibration_meta['mm_per_pixel']):.6f}")
            if "zero_row" in self.calibration_meta:
                self.inputs["zero_row"].setText(f"{float(self.calibration_meta['zero_row']):.3f}")
            if "laser_angle_deg" in self.calibration_meta:
                self.inputs["laser_angle_deg"].setText(f"{float(self.calibration_meta['laser_angle_deg']):.3f}")
        depth_rows = []
        valid_indices: list[int] = []
        bad = 0
        zero_row_auto_set = False
        for idx, p in enumerate(self.sequence_paths):
            try:
                bgr = read_image(p)
                # Match single-image behavior: if zero_row is unset, initialize from first valid frame.
                if not zero_row_auto_set and self._f("zero_row") <= 0:
                    self.inputs["zero_row"].setText(f"{bgr.shape[0] * 0.5:.3f}")
                    zero_row_auto_set = True
                _, _, _, d = self._process_depth(bgr)
                depth_rows.append(d)
                valid_indices.append(idx)
            except Exception:
                bad += 1
        if not depth_rows:
            QMessageBox.warning(self, "No Frames", "No readable/processable frames.")
            return
        mode = self.sequence_spacing_mode.currentText()
        output_mode = self.sequence_output_mode.currentText()
        if mode == "Manual frame positions (mm)":
            all_positions = self._parse_manual_positions(len(self.sequence_paths))
            positions = all_positions[np.array(valid_indices, dtype=np.int32)]
            if positions.shape[0] > 0:
                positions = positions - float(positions[0])
            if positions.shape[0] > 1:
                steps = np.diff(positions)
                self.sequence_spacing_label.setText(
                    "Sequence spacing: manual "
                    f"(min step={float(np.min(steps)):.4f} mm, max step={float(np.max(steps)):.4f} mm)"
                )
            else:
                self.sequence_spacing_label.setText("Sequence spacing: manual (single frame)")
        else:
            self.layered_cloud = build_layered_cloud_from_depth_rows(
                depth_rows, self._f("mm_per_pixel"), self._f("frame_step_mm"), self.scan_axis.currentText()
            )
            positions = np.array(
                [i * self._f("frame_step_mm") for i in range(len(depth_rows))],
                dtype=np.float32,
            )
            self.sequence_spacing_label.setText(
                f"Sequence spacing: uniform step {self._f('frame_step_mm'):.4f} mm"
            )
        if output_mode == "Depth map only (fast)":
            # Fast path: skip cloud construction + 3D population for lower latency.
            self.layered_cloud = np.zeros((0, 3), dtype=np.float32)
        else:
            self.layered_cloud = self._build_layered_cloud_with_positions(
                depth_rows,
                self._f("mm_per_pixel"),
                positions,
                self.scan_axis.currentText(),
            )
        self.sequence_depth_rows = depth_rows
        self.sequence_frame_positions_mm = positions
        self._sync_sequence_profile_controls()
        self.depth_map = self._build_depth_map(depth_rows)
        self.apply_filters()
        if output_mode == "Depth map only (fast)":
            self.status_label.setText(
                f"Depth map built (fast mode): frames={len(depth_rows)}, unreadable={bad}, "
                "3D cloud skipped"
            )
        else:
            self.status_label.setText(
                f"Layered cloud + depth map: frames={len(depth_rows)}, unreadable={bad}, "
                f"points={self.layered_cloud.shape[0]}"
            )

    def save_depth_map_image(self) -> None:
        if self.depth_map is None or self.depth_map.size == 0:
            QMessageBox.warning(self, "No Depth Map", "Build layered cloud first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save depth map image", "depth_map.png", "PNG (*.png)")
        if not out:
            return
        self.depth_map_fig.savefig(Path(out), dpi=150, bbox_inches="tight")
        self.status_label.setText(f"Saved depth map image: {out}")

    def save_depth_map_csv(self) -> None:
        if self.depth_map is None or self.depth_map.size == 0:
            QMessageBox.warning(self, "No Depth Map", "Build layered cloud first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save depth map CSV", "depth_map.csv", "CSV (*.csv)")
        if not out:
            return
        depth_map = self.filtered_depth_map if self.filtered_depth_map is not None else self.depth_map
        depth_map = self._ground_depth_map_for_display(depth_map)
        np.savetxt(Path(out), depth_map, delimiter=",")
        meta = Path(out).with_suffix(".meta.csv")
        y_mm = self.sequence_frame_positions_mm.astype(np.float32, copy=False)
        rows = np.column_stack([np.arange(y_mm.shape[0], dtype=np.int32), y_mm])
        np.savetxt(meta, rows, delimiter=",", header="frame_index,scan_position_mm", comments="")
        self.status_label.setText(f"Saved depth map CSV: {out} (+ {meta.name})")

    def load_cal_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration", "", "JSON (*.json)")
        if not path:
            return
        payload = load_calibration(path)
        self.calibration = payload.get("model")
        if not self.calibration:
            QMessageBox.warning(self, "Invalid Calibration", "Selected file has no calibration model.")
            return
        meta = payload.get("meta", {})
        self.calibration_meta = dict(meta) if isinstance(meta, dict) else {}
        if "mm_per_pixel" in meta:
            self.inputs["mm_per_pixel"].setText(f"{float(meta['mm_per_pixel']):.6f}")
        if "zero_row" in meta:
            self.inputs["zero_row"].setText(f"{float(meta['zero_row']):.3f}")
        if "laser_angle_deg" in meta:
            self.inputs["laser_angle_deg"].setText(f"{float(meta['laser_angle_deg']):.3f}")
        self.calibration_label.setText(f"Calibration: loaded {Path(path).name}")
        self.status_label.setText("Calibration loaded and applied to all processing.")

    def save_cloud_csv(self) -> None:
        if self.cloud is None or self.cloud.size == 0:
            QMessageBox.warning(self, "No Cloud", "Process image first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save cloud CSV", "cloud.csv", "CSV (*.csv)")
        if out:
            cloud_to_save = self.filtered_cloud if self.filtered_cloud is not None else self.cloud
            cloud_to_save = self._ground_cloud_for_display(cloud_to_save)
            np.savetxt(Path(out), cloud_to_save, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
            self.status_label.setText(f"Saved cloud CSV (filtered view): {out}")

    def save_layered_cloud_csv(self) -> None:
        if self.layered_cloud is None or self.layered_cloud.size == 0:
            QMessageBox.warning(self, "No Layered Cloud", "Build layered cloud first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save layered cloud CSV", "layered_cloud.csv", "CSV (*.csv)")
        if out:
            cloud_to_save = self.filtered_layered_cloud if self.filtered_layered_cloud is not None else self.layered_cloud
            cloud_to_save = self._ground_cloud_for_display(cloud_to_save)
            np.savetxt(Path(out), cloud_to_save, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
            self.status_label.setText(f"Saved layered cloud CSV (filtered view): {out}")

    def open_calibration_window(self) -> None:
        if self.calibration_window is None:
            self.calibration_window = CalibrationWindow()
            self.calibration_window.calibration_saved.connect(self._apply_calibration_payload)
        self.calibration_window.show()
        self.calibration_window.raise_()
        self.calibration_window.activateWindow()

    def _apply_calibration_payload(self, payload: dict) -> None:
        model = payload.get("model")
        if not model:
            return
        self.calibration = model
        meta = payload.get("meta", {})
        self.calibration_meta = dict(meta) if isinstance(meta, dict) else {}
        if "mm_per_pixel" in meta:
            self.inputs["mm_per_pixel"].setText(f"{float(meta['mm_per_pixel']):.6f}")
        if "zero_row" in meta:
            self.inputs["zero_row"].setText(f"{float(meta['zero_row']):.3f}")
        if "laser_angle_deg" in meta:
            self.inputs["laser_angle_deg"].setText(f"{float(meta['laser_angle_deg']):.3f}")
        samples = payload.get("samples", [])
        sample_count = len(samples) if isinstance(samples, list) else 0
        self.calibration_label.setText("Calibration: loaded from calibration window")
        self.status_label.setText(
            f"Calibration imported from calibration window: samples={sample_count}"
        )

    def _update_preview(self, mask: np.ndarray, overlay_bgr: np.ndarray) -> None:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        merged = np.hstack([mask_rgb, overlay_bgr])
        merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
        self.ax_image.clear()
        self.ax_image.imshow(merged)
        self.ax_image.set_title("Mask + Overlay")
        self.ax_image.axis("off")
        self.image_canvas.draw_idle()

    def _update_plots(self, cloud: np.ndarray) -> None:
        self.ax_profile.clear()
        if self.profile_source_combo.currentText() == "Sequence frame":
            self.ax_profile.set_title(
                f"Depth Profile (Sequence frame {self.sequence_profile_spin.value()}, Display-Scaled)"
            )
        else:
            self.ax_profile.set_title("Depth Profile (Current Image, Display-Scaled)")
        self.ax_profile.set_xlabel("Column")
        self.ax_profile.set_ylabel("Z (mm, relative)" if self._z_grounding_enabled() else "Z (mm)")
        try:
            visual_scale = float(self.profile_visual_scale.text().strip())
        except Exception:
            visual_scale = 1.0
        if visual_scale <= 0:
            visual_scale = 1.0
        profile_depths = self.display_depths if self.display_depths is not None else self.depths
        profile_depths = self._ground_depths_for_display(profile_depths)
        if profile_depths is not None:
            cols = np.arange(profile_depths.shape[0], dtype=np.int32)
            self.ax_profile.plot(cols, profile_depths, linewidth=1.0, color="tab:blue")
            self.ax_profile.grid(True, alpha=0.3)
            # Display-only vertical exaggeration; underlying values remain unchanged.
            self.ax_profile.set_aspect(1.0 / visual_scale, adjustable="datalim")
        if self.caliper_line is not None:
            (p1, p2) = self.caliper_line
            self.ax_profile.plot([p1[0], p2[0]], [p1[1], p2[1]], color="orange", linewidth=2.0)
            self.ax_profile.scatter([p1[0], p2[0]], [p1[1], p2[1]], color="orange", s=24)
        self.profile_canvas.draw_idle()

        self.ax3d.clear()
        self.ax3d.set_title("Point Cloud")
        self.ax3d.set_xlabel("X (mm)")
        self.ax3d.set_ylabel("Y (mm)")
        self.ax3d.set_zlabel(
            "Z (mm, display-scaled, relative)" if self._z_grounding_enabled() else "Z (mm, display-scaled)"
        )
        self.ax3d.view_init(elev=20, azim=-60)
        try:
            cloud_z_scale = float(self.cloud_visual_z_scale.text().strip())
        except Exception:
            cloud_z_scale = 1.0
        if cloud_z_scale <= 0:
            cloud_z_scale = 1.0
        try:
            point_size = float(self.cloud_point_size.text().strip())
        except Exception:
            point_size = 3.0
        if point_size <= 0:
            point_size = 3.0
        # Display-only shape control: this changes perceived vertical exaggeration.
        self.ax3d.set_box_aspect((1.0, 1.0, cloud_z_scale))
        off = self._get_z_ground_offset_mm()
        self.z_ground_offset_mm = off
        if self._z_grounding_enabled():
            self.z_ground_status_label.setText(f"Z offset (subtracted): {off:.4f} mm → lowest point at 0")
        else:
            self.z_ground_status_label.setText("Z offset: 0.000 mm (absolute sensor/calib)")
        cloud_plot = self._ground_cloud_for_display(cloud)
        if cloud_plot.size > 0:
            z_disp = cloud_plot[:, 2] * cloud_z_scale
            # Depth-colored cloud makes the 3D form easier to read visually.
            self.ax3d.scatter(
                cloud_plot[:, 0],
                cloud_plot[:, 1],
                z_disp,
                c=cloud_plot[:, 2],
                cmap="viridis",
                s=point_size,
                alpha=0.9,
            )
        self.cloud_canvas.draw_idle()

    def _on_profile_hover(self, event) -> None:
        if event.inaxes != self.ax_profile or self.display_depths is None or event.xdata is None:
            self.depth_hover_label.setText("Depth @ cursor: -")
            return
        idx = int(round(event.xdata))
        if idx < 0 or idx >= len(self.display_depths):
            self.depth_hover_label.setText("Depth @ cursor: -")
            return
        gd = self._ground_depths_for_display(self.display_depths)
        z = float(gd[idx]) if gd is not None else float("nan")
        if not np.isfinite(z):
            self.depth_hover_label.setText(f"Depth @ cursor: col={idx}, z=nan")
            return
        rel = " (relative)" if self._z_grounding_enabled() else ""
        self.depth_hover_label.setText(f"Depth @ cursor: col={idx}, z={z:.4f} mm{rel}")

    def start_profile_calipers(self) -> None:
        target = self._active_measurement_target()
        if target == "profile":
            if self.display_depths is None:
                QMessageBox.warning(self, "No Profile", "Process an image first.")
                self._set_caliper_toggle(False)
                return
            self.caliper_target = "profile"
            self.caliper_mode = True
            self.caliper_points = []
            self.caliper_label.setText("Calipers: pick 2 points on Depth Profile")
            self.status_label.setText("Calipers active on Depth Profile. Click 2 points.")
            self._set_caliper_toggle(True)
            return
        if target == "depth_map":
            depth_map = self.filtered_depth_map if self.filtered_depth_map is not None else self.depth_map
            if depth_map is None or depth_map.size == 0:
                QMessageBox.warning(self, "No Depth Map", "Build layered cloud first.")
                self._set_caliper_toggle(False)
                return
            self.caliper_target = "depth_map"
            self.caliper_mode = True
            self.caliper_points = []
            self.caliper_label.setText("Calipers: pick 2 points on Depth Map")
            self.status_label.setText("Calipers active on Depth Map. Click 2 points.")
            self._set_caliper_toggle(True)
            return
        QMessageBox.information(
            self,
            "Calipers",
            "Open Depth Profile or Depth Map tab to use calipers.",
        )
        self._set_caliper_toggle(False)

    def clear_profile_calipers(self) -> None:
        self.caliper_mode = False
        self.caliper_points = []
        self.caliper_line = None
        self.depth_map_caliper_line = None
        self.caliper_label.setText("Calipers: off")
        self._set_caliper_toggle(False)
        cloud = self._active_filtered_cloud()
        self._update_plots(cloud)
        self._update_depth_map_plot()

    def _on_profile_click(self, event) -> None:
        if not self.caliper_mode or self.caliper_target != "profile":
            return
        if self.display_depths is None or event.inaxes != self.ax_profile or event.xdata is None or event.ydata is None:
            return
        self.caliper_points.append((float(event.xdata), float(event.ydata)))
        if len(self.caliper_points) < 2:
            self.caliper_label.setText("Calipers: point 1 set, pick point 2")
            return
        p1, p2 = self.caliper_points[0], self.caliper_points[1]
        self.caliper_line = (p1, p2)
        dx_cols = abs(p2[0] - p1[0])
        dz_mm = abs(p2[1] - p1[1])
        p1_col = int(round(p1[0]))
        p2_col = int(round(p2[0]))
        p1_col = max(0, min(p1_col, len(self.display_depths) - 1))
        p2_col = max(0, min(p2_col, len(self.display_depths) - 1))
        gd = self._ground_depths_for_display(self.display_depths)
        z1 = float(gd[p1_col]) if gd is not None else float("nan")
        z2 = float(gd[p2_col]) if gd is not None else float("nan")
        dx_mm = dx_cols * self._f("mm_per_pixel")
        p2p_mm = float(np.hypot(dx_mm, dz_mm))
        self.caliper_label.setText(
            f"P2P calibrated distance: {p2p_mm:.4f} mm (P1 z={z1:.4f}, P2 z={z2:.4f})"
        )
        self.caliper_mode = False
        self.caliper_points = []
        self._set_caliper_toggle(False)
        cloud = self._active_filtered_cloud()
        self._update_plots(cloud)

    def _on_depth_map_click(self, event) -> None:
        if not self.caliper_mode or self.caliper_target != "depth_map":
            return
        depth_map = self.filtered_depth_map if self.filtered_depth_map is not None else self.depth_map
        if depth_map is None or event.inaxes != self.ax_depth_map or event.xdata is None or event.ydata is None:
            return
        self.caliper_points.append((float(event.xdata), float(event.ydata)))
        if len(self.caliper_points) < 2:
            self.caliper_label.setText("Calipers: point 1 set on Depth Map, pick point 2")
            return
        p1, p2 = self.caliper_points[0], self.caliper_points[1]
        self.depth_map_caliper_line = (p1, p2)
        x1_mm, y1_mm = float(p1[0]), float(p1[1])
        x2_mm, y2_mm = float(p2[0]), float(p2[1])
        dx_mm = x2_mm - x1_mm
        dy_mm = y2_mm - y1_mm
        mm_per_pixel = self._f("mm_per_pixel")
        col1 = int(round(x1_mm / mm_per_pixel))
        col2 = int(round(x2_mm / mm_per_pixel))
        row1 = int(np.argmin(np.abs(self.sequence_frame_positions_mm - y1_mm)))
        row2 = int(np.argmin(np.abs(self.sequence_frame_positions_mm - y2_mm)))
        col1 = max(0, min(col1, depth_map.shape[1] - 1))
        col2 = max(0, min(col2, depth_map.shape[1] - 1))
        off = self._get_z_ground_offset_mm()
        z1 = float(depth_map[row1, col1])
        z2 = float(depth_map[row2, col2])
        if self._z_grounding_enabled():
            if np.isfinite(z1):
                z1 -= off
            if np.isfinite(z2):
                z2 -= off
        if np.isfinite(z1) and np.isfinite(z2):
            dz_mm = z2 - z1
            p2p_mm = float(np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm))
            self.caliper_label.setText(
                f"Depth-map P2P: {p2p_mm:.4f} mm (dx={abs(dx_mm):.4f}, dy={abs(dy_mm):.4f}, dz={abs(dz_mm):.4f})"
            )
        else:
            p2p_mm = float(np.hypot(dx_mm, dy_mm))
            self.caliper_label.setText(
                f"Depth-map XY distance: {p2p_mm:.4f} mm (one/both Z values are invalid)"
            )
        self.caliper_mode = False
        self.caliper_points = []
        self._set_caliper_toggle(False)
        self._update_depth_map_plot()

    def refresh_profile_view(self) -> None:
        cloud = self._active_filtered_cloud()
        self._update_plots(cloud)
        self.status_label.setText("Profile/3D view scale updated (display only).")


def main() -> None:
    app = QApplication(sys.argv)
    win = LaserPrototypeQt()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
