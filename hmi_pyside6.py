from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
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
        self.display_depths = None
        self.filtered_cloud = None
        self.filtered_layered_cloud = None
        self.sequence_paths: list[str] = []
        self.calibration = None
        self.caliper_mode = False
        self.caliper_points: list[tuple[float, float]] = []
        self.caliper_line: tuple[tuple[float, float], tuple[float, float]] | None = None

        self.inputs: dict[str, QLineEdit] = {}
        self.calibration_window: CalibrationWindow | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        # Left: scrollable controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        left_scroll.setWidget(left_panel)
        left_layout = QVBoxLayout(left_panel)
        root_layout.addWidget(left_scroll, 0)

        # Right: tabbed views
        right_tabs = QTabWidget()
        root_layout.addWidget(right_tabs, 1)

        self.image_fig = Figure(figsize=(7, 5))
        self.ax_image = self.image_fig.add_subplot(111)
        self.image_canvas = FigureCanvas(self.image_fig)
        right_tabs.addTab(self.image_canvas, "Mask + Overlay")

        self.profile_fig = Figure(figsize=(7, 5))
        self.ax_profile = self.profile_fig.add_subplot(111)
        self.profile_canvas = FigureCanvas(self.profile_fig)
        self.profile_canvas.mpl_connect("motion_notify_event", self._on_profile_hover)
        self.profile_canvas.mpl_connect("button_press_event", self._on_profile_click)
        right_tabs.addTab(self.profile_canvas, "Depth Profile")

        self.cloud_fig = Figure(figsize=(7, 5))
        self.ax3d = self.cloud_fig.add_subplot(111, projection="3d")
        self.cloud_canvas = FigureCanvas(self.cloud_fig)
        right_tabs.addTab(self.cloud_canvas, "3D Cloud")

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
        self.profile_visual_scale = QLineEdit("1.0")
        self.apply_profile_view_btn = QPushButton("Apply Profile View Scale")
        self.apply_profile_view_btn.clicked.connect(self.refresh_profile_view)
        self.cloud_visual_z_scale = QLineEdit("1.0")
        self.cloud_point_size = QLineEdit("3.0")
        self.apply_cloud_view_btn = QPushButton("Apply 3D Cloud Z Scale")
        self.apply_cloud_view_btn.clicked.connect(self.refresh_profile_view)
        self.roi_inputs: dict[str, QLineEdit] = {}
        self.manual_positions_input: QLineEdit | None = None
        self.filter_status_label = QLabel("Filter: inactive")
        self.sequence_spacing_label = QLabel("Sequence spacing: uniform step")
        self.apply_filter_btn = QPushButton("Apply ROI/Z Filter")
        self.apply_filter_btn.clicked.connect(self.apply_filters)
        self.clear_filter_btn = QPushButton("Clear ROI/Z Filter")
        self.clear_filter_btn.clicked.connect(self.clear_filters)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.path_label)
        status_layout.addWidget(self.seq_label)
        status_layout.addWidget(self.calibration_label)
        status_layout.addWidget(self.depth_hover_label)
        status_layout.addWidget(self.caliper_label)
        status_layout.addWidget(QLabel("Profile visual Z scale (display only)"))
        status_layout.addWidget(self.profile_visual_scale)
        status_layout.addWidget(self.apply_profile_view_btn)
        status_layout.addWidget(QLabel("3D cloud visual Z scale (display only)"))
        status_layout.addWidget(self.cloud_visual_z_scale)
        status_layout.addWidget(QLabel("3D cloud point size"))
        status_layout.addWidget(self.cloud_point_size)
        status_layout.addWidget(self.apply_cloud_view_btn)
        status_layout.addWidget(self.filter_status_label)
        status_layout.addWidget(self.apply_filter_btn)
        status_layout.addWidget(self.clear_filter_btn)
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

        filter_panel = QWidget()
        filter_form = QFormLayout(filter_panel)
        filter_defaults = {
            "x_min": "", "x_max": "",
            "y_min": "", "y_max": "",
            "z_min": "", "z_max": "",
        }
        for key, value in filter_defaults.items():
            entry = QLineEdit(value)
            entry.setPlaceholderText("disabled")
            self.roi_inputs[key] = entry
            filter_form.addRow(key, entry)
        left_layout.addWidget(CollapsibleSection("ROI / Z Filter (mm)", filter_panel, expanded=True))

        process_panel = QWidget()
        process_layout = QVBoxLayout(process_panel)
        process_layout.setContentsMargins(0, 0, 0, 0)
        self.sequence_spacing_mode = QComboBox()
        self.sequence_spacing_mode.addItems(["Uniform step (frame_step_mm)", "Manual frame positions (mm)"])
        process_layout.addWidget(QLabel("Sequence spacing mode"))
        process_layout.addWidget(self.sequence_spacing_mode)
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

    def _apply_cloud_roi_filter(self, cloud: np.ndarray) -> np.ndarray:
        if cloud is None or cloud.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        limits = self._roi_limits()
        mask = np.isfinite(cloud).all(axis=1)
        x = cloud[:, 0]
        y = cloud[:, 1]
        z = cloud[:, 2]
        if limits["x_min"] is not None:
            mask &= x >= float(limits["x_min"])
        if limits["x_max"] is not None:
            mask &= x <= float(limits["x_max"])
        if limits["y_min"] is not None:
            mask &= y >= float(limits["y_min"])
        if limits["y_max"] is not None:
            mask &= y <= float(limits["y_max"])
        if limits["z_min"] is not None:
            mask &= z >= float(limits["z_min"])
        if limits["z_max"] is not None:
            mask &= z <= float(limits["z_max"])
        return cloud[mask]

    def _apply_profile_roi_filter(self, depths: np.ndarray | None) -> np.ndarray | None:
        if depths is None:
            return None
        out = depths.astype(np.float64, copy=True)
        limits = self._roi_limits()
        mm_per_pixel = self._f("mm_per_pixel")
        cols_mm = np.arange(out.shape[0], dtype=np.float64) * mm_per_pixel
        if limits["x_min"] is not None:
            out[cols_mm < float(limits["x_min"])] = np.nan
        if limits["x_max"] is not None:
            out[cols_mm > float(limits["x_max"])] = np.nan
        if limits["z_min"] is not None:
            out[out < float(limits["z_min"])] = np.nan
        if limits["z_max"] is not None:
            out[out > float(limits["z_max"])] = np.nan
        y_min = limits["y_min"]
        y_max = limits["y_max"]
        if (y_min is not None and 0.0 < float(y_min)) or (y_max is not None and 0.0 > float(y_max)):
            out[:] = np.nan
        return out

    def _active_raw_cloud(self) -> np.ndarray:
        if self.layered_cloud is not None:
            return self.layered_cloud
        if self.cloud is not None:
            return self.cloud
        return np.zeros((0, 3), dtype=np.float32)

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
            self.display_depths = self._apply_profile_roi_filter(self.depths)
            self.filtered_cloud = self._apply_cloud_roi_filter(self.cloud)
            self.filtered_layered_cloud = self._apply_cloud_roi_filter(self.layered_cloud)
            raw_active = self._active_raw_cloud()
            filtered_active = self._active_filtered_cloud()
            self._update_plots(filtered_active)
            self.filter_status_label.setText(
                f"Filter active: {filtered_active.shape[0]}/{raw_active.shape[0]} points shown"
            )
            self.status_label.setText("ROI/Z filters applied.")
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", str(e))

    def clear_filters(self) -> None:
        for entry in self.roi_inputs.values():
            entry.clear()
        self.display_depths = self.depths
        self.filtered_cloud = self.cloud
        self.filtered_layered_cloud = self.layered_cloud
        self._update_plots(self._active_raw_cloud())
        self.filter_status_label.setText("Filter: inactive")
        self.status_label.setText("ROI/Z filters cleared.")

    def _process_depth(self, bgr: np.ndarray):
        cents, mask = self._extract(bgr)
        if self.calibration:
            # Decouple X-width scale from Z-depth calibration.
            raw = centroids_to_depth(cents, self._f("laser_angle_deg"), 1.0, self._f("zero_row"))
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
        self.seq_label.setText(f"Sequence loaded: {len(self.sequence_paths)} images")
        self.autofill_manual_positions()
        self.status_label.setText("Sequence loaded. Click Build Layered Cloud.")

    def build_layered_cloud(self) -> None:
        if not self.sequence_paths:
            QMessageBox.warning(self, "No Sequence", "Load sequence images first.")
            return
        depth_rows = []
        valid_indices: list[int] = []
        bad = 0
        for idx, p in enumerate(self.sequence_paths):
            try:
                bgr = read_image(p)
                _, _, _, d = self._process_depth(bgr)
                depth_rows.append(d)
                valid_indices.append(idx)
            except Exception:
                bad += 1
        if not depth_rows:
            QMessageBox.warning(self, "No Frames", "No readable/processable frames.")
            return
        mode = self.sequence_spacing_mode.currentText()
        if mode == "Manual frame positions (mm)":
            all_positions = self._parse_manual_positions(len(self.sequence_paths))
            positions = all_positions[np.array(valid_indices, dtype=np.int32)]
            self.layered_cloud = self._build_layered_cloud_with_positions(
                depth_rows,
                self._f("mm_per_pixel"),
                positions,
                self.scan_axis.currentText(),
            )
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
            self.sequence_spacing_label.setText(
                f"Sequence spacing: uniform step {self._f('frame_step_mm'):.4f} mm"
            )
        self.apply_filters()
        self.status_label.setText(f"Layered cloud: frames={len(depth_rows)}, unreadable={bad}, points={self.layered_cloud.shape[0]}")

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
            np.savetxt(Path(out), cloud_to_save, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
            self.status_label.setText(f"Saved cloud CSV (filtered view): {out}")

    def save_layered_cloud_csv(self) -> None:
        if self.layered_cloud is None or self.layered_cloud.size == 0:
            QMessageBox.warning(self, "No Layered Cloud", "Build layered cloud first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save layered cloud CSV", "layered_cloud.csv", "CSV (*.csv)")
        if out:
            cloud_to_save = self.filtered_layered_cloud if self.filtered_layered_cloud is not None else self.layered_cloud
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
        self.ax_profile.set_title("Depth Profile (Current Image, Display-Scaled)")
        self.ax_profile.set_xlabel("Column")
        self.ax_profile.set_ylabel("Z (mm)")
        try:
            visual_scale = float(self.profile_visual_scale.text().strip())
        except Exception:
            visual_scale = 1.0
        if visual_scale <= 0:
            visual_scale = 1.0
        profile_depths = self.display_depths if self.display_depths is not None else self.depths
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
        self.ax3d.set_zlabel("Z (mm, display-scaled)")
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
        if cloud.size > 0:
            z_disp = cloud[:, 2] * cloud_z_scale
            # Depth-colored cloud makes the 3D form easier to read visually.
            self.ax3d.scatter(
                cloud[:, 0],
                cloud[:, 1],
                z_disp,
                c=cloud[:, 2],
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
        z = float(self.display_depths[idx])
        if not np.isfinite(z):
            self.depth_hover_label.setText(f"Depth @ cursor: col={idx}, z=nan")
            return
        self.depth_hover_label.setText(f"Depth @ cursor: col={idx}, z={z:.4f} mm")

    def start_profile_calipers(self) -> None:
        if self.display_depths is None:
            QMessageBox.warning(self, "No Profile", "Process an image first.")
            return
        self.caliper_mode = True
        self.caliper_points = []
        self.caliper_label.setText("Calipers: pick 2 points on Depth Profile")
        self.status_label.setText("Calipers active. Click 2 points on the Depth Profile plot.")

    def clear_profile_calipers(self) -> None:
        self.caliper_mode = False
        self.caliper_points = []
        self.caliper_line = None
        self.caliper_label.setText("Calipers: off")
        cloud = self._active_filtered_cloud()
        self._update_plots(cloud)

    def _on_profile_click(self, event) -> None:
        if not self.caliper_mode:
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
        z1 = float(self.display_depths[p1_col])
        z2 = float(self.display_depths[p2_col])
        dx_mm = dx_cols * self._f("mm_per_pixel")
        p2p_mm = float(np.hypot(dx_mm, dz_mm))
        self.caliper_label.setText(
            f"P2P calibrated distance: {p2p_mm:.4f} mm (P1 z={z1:.4f}, P2 z={z2:.4f})"
        )
        self.caliper_mode = False
        self.caliper_points = []
        cloud = self._active_filtered_cloud()
        self._update_plots(cloud)

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
