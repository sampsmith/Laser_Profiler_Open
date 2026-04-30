from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import (
    centroids_to_depth,
    draw_overlay,
    extract_green_centroids,
    load_calibration,
    read_image,
    save_calibration,
)


class CalibrationWindow(QMainWindow):
    calibration_saved = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Laser Calibration Window")
        self.resize(1400, 820)

        self.bgr = None
        self.depths = None
        self.raw_depths = None
        self.overlay_rgb = None
        self.samples: list[dict] = []
        self.model = None

        self.pick_mode: str | None = None  # width|height
        self.pick_points: list[tuple[float, float]] = []
        self.pick_line = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QWidget()
        left_l = QVBoxLayout(left)
        layout.addWidget(left, 0)

        right = QWidget()
        right_l = QVBoxLayout(right)
        layout.addWidget(right, 1)

        for txt, cb in [
            ("Load Known Image", self.load_image),
            ("Process Image", self.process_image),
            ("Set Zero From Current Image", self.set_zero),
            ("Pick Width On Profile (2 clicks)", self.start_pick_width),
            ("Pick Height On Profile (2 clicks)", self.start_pick_height),
            ("Build Simple Calibration", self.fit_calibration),
            ("Load Calibration JSON", self.load_calibration_json),
            ("Save Calibration JSON", self.save_calibration_json),
            ("Clear Picks", self.clear_picks),
        ]:
            b = QPushButton(txt)
            b.clicked.connect(cb)
            left_l.addWidget(b)

        form_box = QGroupBox("Calibration Inputs")
        form = QFormLayout(form_box)
        self.in_h_min = QLineEdit("50")
        self.in_h_max = QLineEdit("80")
        self.in_s_min = QLineEdit("80")
        self.in_s_max = QLineEdit("255")
        self.in_v_min = QLineEdit("80")
        self.in_v_max = QLineEdit("255")
        self.in_blur = QLineEdit("5")
        self.in_angle = QLineEdit("30")
        self.in_mpp = QLineEdit("0.1")
        self.in_zero = QLineEdit("0.0")
        self.in_known_width = QLineEdit("10.0")
        self.in_known_height = QLineEdit("5.0")
        for k, w in [
            ("h_min", self.in_h_min), ("h_max", self.in_h_max),
            ("s_min", self.in_s_min), ("s_max", self.in_s_max),
            ("v_min", self.in_v_min), ("v_max", self.in_v_max),
            ("blur_kernel", self.in_blur),
            ("laser_angle_deg", self.in_angle),
            ("mm_per_pixel", self.in_mpp),
            ("zero_row", self.in_zero),
            ("known_width_mm", self.in_known_width),
            ("known_height_mm", self.in_known_height),
        ]:
            form.addRow(k, w)
        left_l.addWidget(form_box)

        self.status = QLabel("Load known image to start calibration.")
        self.status.setWordWrap(True)
        left_l.addWidget(self.status)
        left_l.addStretch(1)

        self.fig_overlay = Figure(figsize=(7, 3.5))
        self.ax_overlay = self.fig_overlay.add_subplot(111)
        self.canvas_overlay = FigureCanvas(self.fig_overlay)
        right_l.addWidget(self.canvas_overlay, 1)

        self.fig_profile = Figure(figsize=(7, 3.5))
        self.ax_profile = self.fig_profile.add_subplot(111)
        self.canvas_profile = FigureCanvas(self.fig_profile)
        self.canvas_profile.mpl_connect("button_press_event", self._on_profile_click)
        right_l.addWidget(self.canvas_profile, 1)

    def _i(self, w: QLineEdit) -> int:
        return int(float(w.text().strip()))

    def _f(self, w: QLineEdit) -> float:
        return float(w.text().strip())

    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load image", "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
        if not path:
            return
        try:
            self.bgr = read_image(path)
            if self._f(self.in_zero) <= 0:
                self.in_zero.setText(f"{self.bgr.shape[0] * 0.5:.3f}")
            self.status.setText(f"Loaded: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _extract(self):
        return extract_green_centroids(
            self.bgr,
            self._i(self.in_h_min), self._i(self.in_h_max),
            self._i(self.in_s_min), self._i(self.in_s_max),
            self._i(self.in_v_min), self._i(self.in_v_max),
            self._i(self.in_blur),
        )

    def process_image(self) -> None:
        if self.bgr is None:
            QMessageBox.warning(self, "No Image", "Load image first.")
            return
        cents, mask = self._extract()
        # Z calibration uses raw depth independent of width (X) pixel scale.
        self.raw_depths = centroids_to_depth(cents, self._f(self.in_angle), 1.0, self._f(self.in_zero))
        self.depths = self.raw_depths.copy()
        overlay = draw_overlay(self.bgr, cents)
        self.overlay_rgb = cv2.cvtColor(np.hstack([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay]), cv2.COLOR_BGR2RGB)
        self._redraw()
        self.status.setText(
            f"Processed. Valid profile points: {int(np.isfinite(self.depths).sum())}/{len(self.depths)}. "
            "Width and Z calibration are decoupled."
        )

    def set_zero(self) -> None:
        if self.bgr is None:
            return
        cents, _ = self._extract()
        valid = cents[np.isfinite(cents)]
        if valid.size == 0:
            QMessageBox.warning(self, "No Line", "No valid line detected.")
            return
        zr = float(np.median(valid))
        self.in_zero.setText(f"{zr:.3f}")
        self.status.setText(f"Zero set from current image: {zr:.3f}")

    def start_pick_width(self) -> None:
        self.pick_mode = "width"
        self.pick_points = []
        self.status.setText("Click 2 points on profile for KNOWN WIDTH.")

    def start_pick_height(self) -> None:
        self.pick_mode = "height"
        self.pick_points = []
        self.status.setText("Click 2 points on profile for KNOWN HEIGHT.")

    def clear_picks(self) -> None:
        self.pick_mode = None
        self.pick_points = []
        self.pick_line = None
        self._redraw()

    def _on_profile_click(self, event) -> None:
        if self.pick_mode is None or self.depths is None:
            return
        if event.inaxes != self.ax_profile or event.xdata is None or event.ydata is None:
            return
        self.pick_points.append((float(event.xdata), float(event.ydata)))
        if len(self.pick_points) < 2:
            return
        p1, p2 = self.pick_points[0], self.pick_points[1]
        self.pick_line = (p1, p2, "orange" if self.pick_mode == "height" else "cyan")

        try:
            if self.pick_mode == "width":
                col_span = abs(p2[0] - p1[0])
                if col_span < 1e-6:
                    raise ValueError("Width picks too close.")
                mpp = self._f(self.in_known_width) / col_span
                self.in_mpp.setText(f"{mpp:.6f}")
                self.status.setText(f"Width calibrated: mm_per_pixel={mpp:.6f}")
            else:
                raw_h = abs(p2[1] - p1[1])
                if raw_h < 1e-9:
                    raise ValueError("Height picks too close.")
                known_h = self._f(self.in_known_height)
                self.samples.append({"raw_z_mm": raw_h, "true_z_mm": known_h})
                self.status.setText(f"Height sample added: raw={raw_h:.4f}, true={known_h:.4f}, samples={len(self.samples)}")
        except Exception as e:
            QMessageBox.critical(self, "Pick Error", str(e))

        self.pick_mode = None
        self.pick_points = []
        self._redraw()

    def fit_calibration(self) -> None:
        if len(self.samples) < 1:
            QMessageBox.warning(self, "Need Sample", "Add at least 1 height sample.")
            return
        raw = np.array([float(s["raw_z_mm"]) for s in self.samples], dtype=np.float64)
        true = np.array([float(s["true_z_mm"]) for s in self.samples], dtype=np.float64)
        valid = np.abs(raw) > 1e-9
        if not np.any(valid):
            QMessageBox.warning(self, "Invalid Samples", "Picked raw heights are too small.")
            return
        ratios = true[valid] / raw[valid]
        a = float(np.median(ratios))
        b = 0.0
        pred = a * raw
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        self.model = {"type": "linear", "a": a, "b": b, "rmse_mm": rmse}
        self.status.setText(
            f"Simple calibration built: z_true={self.model['a']:.6f}*z_raw (b=0), rmse={self.model['rmse_mm']:.4f} mm"
        )

    def save_calibration_json(self) -> None:
        if not self.model:
            QMessageBox.warning(self, "No Calibration", "Build or load calibration first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save calibration", "calibration.json", "JSON (*.json)")
        if not out:
            return
        payload = {
            "model": self.model,
            "samples": self.samples,
            "meta": {
                "mm_per_pixel": self._f(self.in_mpp),
                "zero_row": self._f(self.in_zero),
                "laser_angle_deg": self._f(self.in_angle),
            },
        }
        save_calibration(out, payload)
        self.calibration_saved.emit(payload)
        self.status.setText(f"Saved calibration: {out}")

    def load_calibration_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration", "", "JSON (*.json)")
        if not path:
            return
        payload = load_calibration(path)
        model = payload.get("model")
        if not model:
            QMessageBox.warning(self, "Invalid Calibration", "Selected file has no calibration model.")
            return
        self.model = model
        self.samples = list(payload.get("samples", []))
        meta = payload.get("meta", {})
        if "mm_per_pixel" in meta:
            self.in_mpp.setText(f"{float(meta['mm_per_pixel']):.6f}")
        if "zero_row" in meta:
            self.in_zero.setText(f"{float(meta['zero_row']):.3f}")
        if "laser_angle_deg" in meta:
            self.in_angle.setText(f"{float(meta['laser_angle_deg']):.3f}")
        self.calibration_saved.emit(payload)
        self.status.setText(f"Loaded calibration: {Path(path).name}")

    def _redraw(self) -> None:
        self.ax_overlay.clear()
        if self.overlay_rgb is not None:
            self.ax_overlay.imshow(self.overlay_rgb)
        self.ax_overlay.set_title("Mask + Overlay")
        self.ax_overlay.axis("off")
        self.canvas_overlay.draw_idle()

        self.ax_profile.clear()
        self.ax_profile.set_title("Depth Profile")
        self.ax_profile.set_xlabel("Column")
        self.ax_profile.set_ylabel("Z (mm)")
        if self.depths is not None:
            cols = np.arange(self.depths.shape[0], dtype=np.int32)
            self.ax_profile.plot(cols, self.depths, linewidth=1.0)
            self.ax_profile.grid(True, alpha=0.3)
        if self.pick_line is not None:
            p1, p2, c = self.pick_line
            self.ax_profile.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, linewidth=2.0)
            self.ax_profile.scatter([p1[0], p2[0]], [p1[1], p2[1]], color=c, s=24)
        self.canvas_profile.draw_idle()


def main() -> None:
    app = QApplication(sys.argv)
    w = CalibrationWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
