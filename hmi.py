import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from app import (
    read_image,
    extract_green_centroids,
    centroids_to_depth,
    build_single_line_cloud,
    build_layered_cloud_from_depth_rows,
    draw_overlay,
    fit_linear_calibration,
    apply_calibration,
    save_calibration,
    load_calibration,
)


class LaserPrototypeHMI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Laser Prototype HMI (Single Image)")
        self.root.geometry("1400x820")

        self.image_path = ""
        self.bgr = None
        self.centroids = None
        self.depths = None
        self.cloud = None
        self.sequence_paths: list[str] = []
        self.layered_cloud = None
        self.last_preview = None
        self.raw_depths = None
        self.calibration = None
        self.calibration_samples: list[dict] = []

        self._build_ui()

    def _build_ui(self) -> None:
        left_shell = tk.Frame(self.root)
        left_shell.pack(side=tk.LEFT, fill=tk.Y)

        left_canvas = tk.Canvas(left_shell, width=380, highlightthickness=0)
        left_scroll = tk.Scrollbar(left_shell, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        left = tk.Frame(left_canvas, padx=10, pady=10)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_configure(_event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _on_canvas_configure(event):
            left_canvas.itemconfigure(left_window, width=event.width)

        def _on_mousewheel(event):
            # Windows mouse wheel support.
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        left.bind("<Configure>", _on_left_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        right = tk.Frame(self.root, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Button(left, text="Load Image", width=24, command=self.load_image).pack(pady=4)
        tk.Button(left, text="Process", width=24, command=self.process).pack(pady=4)
        tk.Button(left, text="Set Zero From Current Image", width=24, command=self.set_zero_from_current_image).pack(pady=4)
        tk.Button(left, text="Save Cloud CSV", width=24, command=self.save_csv).pack(pady=4)
        tk.Button(left, text="Load Sequence Images", width=24, command=self.load_sequence_images).pack(pady=4)
        tk.Button(left, text="Build Layered Cloud", width=24, command=self.build_layered_cloud).pack(pady=4)
        tk.Button(left, text="Save Layered Cloud CSV", width=24, command=self.save_layered_cloud_csv).pack(pady=4)
        tk.Button(left, text="Add Calibration Sample", width=24, command=self.add_calibration_sample).pack(pady=4)
        tk.Button(left, text="Add Sample From Profile ROI", width=24, command=self.add_calibration_sample_from_roi).pack(pady=4)
        tk.Button(left, text="Fit Calibration", width=24, command=self.fit_calibration).pack(pady=4)
        tk.Button(left, text="Save Calibration JSON", width=24, command=self.save_calibration_json).pack(pady=4)
        tk.Button(left, text="Load Calibration JSON", width=24, command=self.load_calibration_json).pack(pady=4)

        self.path_label = tk.Label(left, text="No image loaded", wraplength=300, justify=tk.LEFT)
        self.path_label.pack(pady=6, anchor="w")
        self.seq_label = tk.Label(left, text="No sequence loaded", wraplength=300, justify=tk.LEFT)
        self.seq_label.pack(pady=2, anchor="w")

        self.status = tk.StringVar(value="Ready")
        tk.Label(left, textvariable=self.status, fg="#1f7a4a", wraplength=300, justify=tk.LEFT).pack(pady=6, anchor="w")
        self.calib_status = tk.StringVar(value="Calibration: none")
        tk.Label(left, textvariable=self.calib_status, fg="#345f99", wraplength=300, justify=tk.LEFT).pack(pady=2, anchor="w")
        self.zero_status = tk.StringVar(value="Zero: not set from image")
        tk.Label(left, textvariable=self.zero_status, fg="#7a4a1f", wraplength=300, justify=tk.LEFT).pack(pady=2, anchor="w")

        frame_true = tk.Frame(left)
        frame_true.pack(fill=tk.X, pady=2)
        tk.Label(frame_true, text="true_z_mm", width=13, anchor="w").pack(side=tk.LEFT)
        self.true_z_entry = tk.Entry(frame_true, width=10)
        self.true_z_entry.insert(0, "0.0")
        self.true_z_entry.pack(side=tk.LEFT)

        frame_roi = tk.Frame(left)
        frame_roi.pack(fill=tk.X, pady=2)
        tk.Label(frame_roi, text="profile ROI", width=13, anchor="w").pack(side=tk.LEFT)
        self.roi_start_entry = tk.Entry(frame_roi, width=5)
        self.roi_start_entry.insert(0, "0")
        self.roi_start_entry.pack(side=tk.LEFT)
        tk.Label(frame_roi, text="to").pack(side=tk.LEFT)
        self.roi_end_entry = tk.Entry(frame_roi, width=5)
        self.roi_end_entry.insert(0, "0")
        self.roi_end_entry.pack(side=tk.LEFT)

        self.params = {}
        controls = [
            ("h_min", 50, 0, 180),
            ("h_max", 80, 0, 180),
            ("s_min", 80, 0, 255),
            ("s_max", 255, 0, 255),
            ("v_min", 80, 0, 255),
            ("v_max", 255, 0, 255),
            ("blur_kernel", 5, 1, 21),
            ("median_kernel", 0, 0, 21),
            ("morph_open", 0, 0, 15),
            ("morph_close", 0, 0, 15),
            ("min_blob_area", 0, 0, 2000),
            ("centroid_smooth_window", 0, 0, 51),
            ("laser_angle_deg", 30, 1, 89),
            ("mm_per_pixel", 0.1, 0.001, 2.0),
            ("zero_row", 0.0, 0.0, 2000.0),
        ]

        tk.Label(left, text="Parameters", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 2))
        for name, default, lo, hi in controls:
            frame = tk.Frame(left)
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=name, width=13, anchor="w").pack(side=tk.LEFT)
            ent = tk.Entry(frame, width=10)
            ent.insert(0, str(default))
            ent.pack(side=tk.LEFT)
            tk.Label(frame, text=f"[{lo}, {hi}]").pack(side=tk.LEFT)
            self.params[name] = ent

        self.use_calibration_var = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Apply loaded/fitted calibration", variable=self.use_calibration_var).pack(anchor="w", pady=(8, 2))

        frame_step = tk.Frame(left)
        frame_step.pack(fill=tk.X, pady=2)
        tk.Label(frame_step, text="frame_step_mm", width=13, anchor="w").pack(side=tk.LEFT)
        self.frame_step_entry = tk.Entry(frame_step, width=10)
        self.frame_step_entry.insert(0, "1.0")
        self.frame_step_entry.pack(side=tk.LEFT)

        frame_axis = tk.Frame(left)
        frame_axis.pack(fill=tk.X, pady=2)
        tk.Label(frame_axis, text="scan_axis", width=13, anchor="w").pack(side=tk.LEFT)
        self.scan_axis_var = tk.StringVar(value="y")
        tk.OptionMenu(frame_axis, self.scan_axis_var, "y", "x").pack(side=tk.LEFT)

        # Right side tabbed views for better visibility.
        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_preview = tk.Frame(notebook)
        tab_profile = tk.Frame(notebook)
        tab_cloud = tk.Frame(notebook)
        notebook.add(tab_preview, text="Mask + Overlay")
        notebook.add(tab_profile, text="Depth Profile")
        notebook.add(tab_cloud, text="3D Cloud")

        self.preview_label = tk.Label(tab_preview, bg="#111")
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=(6, 6))

        fig_profile = Figure(figsize=(8.0, 4.0), dpi=100)
        self.ax_profile = fig_profile.add_subplot(111)
        self.ax_profile.set_title("Depth Profile (Current Image)")
        self.ax_profile.set_xlabel("Column")
        self.ax_profile.set_ylabel("Z (mm)")
        self.profile_canvas = FigureCanvasTkAgg(fig_profile, master=tab_profile)
        self.profile_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        fig_cloud = Figure(figsize=(8.0, 4.0), dpi=100)
        self.ax3d = fig_cloud.add_subplot(111, projection="3d")
        self.ax3d.set_title("Single-Image Point Cloud")
        self.ax3d.set_xlabel("X (mm)")
        self.ax3d.set_ylabel("Y (mm)")
        self.ax3d.set_zlabel("Z (mm)")
        self.ax3d.view_init(elev=20, azim=-60)
        self.cloud_canvas = FigureCanvasTkAgg(fig_cloud, master=tab_cloud)
        self.cloud_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _get_float(self, name: str) -> float:
        return float(self.params[name].get().strip())

    def _get_int(self, name: str) -> int:
        return int(float(self.params[name].get().strip()))

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.bgr = read_image(path)
            self.image_path = path
            self.path_label.config(text=path)
            self.status.set("Image loaded. Press Process.")
            if self._get_float("zero_row") <= 0.0:
                self.params["zero_row"].delete(0, tk.END)
                self.params["zero_row"].insert(0, str(self.bgr.shape[0] * 0.5))
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def process(self) -> None:
        if self.bgr is None:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        try:
            cents, mask = extract_green_centroids(
                self.bgr,
                self._get_int("h_min"),
                self._get_int("h_max"),
                self._get_int("s_min"),
                self._get_int("s_max"),
                self._get_int("v_min"),
                self._get_int("v_max"),
                self._get_int("blur_kernel"),
                self._get_int("median_kernel"),
                self._get_int("morph_open"),
                self._get_int("morph_close"),
                self._get_int("min_blob_area"),
                self._get_int("centroid_smooth_window"),
            )
            raw_depths = centroids_to_depth(
                cents,
                self._get_float("laser_angle_deg"),
                self._get_float("mm_per_pixel"),
                self._get_float("zero_row"),
            )
            depths = (
                apply_calibration(raw_depths, self.calibration)
                if self.use_calibration_var.get()
                else raw_depths.copy()
            )
            cloud = build_single_line_cloud(depths, self._get_float("mm_per_pixel"))
            overlay = draw_overlay(self.bgr, cents)

            self.centroids = cents
            self.raw_depths = raw_depths
            self.depths = depths
            self.cloud = cloud
            self._update_preview(mask, overlay)
            self._update_cloud_plot(cloud)

            valid = int(np.isfinite(depths).sum())
            total = int(depths.shape[0])
            mode = "calibrated" if (self.use_calibration_var.get() and self.calibration is not None) else "raw"
            self.status.set(f"Processed ({mode}): valid points {valid}/{total}, cloud points {cloud.shape[0]}")
        except Exception as e:
            messagebox.showerror("Process Error", str(e))

    def set_zero_from_current_image(self) -> None:
        if self.bgr is None:
            messagebox.showwarning("No Image", "Load and process a flat reference image first.")
            return
        try:
            # Use centroid median row from the current image as zero plane.
            cents, _ = extract_green_centroids(
                self.bgr,
                self._get_int("h_min"),
                self._get_int("h_max"),
                self._get_int("s_min"),
                self._get_int("s_max"),
                self._get_int("v_min"),
                self._get_int("v_max"),
                self._get_int("blur_kernel"),
                self._get_int("median_kernel"),
                self._get_int("morph_open"),
                self._get_int("morph_close"),
                self._get_int("min_blob_area"),
                self._get_int("centroid_smooth_window"),
            )
            valid = cents[np.isfinite(cents)]
            if valid.size == 0:
                raise ValueError("No valid laser line points found in current image.")
            zero_row = float(np.median(valid))
            self.params["zero_row"].delete(0, tk.END)
            self.params["zero_row"].insert(0, f"{zero_row:.3f}")
            self.zero_status.set(f"Zero set from current image: zero_row={zero_row:.3f}")
            self.status.set("Zero captured. Now load known-height image, set true_z_mm, and add calibration sample.")
        except Exception as e:
            messagebox.showerror("Set Zero Error", str(e))

    def load_sequence_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select sequence images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not paths:
            return
        self.sequence_paths = sorted(list(paths))
        self.seq_label.config(text=f"Sequence loaded: {len(self.sequence_paths)} image(s)")
        self.status.set("Sequence loaded. Click Build Layered Cloud.")

    def _process_frame_to_depth(self, bgr: np.ndarray) -> np.ndarray:
        cents, _ = extract_green_centroids(
            bgr,
            self._get_int("h_min"),
            self._get_int("h_max"),
            self._get_int("s_min"),
            self._get_int("s_max"),
            self._get_int("v_min"),
            self._get_int("v_max"),
            self._get_int("blur_kernel"),
            self._get_int("median_kernel"),
            self._get_int("morph_open"),
            self._get_int("morph_close"),
            self._get_int("min_blob_area"),
            self._get_int("centroid_smooth_window"),
        )
        raw_depths = centroids_to_depth(
            cents,
            self._get_float("laser_angle_deg"),
            self._get_float("mm_per_pixel"),
            self._get_float("zero_row"),
        )
        return (
            apply_calibration(raw_depths, self.calibration)
            if self.use_calibration_var.get() and self.calibration is not None
            else raw_depths
        )

    def build_layered_cloud(self) -> None:
        if not self.sequence_paths:
            messagebox.showwarning("No Sequence", "Load sequence images first.")
            return
        try:
            frame_step_mm = float(self.frame_step_entry.get().strip())
            scan_axis = self.scan_axis_var.get().strip().lower()

            depth_rows = []
            unreadable = 0
            for p in self.sequence_paths:
                try:
                    bgr = read_image(p)
                except Exception:
                    unreadable += 1
                    continue
                depth_rows.append(self._process_frame_to_depth(bgr))

            if not depth_rows:
                raise ValueError("No readable frames in sequence.")

            cloud = build_layered_cloud_from_depth_rows(
                depth_rows,
                self._get_float("mm_per_pixel"),
                frame_step_mm,
                scan_axis=scan_axis,
            )
            self.layered_cloud = cloud
            self._update_cloud_plot(cloud)

            self.status.set(
                f"Layered cloud built: frames={len(depth_rows)}, unreadable={unreadable}, "
                f"points={cloud.shape[0]}, step={frame_step_mm:.3f} mm, axis={scan_axis}"
            )
        except Exception as e:
            messagebox.showerror("Layered Cloud Error", str(e))

    def _current_raw_depth_median(self) -> float:
        if self.raw_depths is None:
            raise ValueError("No processed image yet. Click Process first.")
        valid = self.raw_depths[np.isfinite(self.raw_depths)]
        if valid.size == 0:
            raise ValueError("No valid line points in current image.")
        return float(np.median(valid))

    def _current_raw_depth_median_roi(self) -> float:
        if self.raw_depths is None:
            raise ValueError("No processed image yet. Click Process first.")
        n = int(self.raw_depths.shape[0])
        s = int(float(self.roi_start_entry.get().strip()))
        e = int(float(self.roi_end_entry.get().strip()))
        s = max(0, min(s, n - 1))
        e = max(0, min(e, n - 1))
        if e < s:
            s, e = e, s
        roi = self.raw_depths[s : e + 1]
        valid = roi[np.isfinite(roi)]
        if valid.size == 0:
            raise ValueError("No valid points in selected ROI.")
        return float(np.median(valid))

    def add_calibration_sample(self) -> None:
        try:
            true_z = float(self.true_z_entry.get().strip())
            raw_z = self._current_raw_depth_median()
            sample = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image_path": self.image_path,
                "raw_z_mm": raw_z,
                "true_z_mm": true_z,
            }
            self.calibration_samples.append(sample)
            self.calib_status.set(f"Calibration samples: {len(self.calibration_samples)}")
            self.status.set(f"Added sample #{len(self.calibration_samples)}: raw={raw_z:.4f}, true={true_z:.4f} mm")
        except Exception as e:
            messagebox.showerror("Calibration Sample Error", str(e))

    def add_calibration_sample_from_roi(self) -> None:
        try:
            true_z = float(self.true_z_entry.get().strip())
            raw_z = self._current_raw_depth_median_roi()
            sample = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image_path": self.image_path,
                "raw_z_mm": raw_z,
                "true_z_mm": true_z,
                "source": "profile_roi",
                "roi_start_col": int(float(self.roi_start_entry.get().strip())),
                "roi_end_col": int(float(self.roi_end_entry.get().strip())),
            }
            self.calibration_samples.append(sample)
            self.calib_status.set(f"Calibration samples: {len(self.calibration_samples)}")
            self.status.set(
                f"Added ROI sample #{len(self.calibration_samples)}: raw={raw_z:.4f}, true={true_z:.4f} mm"
            )
        except Exception as e:
            messagebox.showerror("Calibration ROI Error", str(e))

    def fit_calibration(self) -> None:
        try:
            model = fit_linear_calibration(self.calibration_samples)
            self.calibration = model
            self.calib_status.set(
                f"Calibration fit: z_true={model['a']:.6f}*z_raw+{model['b']:.6f}, "
                f"rmse={model['rmse_mm']:.4f} mm, n={len(self.calibration_samples)}"
            )
            self.status.set("Calibration model fitted. Re-process image to apply.")
        except Exception as e:
            messagebox.showerror("Calibration Fit Error", str(e))

    def save_calibration_json(self) -> None:
        out = filedialog.asksaveasfilename(
            title="Save calibration JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not out:
            return
        payload = {
            "model": self.calibration,
            "samples": self.calibration_samples,
            "meta": {
                "laser_angle_deg": self._get_float("laser_angle_deg"),
                "mm_per_pixel": self._get_float("mm_per_pixel"),
                "zero_row": self._get_float("zero_row"),
            },
        }
        save_calibration(out, payload)
        self.status.set(f"Saved calibration JSON: {out}")

    def load_calibration_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Load calibration JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = load_calibration(path)
        self.calibration = payload.get("model")
        self.calibration_samples = list(payload.get("samples", []))
        if self.calibration:
            m = self.calibration
            self.calib_status.set(
                f"Calibration loaded: z_true={m['a']:.6f}*z_raw+{m['b']:.6f}, "
                f"rmse={m.get('rmse_mm', 0.0):.4f} mm, n={len(self.calibration_samples)}"
            )
        else:
            self.calib_status.set("Calibration loaded: no model inside file")
        self.status.set(f"Loaded calibration JSON: {path}")

    def _update_preview(self, mask: np.ndarray, overlay_bgr: np.ndarray) -> None:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([mask_rgb, overlay_bgr])
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        max_w = 980
        max_h = 360
        h, w = combined.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(combined)
        tk_img = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=tk_img)
        self.preview_label.image = tk_img
        self.last_preview = tk_img

    def _update_cloud_plot(self, cloud: np.ndarray) -> None:
        self.ax_profile.clear()
        self.ax_profile.set_title("Depth Profile (Current Image)")
        self.ax_profile.set_xlabel("Column")
        self.ax_profile.set_ylabel("Z (mm)")
        if self.depths is not None and self.depths.size > 0:
            cols = np.arange(self.depths.shape[0], dtype=np.int32)
            self.ax_profile.plot(cols, self.depths, linewidth=1.0, label="depth")
            if self.raw_depths is not None and self.use_calibration_var.get() and self.calibration is not None:
                self.ax_profile.plot(cols, self.raw_depths, linewidth=1.0, alpha=0.5, label="raw")
            try:
                s = int(float(self.roi_start_entry.get().strip()))
                e = int(float(self.roi_end_entry.get().strip()))
                if e < s:
                    s, e = e, s
                self.ax_profile.axvspan(s, e, alpha=0.15, color="orange")
            except Exception:
                pass
            self.ax_profile.grid(True, alpha=0.3)
            self.ax_profile.legend(loc="best")
        self.profile_canvas.draw_idle()

        self.ax3d.clear()
        self.ax3d.set_title("Single-Image Point Cloud")
        self.ax3d.set_xlabel("X (mm)")
        self.ax3d.set_ylabel("Y (mm)")
        self.ax3d.set_zlabel("Z (mm)")
        self.ax3d.view_init(elev=20, azim=-60)
        if cloud.size > 0:
            self.ax3d.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=3)
        self.cloud_canvas.draw_idle()

    def save_csv(self) -> None:
        if self.cloud is None or self.cloud.size == 0:
            messagebox.showwarning("No Cloud", "Process an image first.")
            return
        out = filedialog.asksaveasfilename(
            title="Save cloud CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not out:
            return
        np.savetxt(Path(out), self.cloud, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
        self.status.set(f"Saved cloud CSV: {out}")

    def save_layered_cloud_csv(self) -> None:
        if self.layered_cloud is None or self.layered_cloud.size == 0:
            messagebox.showwarning("No Layered Cloud", "Build layered cloud first.")
            return
        out = filedialog.asksaveasfilename(
            title="Save layered cloud CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not out:
            return
        np.savetxt(Path(out), self.layered_cloud, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
        self.status.set(f"Saved layered cloud CSV: {out}")


def main() -> None:
    root = tk.Tk()
    LaserPrototypeHMI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
