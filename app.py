import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(path: str) -> np.ndarray:
    # Robust Windows-safe file read.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def extract_green_centroids(
    bgr: np.ndarray,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
    blur_kernel: int,
    median_kernel: int = 0,
    morph_open: int = 0,
    morph_close: int = 0,
    min_blob_area: int = 0,
    centroid_smooth_window: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if blur_kernel < 1:
        blur_kernel = 1
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    work = cv2.GaussianBlur(bgr, (blur_kernel, blur_kernel), 0) if blur_kernel > 1 else bgr
    if median_kernel and median_kernel > 1:
        if median_kernel % 2 == 0:
            median_kernel += 1
        work = cv2.medianBlur(work, median_kernel)

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

    if morph_open and morph_open > 1:
        k = morph_open + (1 - morph_open % 2)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)

    if morph_close and morph_close > 1:
        k = morph_close + (1 - morph_close % 2)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)

    if min_blob_area and min_blob_area > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for lab in range(1, n):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_blob_area:
                clean[labels == lab] = 255
        mask = clean

    h, w = mask.shape
    centroids = np.full(w, np.nan, dtype=np.float32)
    rows = np.arange(h, dtype=np.float32)
    # Weight centroid by green intensity (inside mask) to reduce jitter.
    green = work[:, :, 1].astype(np.float32)

    for c in range(w):
        col = (mask[:, c].astype(np.float32) / 255.0) * green[:, c]
        wsum = col.sum()
        if wsum > 1.0:
            centroids[c] = float((col * rows).sum() / wsum)

    if centroid_smooth_window and centroid_smooth_window > 1:
        k = centroid_smooth_window + (1 - centroid_smooth_window % 2)
        half = k // 2
        out = centroids.copy()
        for c in range(w):
            s = max(0, c - half)
            e = min(w, c + half + 1)
            seg = centroids[s:e]
            valid = seg[np.isfinite(seg)]
            if valid.size:
                out[c] = float(np.median(valid))
        centroids = out

    return centroids, mask


def centroids_to_depth(
    centroids: np.ndarray,
    laser_angle_deg: float,
    mm_per_pixel: float,
    zero_row: float,
) -> np.ndarray:
    angle_rad = math.radians(laser_angle_deg)
    tan_a = math.tan(angle_rad)
    if abs(tan_a) < 1e-9:
        raise ValueError("Laser angle too close to 0 deg for triangulation.")
    scale = mm_per_pixel / tan_a
    return (zero_row - centroids) * scale


def build_single_line_cloud(depths: np.ndarray, mm_per_pixel: float) -> np.ndarray:
    valid = ~np.isnan(depths)
    cols = np.where(valid)[0].astype(np.float32)
    x = cols * mm_per_pixel
    y = np.zeros_like(x)
    z = depths[valid].astype(np.float32)
    return np.column_stack([x, y, z])


def build_layered_cloud_from_depth_rows(
    depth_rows: list[np.ndarray],
    mm_per_pixel: float,
    frame_step_mm: float,
    scan_axis: str = "y",
) -> np.ndarray:
    """
    Build cloud from multiple depth rows (one row per frame).
    scan_axis:
      - "y": columns map to X, frame index maps to Y
      - "x": frame index maps to X, columns map to Y
    """
    if frame_step_mm <= 0:
        raise ValueError("frame_step_mm must be > 0")
    if scan_axis not in ("x", "y"):
        raise ValueError("scan_axis must be 'x' or 'y'")

    points = []
    for i, depths in enumerate(depth_rows):
        valid = ~np.isnan(depths)
        cols = np.where(valid)[0].astype(np.float32)
        z = depths[valid].astype(np.float32)
        if cols.size == 0:
            continue
        if scan_axis == "y":
            x = cols * mm_per_pixel
            y = np.full_like(x, i * frame_step_mm, dtype=np.float32)
        else:
            x = np.full_like(cols, i * frame_step_mm, dtype=np.float32)
            y = cols * mm_per_pixel
        points.append(np.column_stack([x, y, z]))

    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(points).astype(np.float32)


def fit_linear_calibration(samples: list[dict]) -> dict:
    if len(samples) < 2:
        raise ValueError("Need at least 2 calibration samples.")
    raw = np.array([float(s["raw_z_mm"]) for s in samples], dtype=np.float64)
    true = np.array([float(s["true_z_mm"]) for s in samples], dtype=np.float64)
    a, b = np.polyfit(raw, true, 1)
    pred = a * raw + b
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return {"type": "linear", "a": float(a), "b": float(b), "rmse_mm": rmse}


def apply_calibration(depths: np.ndarray, calibration: dict | None) -> np.ndarray:
    if calibration is None:
        return depths.copy()
    if calibration.get("type") != "linear":
        raise ValueError(f"Unsupported calibration type: {calibration.get('type')}")
    a = float(calibration["a"])
    b = float(calibration["b"])
    out = depths.copy()
    valid = np.isfinite(out)
    out[valid] = a * out[valid] + b
    return out


def save_calibration(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def draw_overlay(bgr: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    prev = None
    for c, r in enumerate(centroids):
        if np.isnan(r):
            prev = None
            continue
        p = (int(c), int(round(r)))
        if prev is not None:
            cv2.line(out, prev, p, (0, 0, 255), 2)  # red
        prev = p
    return out


def visualize(mask: np.ndarray, overlay_bgr: np.ndarray, cloud_xyz: np.ndarray) -> None:
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Green Mask")
    ax1.imshow(mask, cmap="gray")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Detected Laser Line")
    ax2.imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.set_title("Single-Image Point Cloud")
    if cloud_xyz.size > 0:
        ax3.scatter(cloud_xyz[:, 0], cloud_xyz[:, 1], cloud_xyz[:, 2], s=2)
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.set_zlabel("Depth Z (mm)")
    ax3.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype: extract green laser line and build single-image point cloud."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--laser-angle-deg", type=float, default=30.0)
    parser.add_argument("--mm-per-pixel", type=float, default=0.1)
    parser.add_argument("--zero-row", type=float, default=None, help="Reference row in pixels")
    parser.add_argument("--h-min", type=int, default=50)
    parser.add_argument("--h-max", type=int, default=80)
    parser.add_argument("--s-min", type=int, default=80)
    parser.add_argument("--s-max", type=int, default=255)
    parser.add_argument("--v-min", type=int, default=80)
    parser.add_argument("--v-max", type=int, default=255)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--median-kernel", type=int, default=0)
    parser.add_argument("--morph-open", type=int, default=0)
    parser.add_argument("--morph-close", type=int, default=0)
    parser.add_argument("--min-blob-area", type=int, default=0)
    parser.add_argument("--centroid-smooth-window", type=int, default=0)
    parser.add_argument("--save-csv", default="", help="Optional CSV output for point cloud")
    parser.add_argument("--calibration-json", default="", help="Optional calibration JSON to correct Z")
    args = parser.parse_args()

    img = read_image(args.image)
    centroids, mask = extract_green_centroids(
        img,
        args.h_min,
        args.h_max,
        args.s_min,
        args.s_max,
        args.v_min,
        args.v_max,
        args.blur_kernel,
        args.median_kernel,
        args.morph_open,
        args.morph_close,
        args.min_blob_area,
        args.centroid_smooth_window,
    )

    zero_row = args.zero_row if args.zero_row is not None else (img.shape[0] * 0.5)
    raw_depths = centroids_to_depth(centroids, args.laser_angle_deg, args.mm_per_pixel, zero_row)
    calib_payload = load_calibration(args.calibration_json) if args.calibration_json else None
    calib = None
    if calib_payload is not None:
        calib = calib_payload.get("model") if isinstance(calib_payload, dict) and "model" in calib_payload else calib_payload
    depths = apply_calibration(raw_depths, calib)
    cloud = build_single_line_cloud(depths, args.mm_per_pixel)
    overlay = draw_overlay(img, centroids)

    valid = np.isfinite(depths)
    print(f"Valid line points: {int(valid.sum())}/{len(depths)}")
    if valid.any():
        print(f"Depth range (mm): {float(np.nanmin(depths)):.4f} .. {float(np.nanmax(depths)):.4f}")
    if calib:
        print(
            "Applied calibration: "
            f"type={calib.get('type')} a={calib.get('a')} b={calib.get('b')}"
        )
    print(f"Point cloud points: {cloud.shape[0]}")

    if args.save_csv:
        out = Path(args.save_csv)
        np.savetxt(out, cloud, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
        print(f"Saved point cloud CSV: {out}")

    visualize(mask, overlay, cloud)


if __name__ == "__main__":
    main()
