# Python Laser Prototype

Standalone prototype (separate from the C++ app) to:

1. Read one image
2. Detect the green laser line
3. Convert line displacement to depth using triangulation
4. Plot single-image point cloud points along that line in 3D

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py --image "C:\path\to\image.jpg"
```

## Run HMI (GUI)

```bash
python hmi.py
```

## Run PySide6 HMI (refactor)

```bash
python hmi_pyside6.py
```

This is the newer, more scalable UI with:

- scrollable controls panel
- tabbed views (`Mask + Overlay`, `Depth Profile`, `3D Cloud`)
- calibration + sequence layering workflow

## Run Calibration-Only Window

```bash
python calibration_window.py
```

This is the stripped calibration flow only:

1. Load known image
2. Process image
3. Set zero from flat reference
4. Pick width on depth profile (2 clicks) + known width
5. Pick height on depth profile (2 clicks) + known height
6. Fit calibration
7. Save calibration JSON

In the HMI:

1. Click `Load Image`
2. Tune HSV / triangulation params
3. Click `Process`
4. Inspect:
   - mask + detected red line preview
   - single-image point cloud in 3D
5. Optional: `Save Cloud CSV`

## Multi-image layered cloud (HMI)

1. Click `Load Sequence Images` and select multiple frames.
2. Set:
   - `frame_step_mm` (distance between images)
   - `scan_axis`:
     - `y`: frames are layered along Y (common line-scan layout)
     - `x`: frames are layered along X
3. Click `Build Layered Cloud`.
4. Optional: `Save Layered Cloud CSV`.

Calibration (if enabled) is applied to each frame before layering.

Optional useful args:

- `--laser-angle-deg 30`
- `--mm-per-pixel 0.1`
- `--zero-row 540` (reference row in pixels; default is image midpoint)
- `--save-csv cloud.csv`
- `--calibration-json calibration.json`
- `--median-kernel 3`
- `--morph-open 3`
- `--morph-close 3`
- `--min-blob-area 40`
- `--centroid-smooth-window 7`

Example:

```bash
python app.py --image "C:\data\scan_001.jpg" --laser-angle-deg 30 --mm-per-pixel 0.1 --zero-row 540 --save-csv cloud.csv
```

## Calibration (HMI)

Two-image quick workflow:

1. Load a flat reference image and click `Process`.
2. Click `Set Zero From Current Image`.
3. Load an image with known height target and click `Process`.
4. Enter known `true_z_mm`.
5. Click `Add Sample From Profile ROI` (or `Add Calibration Sample`).
6. Repeat for more known heights (2+; 4-8 recommended).
7. Click `Fit Calibration`.
8. Keep `Apply loaded/fitted calibration` checked, then run sequence.
9. Save via `Save Calibration JSON` for later reuse.

The fitted model is linear:

- `z_true = a * z_raw + b`

## Denoising tips

If the laser extraction is noisy, start with:

- `median_kernel = 3`
- `morph_open = 3`
- `morph_close = 3`
- `min_blob_area = 20..100`
- `centroid_smooth_window = 5..11`

Increase slowly to avoid over-smoothing actual profile shape.

## Notes

- Depth formula:
  - `depth_mm = (zero_row - centroid_row) * mm_per_pixel / tan(laser_angle)`
- This prototype builds point cloud from **one image only**:
  - `X = column * mm_per_pixel`
  - `Y = 0`
  - `Z = depth`
- Multi-image layering can be added next by assigning `Y = frame_index * scan_step_mm`.
