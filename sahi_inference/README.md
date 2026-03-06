# SAHI Small Polyp Detection

This folder contains scripts to run **SAHI (Slicing Aided Hyper Inference)** on your trained YOLO model to dramatically improve detection of **small polyps**.

## Why SAHI?

Standard YOLO inference resizes an image to 640×640 — small polyps can shrink to just a few pixels and get missed. SAHI fixes this by:

1. Dividing the image into overlapping slices (e.g., 320×320)
2. Running YOLO on each slice at full resolution
3. Merging all detections with Non-Maximum Merging (NMM)

## Files

| File | Purpose |
|------|---------|
| `sahi_predict.py` | Run SAHI inference on images/folders |
| `sahi_evaluate.py` | Compare standard vs SAHI detection counts |
| `requirements_sahi.txt` | Python dependencies |

## Quick Start

### 1. Install SAHI

```bash
# From the project root, activate your venv first
.\.venv\Scripts\activate       # Windows
pip install -r sahi_inference/requirements_sahi.txt
```

### 2. Run SAHI inference on a single image

```bash
cd sahi_inference
python sahi_predict.py \
    --source path/to/test_image.jpg \
    --model ../models/detection/weights/best.pt \
    --device cpu \
    --slice_height 320 \
    --slice_width 320 \
    --overlap_ratio 0.2 \
    --output outputs
```

### 3. Compare standard vs SAHI (side-by-side)

```bash
python sahi_predict.py \
    --source path/to/test_images/ \
    --model ../models/detection/weights/best.pt \
    --compare
```

### 4. Evaluate on a test set

```bash
python sahi_evaluate.py \
    --images_dir ../data/test/images \
    --model ../models/detection/weights/best.pt
```

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--slice_height` | 320 | Smaller = catches tinier polyps, slower |
| `--slice_width` | 320 | Match slice_height usually |
| `--overlap_ratio` | 0.2 | 0.2 = 20% overlap. Increase if polyps get cut by slice edges |
| `--confidence` | 0.25 | Lower it slightly to catch more but expect more FPs |
| `--device` | `cpu` | Use `0` or `cuda:0` if you have a GPU |

## Tips for Polyp Detection

- Start with `slice_height=320, overlap=0.2` and check outputs visually
- If you see sliced polyps at boundaries, **increase overlap to 0.3–0.4**
- If too many false positives, **raise confidence to 0.4**
- Use `--compare` to see exactly what standard YOLO misses
