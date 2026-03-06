"""
SAHI-based inference for small polyp detection.
Uses Slicing Aided Hyper Inference (SAHI) to detect small polyps
by slicing images and running YOLO on each slice.

Usage:
    python sahi_predict.py --source path/to/image_or_folder
                           --model ../models/detection/weights/best.pt
                           --slice_height 320 --slice_width 320
                           --overlap_ratio 0.2
"""

import argparse
import os
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.utils.cv import visualize_object_predictions, read_image
from PIL import Image


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_path: str, confidence_threshold: float = 0.25, device: str = "cpu"):
    """Load a YOLO model using SAHI's AutoDetectionModel wrapper."""
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",          # works for YOLOv8 / YOLOv11
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )
    return detection_model


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #

def predict_single(image_path: str, model, output_dir: str):
    """Run standard (full-image) YOLO inference — baseline comparison."""
    result = get_prediction(
        image=image_path,
        detection_model=model,
    )
    _save_result(image_path, result, output_dir, prefix="standard")
    return result


def predict_sliced(
    image_path: str,
    model,
    output_dir: str,
    slice_height: int = 320,
    slice_width: int = 320,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
):
    """
    Run SAHI sliced inference.
    The image is divided into overlapping slices; YOLO runs on each slice
    and predictions are merged with NMM (Non-Maximum Merging).
    """
    result = get_sliced_prediction(
        image=image_path,
        detection_model=model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    _save_result(image_path, result, output_dir, prefix="sahi")
    return result


# --------------------------------------------------------------------------- #
# Result saving
# --------------------------------------------------------------------------- #

def _save_result(image_path: str, result, output_dir: str, prefix: str):
    """Save annotated image and print detection summary."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    out_path = os.path.join(output_dir, f"{prefix}_{stem}.jpg")

    result.export_visuals(export_dir=output_dir, file_name=f"{prefix}_{stem}")
    print(f"\n{'='*60}")
    print(f"  [{prefix.upper()}] Results for: {image_path}")
    print(f"  Detections: {len(result.object_prediction_list)}")
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        score = pred.score.value
        label = pred.category.name
        print(f"    • {label}  score={score:.3f}  box={[round(v,1) for v in bbox]}")
    print(f"  Saved → {output_dir}")
    print(f"{'='*60}")


# --------------------------------------------------------------------------- #
# Batch processing
# --------------------------------------------------------------------------- #

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def run_batch(source: str, model, args):
    """Process a single image or a directory of images."""
    source_path = Path(source)
    images = []

    if source_path.is_dir():
        images = [
            str(p)
            for p in source_path.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    elif source_path.is_file():
        images = [str(source_path)]
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    print(f"\nFound {len(images)} image(s) to process.\n")

    for img_path in images:
        if args.compare:
            # Run both standard and SAHI inference side-by-side for comparison
            predict_single(img_path, model, args.output)
        predict_sliced(
            image_path=img_path,
            model=model,
            output_dir=args.output,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_ratio,
            overlap_width_ratio=args.overlap_ratio,
        )


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAHI sliced inference for small polyp detection"
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to an image file or a directory of images"
    )
    parser.add_argument(
        "--model",
        default=os.path.join("..", "models", "detection", "weights", "best.pt"),
        help="Path to your YOLO .pt weights file"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: 'cpu', '0' (GPU), 'cuda:0', etc."
    )
    parser.add_argument(
        "--slice_height", type=int, default=320,
        help="Height of each image slice in pixels (default: 320)"
    )
    parser.add_argument(
        "--slice_width", type=int, default=320,
        help="Width of each image slice in pixels (default: 320)"
    )
    parser.add_argument(
        "--overlap_ratio", type=float, default=0.2,
        help="Overlap ratio between slices (0–1, default: 0.2)"
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Directory to save annotated results (default: ./outputs)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run standard (full-image) inference for side-by-side comparison"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n📦 Loading model from: {args.model}")
    model = load_model(args.model, args.confidence, args.device)
    run_batch(args.source, model, args)
    print("\n✅ Done! Results saved to:", args.output)
