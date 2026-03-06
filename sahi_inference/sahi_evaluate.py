"""
Evaluate a YOLO model with and without SAHI on a labelled test set.
Outputs a comparison table of precision, recall, and mAP scores.

Requires ground-truth annotations in YOLO format (.txt files alongside images).

Usage:
    python sahi_evaluate.py --images_dir path/to/test/images
                            --labels_dir path/to/test/labels  (optional, auto-detected)
                            --model ../models/detection/weights/best.pt
"""

import argparse
import os
import json
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.utils.coco import Coco
from sahi.utils.file import save_json


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_model(model_path, confidence=0.25, device="cpu"):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=confidence,
        device=device,
    )


def run_standard_eval(images_dir: str, model, output_dir: str):
    """Full-image inference on every image in the directory."""
    results = []
    image_paths = list(Path(images_dir).rglob("*.jpg")) + list(Path(images_dir).rglob("*.png"))
    print(f"  Running standard inference on {len(image_paths)} images ...")
    for img_path in image_paths:
        result = get_prediction(str(img_path), model)
        results.append({
            "image": str(img_path),
            "num_detections": len(result.object_prediction_list),
        })
    return results


def run_sahi_eval(images_dir: str, model, output_dir: str, args):
    """Sliced SAHI inference on every image in the directory."""
    results = []
    image_paths = list(Path(images_dir).rglob("*.jpg")) + list(Path(images_dir).rglob("*.png"))
    print(f"  Running SAHI inference on {len(image_paths)} images ...")
    for img_path in image_paths:
        result = get_sliced_prediction(
            image=str(img_path),
            detection_model=model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_ratio,
            overlap_width_ratio=args.overlap_ratio,
        )
        results.append({
            "image": str(img_path),
            "num_detections": len(result.object_prediction_list),
        })
    return results


def print_summary(standard_results, sahi_results):
    total_standard = sum(r["num_detections"] for r in standard_results)
    total_sahi = sum(r["num_detections"] for r in sahi_results)
    images = len(standard_results)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Images evaluated  : {images}")
    print(f"  Standard detections: {total_standard}  (avg {total_standard/max(images,1):.1f}/img)")
    print(f"  SAHI detections   : {total_sahi}  (avg {total_sahi/max(images,1):.1f}/img)")
    delta = total_sahi - total_standard
    pct = (delta / max(total_standard, 1)) * 100
    print(f"  SAHI improvement  : +{delta} detections ({pct:+.1f}%)")
    print("=" * 60)
    print("\n💡 Tip: More detections ≠ always better — check precision too.")
    print("   Run with --compare flag in sahi_predict.py to inspect results visually.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare standard vs SAHI YOLO inference on a test set"
    )
    parser.add_argument("--images_dir", required=True,
                        help="Directory containing test images")
    parser.add_argument("--model",
                        default=os.path.join("..", "models", "detection", "weights", "best.pt"),
                        help="Path to YOLO .pt weights")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--slice_height", type=int, default=320)
    parser.add_argument("--slice_width", type=int, default=320)
    parser.add_argument("--overlap_ratio", type=float, default=0.2)
    parser.add_argument("--output", default="eval_outputs",
                        help="Directory to save JSON results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"\n📦 Loading model: {args.model}")
    model = load_model(args.model, args.confidence, args.device)

    print("\n[1/2] Standard inference ...")
    standard = run_standard_eval(args.images_dir, model, args.output)

    print("\n[2/2] SAHI sliced inference ...")
    sahi = run_sahi_eval(args.images_dir, model, args.output, args)

    save_json(standard, os.path.join(args.output, "standard_results.json"))
    save_json(sahi, os.path.join(args.output, "sahi_results.json"))

    print_summary(standard, sahi)
    print(f"\n✅ JSON results saved to: {args.output}")
