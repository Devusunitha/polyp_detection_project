import os
import cv2
import csv
import argparse
from tqdm import tqdm

def generate_bbox_from_mask(mask_path, class_id=0):
    """
    Generates YOLO formatted labels and raw pixel coordinates from a mask.

    Args:
        mask_path (str): The full path to the mask image file.
        class_id (int): The class ID for the object (default is 0 for polyp).

    Returns:
        tuple: (list of YOLO strings, list of raw bbox dicts [x,y,w,h]).
               Returns ([], []) if no valid contours found or error.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}. Skipping.")
        return [], []

    H, W = mask.shape
    if H == 0 or W == 0:
        print(f"Warning: Invalid mask dimensions for {mask_path}. Skipping.")
        return [], []

    # Find contours of the polyp(s) in the mask
    # RETR_EXTERNAL finds only outer contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_labels = []
    raw_bboxes = []

    if not contours:
        # print(f"Warning: No contours found in {mask_path}. Skipping.")
        return [], []

    # Often there's only one main contour, but handle multiple just in case.
    # If multiple contours, consider taking the largest one or iterating.
    # For simplicity, let's process all sufficiently large contours found.
    valid_contours_found = False
    for contour in contours:
        # Filter out very small contours that might be noise
        if cv2.contourArea(contour) < 20: # Minimum pixel area threshold
            continue

        valid_contours_found = True
        # Get bounding box from contour
        x, y, w, h = cv2.boundingRect(contour)

        # Ensure width and height are positive
        if w <= 0 or h <= 0:
            continue

        # --- Store Raw Pixel Coordinates ---
        raw_bboxes.append({'x': x, 'y': y, 'w': w, 'h': h})

        # --- Convert to YOLO format (normalized) ---
        x_center = (x + w / 2) / W
        y_center = (y + h / 2) / H
        norm_width = w / W
        norm_height = h / H

        # Clamp values to be safe (should not be necessary with correct bbox)
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_width = max(0.0, min(1.0, norm_width))
        norm_height = max(0.0, min(1.0, norm_height))

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

    # if not valid_contours_found:
        # print(f"Warning: No contours large enough found in {mask_path}. Skipping.")

    return yolo_labels, raw_bboxes

def process_all_masks(raw_mask_dir, label_output_dir, csv_output_path):
    """
    Processes all masks in a directory to generate YOLO labels and a CSV manifest.
    """
    print(f"Reading masks from: {raw_mask_dir}")
    print(f"Saving YOLO labels to: {label_output_dir}")
    print(f"Saving BBox manifest to: {csv_output_path}")

    os.makedirs(label_output_dir, exist_ok=True)
    # Ensure the directory for the CSV exists
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    mask_files = [f for f in os.listdir(raw_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))] # Added .tif based on dataset info

    if not mask_files:
        raise FileNotFoundError(f"No mask images found in '{raw_mask_dir}'. Please check the path.")

    print(f"Found {len(mask_files)} masks. Starting bounding box generation...")

    # Open CSV file for writing
    with open(csv_output_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'x', 'y', 'w', 'h']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for mask_filename in tqdm(mask_files, desc="Generating Bounding Boxes"):
            mask_path = os.path.join(raw_mask_dir, mask_filename)
            base_filename = os.path.splitext(mask_filename)[0]

            yolo_labels, raw_bboxes = generate_bbox_from_mask(mask_path)

            # --- Save YOLO Label ---
            if yolo_labels:
                label_filepath = os.path.join(label_output_dir, f"{base_filename}.txt")
                with open(label_filepath, 'w') as f:
                    # If multiple valid contours were found, write all (or decide on a strategy like largest)
                    # Current code writes one line per valid contour. Adjust if needed.
                     f.write("\n".join(yolo_labels))
            # else:
                # print(f"No valid label generated for {mask_filename}")

            # --- Save Raw BBox to CSV ---
            # If multiple bounding boxes were found, save the first or largest one.
            # Here, we save the first one found. Modify if a different strategy is needed.
            if raw_bboxes:
                 # Get the base filename without extension for the CSV
                csv_base_filename = os.path.basename(mask_filename)
                writer.writerow({
                    'filename': csv_base_filename, # Save with extension
                    'x': raw_bboxes[0]['x'],
                    'y': raw_bboxes[0]['y'],
                    'w': raw_bboxes[0]['w'],
                    'h': raw_bboxes[0]['h']
                })
            # else:
                # print(f"No raw bounding box generated for {mask_filename}")


    print(f"✅ Bounding box generation complete.")
    print(f"   YOLO Labels saved in: '{label_output_dir}'")
    print(f"   Raw BBox CSV saved in: '{csv_output_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate YOLO labels and a raw bbox CSV from masks.")
    parser.add_argument('--mask-dir', type=str, default='data/raw/masks',
                        help='Directory containing the raw mask images.')
    parser.add_argument('--label-out-dir', type=str, default='data/interim/labels',
                        help='Directory to save the generated YOLO .txt labels.')
    parser.add_argument('--csv-out-path', type=str, default='data/interim/bounding_boxes.csv',
                        help='Path to save the generated raw bounding box CSV manifest.')

    args = parser.parse_args()

    # Create parent directory for interim files if it doesn't exist
    os.makedirs(os.path.dirname(args.label_out_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_out_path), exist_ok=True)

    process_all_masks(raw_mask_dir=args.mask_dir,
                      label_output_dir=args.label_out_dir,
                      csv_output_path=args.csv_out_path)
