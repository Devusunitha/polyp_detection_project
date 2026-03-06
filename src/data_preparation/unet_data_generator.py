import os
import cv2
import csv
import argparse
from tqdm import tqdm
import numpy as np

def find_image_extension(image_dir, base_filename):
    """Finds the correct extension for an image file (.jpg, .png, etc.)."""
    for ext in ['.jpg', '.png', '.jpeg', '.tif']: # Added .tif
        path = os.path.join(image_dir, base_filename + ext)
        if os.path.exists(path):
            return ext, path
    return None, None

def create_unet_patches(args):
    """
    Generates cropped and resized image and mask patches for U-Net training.
    Reads bounding box info from a CSV and determines splits based on
    the file structure created by data_splitter.py.
    """
    print("--- Starting U-Net Patch Generation ---")

    # --- 1. Load Bounding Box Data ---
    print(f"Loading bounding box data from: {args.csv_path}")
    bbox_data = {}
    try:
        with open(args.csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Use filename *with* extension as the key
                bbox_data[row['filename']] = {
                    'x': int(row['x']),
                    'y': int(row['y']),
                    'w': int(row['w']),
                    'h': int(row['h'])
                }
        if not bbox_data:
             raise FileNotFoundError
        print(f"Loaded bounding box info for {len(bbox_data)} images.")
    except FileNotFoundError:
        print(f"Error: Bounding box CSV file not found at '{args.csv_path}'.")
        print("Please run the modified bbox_generator script first.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    output_size = tuple(args.output_size)
    padding = args.padding

    # --- 2. Loop through splits (train, val, test) ---
    splits = ['train', 'val', 'test']
    for split_name in splits:
        print(f"\nProcessing '{split_name}' split...")

        split_image_dir = os.path.join(args.processed_dir, 'images', split_name)
        if not os.path.isdir(split_image_dir):
            print(f"Warning: Processed image directory not found for split '{split_name}': {split_image_dir}")
            print("         Make sure data_splitter.py has been run.")
            continue

        # Get list of filenames for this split from the processed directory
        split_filenames = [f for f in os.listdir(split_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if not split_filenames:
            print(f"No images found in {split_image_dir}. Skipping split '{split_name}'.")
            continue

        # Create output directories for this split
        unet_img_out_dir = os.path.join(args.processed_dir, 'unet_images', split_name)
        unet_mask_out_dir = os.path.join(args.processed_dir, 'unet_masks', split_name)
        os.makedirs(unet_img_out_dir, exist_ok=True)
        os.makedirs(unet_mask_out_dir, exist_ok=True)

        # --- 3. Process each file in the split ---
        for filename_with_ext in tqdm(split_filenames, desc=f"Generating {split_name} patches"):
            base_filename = os.path.splitext(filename_with_ext)[0]

            # --- Get BBox Info ---
            if filename_with_ext not in bbox_data:
                print(f"Warning: Bounding box info not found for '{filename_with_ext}' in CSV. Skipping.")
                continue
            coords = bbox_data[filename_with_ext]
            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']

            # --- Construct paths to RAW data ---
            raw_img_ext, raw_img_path = find_image_extension(args.raw_image_dir, base_filename)
            raw_mask_ext, raw_mask_path = find_image_extension(args.raw_mask_dir, base_filename)

            if not raw_img_path or not raw_mask_path:
                print(f"Warning: Raw image or mask not found for '{base_filename}'. Skipping.")
                continue

            # --- Load Raw Image and Mask ---
            try:
                original_image = cv2.imread(raw_img_path)
                original_mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
                if original_image is None or original_mask is None:
                    raise IOError("Image/Mask could not be loaded.")
                img_H, img_W = original_image.shape[:2]
            except Exception as e:
                print(f"Error loading image/mask for '{base_filename}': {e}. Skipping.")
                continue

            # --- Calculate Padded Coordinates ---
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_W, x + w + padding)
            y2 = min(img_H, y + h + padding)

            # Ensure coordinates are valid and width/height > 0 after padding/clamping
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w <= 0 or crop_h <= 0:
                 print(f"Warning: Invalid crop dimensions ({crop_w}x{crop_h}) for '{base_filename}' after padding. Skipping.")
                 continue

            # --- Crop Image and Mask ---
            cropped_image = original_image[y1:y2, x1:x2]
            cropped_mask = original_mask[y1:y2, x1:x2]

            # --- Resize Image and Mask ---
            try:
                # Resize image using bilinear interpolation (default)
                resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)

                # Resize mask using nearest neighbor interpolation
                resized_mask = cv2.resize(cropped_mask, output_size, interpolation=cv2.INTER_NEAREST)

                # Ensure mask is still binary after resize (sometimes interpolation can introduce intermediate values)
                _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

            except Exception as e:
                print(f"Error resizing image/mask for '{base_filename}': {e}. Skipping.")
                continue

            # --- Save Resized Patches ---
            # Use PNG format for lossless saving, especially important for masks
            out_img_path = os.path.join(unet_img_out_dir, f"{base_filename}.png")
            out_mask_path = os.path.join(unet_mask_out_dir, f"{base_filename}.png")

            try:
                cv2.imwrite(out_img_path, resized_image)
                cv2.imwrite(out_mask_path, resized_mask)
            except Exception as e:
                 print(f"Error saving image/mask for '{base_filename}': {e}. Skipping.")


    print("\nU-Net patch generation complete!")
    print(f"   Processed data saved in:")
    print(f"     - Images: '{os.path.join(args.processed_dir, 'unet_images')}'")
    print(f"     - Masks:  '{os.path.join(args.processed_dir, 'unet_masks')}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate U-Net patches from raw data using CSV bbox and split info.")

    # Input Paths
    parser.add_argument('--raw-image-dir', type=str, default='data/raw/images',
                        help='Directory of raw original images.')
    parser.add_argument('--raw-mask-dir', type=str, default='data/raw/masks',
                        help='Directory of raw original masks.')
    parser.add_argument('--csv-path', type=str, default='data/interim/bounding_boxes.csv',
                        help='Path to the CSV manifest with raw bounding box coordinates.')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Root directory where data_splitter.py saved its output (used to determine splits).')

    # Output & Processing Parameters
    parser.add_argument('--output-size', nargs=2, type=int, default=[256, 256],
                        help='Target output size (height width) for U-Net patches.')
    parser.add_argument('--padding', type=int, default=20,
                        help='Pixels of padding to add around the bounding box before cropping.')

    args = parser.parse_args()

    # Create parent output directories if they don't exist
    os.makedirs(os.path.join(args.processed_dir, 'unet_images'), exist_ok=True)
    os.makedirs(os.path.join(args.processed_dir, 'unet_masks'), exist_ok=True)

    create_unet_patches(args)
