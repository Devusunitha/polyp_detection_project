import os
import random
import shutil
import argparse
from tqdm import tqdm

def find_image_extension(image_dir, base_filename):
    """Finds the correct extension for an image file (.jpg, .png, etc.)."""
    for ext in ['.jpg', '.png', '.jpeg']:
        if os.path.exists(os.path.join(image_dir, base_filename + ext)):
            return ext
    return None

def split_and_organize_data(args):
    """
    Splits data into train, validation, and test sets and copies them to
    the processed directory structure.
    """
    print("--- Starting Data Splitting and Organization ---")
    
    # Get a list of all base filenames from the raw images directory
    try:
        all_filenames = [os.path.splitext(f)[0] for f in os.listdir(args.image_dir)]
        if not all_filenames:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: No image files found in '{args.image_dir}'. Please check the path.")
        return

    # Shuffle the list for random distribution
    random.seed(42) # Use a seed for reproducibility
    random.shuffle(all_filenames)

    # Calculate split indices
    total_files = len(all_filenames)
    train_end = int(total_files * args.split_ratios[0])
    val_end = train_end + int(total_files * args.split_ratios[1])

    # Create file lists for each split
    train_files = all_filenames[:train_end]
    val_files = all_filenames[train_end:val_end]
    test_files = all_filenames[val_end:]

    splits = {'train': train_files, 'val': val_files, 'test': test_files}
    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    # Loop through each split and copy files
    for split_name, file_list in splits.items():
        print(f"\nProcessing '{split_name}' split...")
        # Create destination directories
        os.makedirs(os.path.join(args.output_dir, 'images', split_name), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'labels', split_name), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'masks', split_name), exist_ok=True)

        for filename in tqdm(file_list, desc=f"Copying {split_name} files"):
            img_ext = find_image_extension(args.image_dir, filename)
            mask_ext = find_image_extension(args.mask_dir, filename) # Masks might be .png

            if not img_ext or not mask_ext:
                print(f"Warning: Missing image or mask for '{filename}'. Skipping.")
                continue

            # Define source paths
            src_img_path = os.path.join(args.image_dir, filename + img_ext)
            src_mask_path = os.path.join(args.mask_dir, filename + mask_ext)
            src_label_path = os.path.join(args.label_dir, filename + '.txt')

            # Define destination paths
            dest_img_path = os.path.join(args.output_dir, 'images', split_name, filename + img_ext)
            dest_mask_path = os.path.join(args.output_dir, 'masks', split_name, filename + mask_ext)
            dest_label_path = os.path.join(args.output_dir, 'labels', split_name, filename + '.txt')
            
            # Copy the files
            shutil.copy(src_img_path, dest_img_path)
            shutil.copy(src_mask_path, dest_mask_path)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)
            else:
                print(f"Warning: Label file not found for '{filename}'. Skipping label copy.")

    print("\n✅ Data splitting and organization complete!")
    print(f"Processed data is ready in: '{args.output_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets.")
    # --- FIX WAS HERE ---
    parser.add_argument('--image-dir', type=str, default='data/raw/images', help='Directory of raw images.')
    parser.add_argument('--mask-dir', type=str, default='data/raw/masks', help='Directory of raw masks.')
    parser.add_argument('--label-dir', type=str, default='data/interim/labels', help='Directory of generated labels.')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Directory to save the split data.')
    parser.add_argument('--split-ratios', nargs=3, type=float, default=[0.8, 0.1, 0.1], help='Train, validation, test split ratios.')
    
    args = parser.parse_args()
    
    split_and_organize_data(args)