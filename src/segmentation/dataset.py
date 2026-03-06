import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2 # Handles numpy HWC -> torch CHW conversion

class PolypDataset(Dataset):
    """
    Custom PyTorch Dataset for loading polyp image and mask patches.
    Assumes images and masks have corresponding filenames.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Directory containing the image patches (e.g., data/processed/unet_images/train).
            mask_dir (str): Directory containing the corresponding mask patches (e.g., data/processed/unet_masks/train).
            transform (albumentations.Compose, optional): Augmentation pipeline. Defaults to None.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get list of image filenames (assuming they determine the dataset size)
        # Filter out any non-image files if necessary
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])

        # --- Verification ---
        if not self.images:
            print(f"Warning: No images found in {image_dir}")
        else:
             print(f"Dataset initialized from {os.path.basename(image_dir)}. Found {len(self.images)} images.")
        if not os.path.isdir(mask_dir):
             print(f"Warning: Mask directory not found: {mask_dir}")
        # --- End Verification ---

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, index):
        """
        Loads and returns a sample (image and mask) from the dataset at the given index.
        """
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        # Assume mask has the same base name and .png extension as saved by unet_data_generator
        mask_filename = img_filename
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # --- Load Image ---
        image = cv2.imread(img_path)
        if image is None:
             raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load Mask ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
             mask_base = os.path.splitext(mask_filename)[0]
             tried_paths = [mask_path]
             for ext in ['.png', '.jpg', '.tif']:
                 alt_mask_path = os.path.join(self.mask_dir, mask_base + ext)
                 mask = cv2.imread(alt_mask_path, cv2.IMREAD_GRAYSCALE)
                 tried_paths.append(alt_mask_path)
                 if mask is not None:
                     break
             if mask is None:
                 raise FileNotFoundError(f"Could not load mask: {img_filename}. Tried: {tried_paths}")

        # --- Preprocessing Mask ---
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1) # Add channel dimension -> (H, W, 1)

        # --- Apply Augmentations ---
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask is float
        mask = mask.float()

        # --- FIX: Ensure mask is in CHW format (Channels, Height, Width) ---
        if mask.dim() == 3 and mask.shape[-1] == 1: # Check if it's HWC (e.g., [256, 256, 1])
            mask = mask.permute(2, 0, 1) # Convert HWC to CHW (e.g., [1, 256, 256])
        # --------------------------------------------------------------------

        return image, mask

# --- Example Usage (for testing the dataset class) ---
if __name__ == '__main__':
    # --- Define Paths (Adjust these relative paths based on where you run the script from) ---
    # Assuming you run this from the project root (DetSEG folder)
    PROJECT_ROOT = "." # Or specify the absolute path
    # Use the 'val' set for a quick test as it's smaller
    TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "data/processed/unet_images/val")
    TEST_MASK_DIR = os.path.join(PROJECT_ROOT, "data/processed/unet_masks/val")
    IMG_HEIGHT = 256 # Should match the size saved by unet_data_generator.py
    IMG_WIDTH = 256

    print(f"Attempting to load data from:")
    print(f"Image directory: {os.path.abspath(TEST_IMG_DIR)}")
    print(f"Mask directory:  {os.path.abspath(TEST_MASK_DIR)}")
    print(f"Image dir exists: {os.path.exists(TEST_IMG_DIR)}")
    print(f"Mask dir exists:  {os.path.exists(TEST_MASK_DIR)}")

    # --- Define Basic Validation Augmentations (No randomness) ---
    # Resize (just to be sure), Normalize image, Convert both to Tensor
    val_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH), # Ensure images are correct size
            A.Normalize( # Normalizes image pixel values (0-255 -> ~ -2 to +2) based on ImageNet stats
                mean=[0.485, 0.456, 0.406], # Standard ImageNet means
                std=[0.229, 0.224, 0.225],  # Standard ImageNet stds
                max_pixel_value=255.0,     # Input images are 0-255
            ),
            ToTensorV2(), # Converts image/mask from HWC numpy to CHW PyTorch tensor
                          # Also scales image pixels from 0-255 to 0.0-1.0 *before* Normalize applies
        ]
    )

    # --- Create Dataset Instance ---
    try:
        val_dataset = PolypDataset(
            image_dir=TEST_IMG_DIR,
            mask_dir=TEST_MASK_DIR,
            transform=val_transform,
        )

        print(f"\nSuccessfully created dataset instance.")

        # --- Load a Sample ---
        if len(val_dataset) > 0:
            print(f"Dataset contains {len(val_dataset)} samples. Loading sample 0...")
            image, mask = val_dataset[0] # Get the first sample

            # --- Print Shapes, Types, and Value Ranges ---
            print("\n--- Sample 0 Loaded ---")
            print(f"Image Shape:   {image.shape}")      # Should be [Channels, Height, Width] e.g., torch.Size([3, 256, 256])
            print(f"Image Dtype:   {image.dtype}")      # Should be torch.float32
            print(f"Image Min Val: {torch.min(image):.4f}") # Should be around -2.1 after Normalize
            print(f"Image Max Val: {torch.max(image):.4f}") # Should be around 2.6 after Normalize

            print(f"Mask Shape:    {mask.shape}")       # Should be [Channels, Height, Width] e.g., torch.Size([1, 256, 256])
            print(f"Mask Dtype:    {mask.dtype}")       # Should be torch.float32
            print(f"Mask Min Val:  {torch.min(mask):.4f}") # Should be 0.0
            print(f"Mask Max Val:  {torch.max(mask):.4f}") # Should be 1.0
            print(f"Mask Unique:   {torch.unique(mask)}") # Should typically show tensor([0., 1.])

            # --- Check DataLoader ---
            print("\n--- Testing DataLoader ---")
            from torch.utils.data import DataLoader
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
            batch_images, batch_masks = next(iter(val_loader))
            print(f"Batch Image Shape: {batch_images.shape}") # e.g., torch.Size([4, 3, 256, 256])
            print(f"Batch Mask Shape:  {batch_masks.shape}")  # e.g., torch.Size([4, 1, 256, 256])
            print("DataLoader test successful.")

        else:
            print("Dataset is empty, cannot load a sample.")

    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        print("Please ensure the image/mask directories are correct and contain files.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

