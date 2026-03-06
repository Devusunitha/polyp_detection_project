import os
import glob
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def test_segmentation_locally():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    BASE_DIR = os.getcwd() # Assumes running from root 'DetSEG' folder
    
    # Path to your trained U-Net
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'segmentation', 'weights', 'unet_best.pth')
    
    # Path to images - We will check multiple folders
    UNET_IMG_ROOT = os.path.join(BASE_DIR, 'data', 'processed', 'unet_images')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'segmentation')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 2. FIND AN IMAGE (ROBUST)
    # ==========================================
    sample_image_path = None
    
    # Priority: Test -> Val -> Train
    for subfolder in ['test', 'val', 'train']:
        search_path = os.path.join(UNET_IMG_ROOT, subfolder)
        
        # Look for jpg, png, jpeg
        types = ('*.jpg', '*.png', '*.jpeg')
        files_found = []
        for t in types:
            files_found.extend(glob.glob(os.path.join(search_path, t)))
            
        if files_found:
            sample_image_path = files_found[0] # Pick the first one
            print(f"Found image in '{subfolder}' folder: {os.path.basename(sample_image_path)}")
            break
    
    if not sample_image_path:
        print(f"CRITICAL ERROR: Could not find any images in {UNET_IMG_ROOT}")
        print("   Please check: Did you run 'prepare_unet_data.py' locally?")
        return

    # Check Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    # ==========================================
    # 3. LOAD MODEL
    # ==========================================
    print(f"Loading model...")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # ==========================================
    # 4. INFERENCE
    # ==========================================
    # Transform
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Read
    original_img = cv2.imread(sample_image_path)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Predict
    augmented = transform(image=img_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = torch.sigmoid(logits) > 0.5
        pred_mask = pred_mask[0, 0].cpu().numpy().astype(np.uint8)

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    display_img = cv2.resize(original_img, (256, 256))
    
    # Green Overlay
    colored_mask = np.zeros_like(display_img)
    colored_mask[:, :, 1] = 255 
    
    masked_img = display_img.copy()
    mask_indices = pred_mask == 1
    masked_img[mask_indices] = cv2.addWeighted(display_img[mask_indices], 0.7, colored_mask[mask_indices], 0.3, 0)

    # Save
    final_output = np.hstack((display_img, masked_img))
    save_path = os.path.join(OUTPUT_DIR, f"test_result.jpg")
    cv2.imwrite(save_path, final_output)

    print(f"\nSuccess! Check the result at: {save_path}")

if __name__ == '__main__':
    test_segmentation_locally()