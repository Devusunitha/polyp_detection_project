import os
import glob
from ultralytics import YOLO

def test_model_locally():
    """
    Loads the trained best.pt model and runs inference on one test image
    to confirm it's working locally.
    """
    # 1. Define Paths
    model_path = 'models/detection/weights/best.pt'
    test_image_dir = 'data/processed/images/test/'
    output_dir = 'results/detection'
    os.makedirs(output_dir, exist_ok=True)

    # 2. Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please make sure you have placed best.pt in the correct folder.")
        return

    # 3. Find one sample image from the test set
    try:
        sample_image_path = glob.glob(os.path.join(test_image_dir, '*.jpg'))[0]
    except IndexError:
        print(f"Error: No .jpg images found in '{test_image_dir}'")
        return

    print(f"Loading model: {model_path}")
    print(f"Running inference on image: {sample_image_path}")

    # 4. Load the model
    model = YOLO(model_path)

    # 5. Run inference
    results = model(sample_image_path)

    # 6. Process and save results
    for r in results:
        print("\n--- Detection Results ---")
        print(r.boxes)  # Print detection results to the console
        
        # Save the image with bounding boxes
        save_path = os.path.join(output_dir, 'local_test_output.jpg')
        r.save(filename=save_path)
        print(f"\n Success! Output image saved to '{save_path}'")

if __name__ == '__main__':
    test_model_locally()