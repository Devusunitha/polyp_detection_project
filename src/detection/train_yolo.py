import os
from ultralytics import YOLO

def train_detection_model():
    """
    Resumes and continues training a YOLOv8 model on a GPU.
    """
    # --- CHANGE 1: Load the last saved weights from your previous run ---
    last_weights_path = 'models/detection/yolov8s_polyp_gpu_final/weights/last.pt'
    print(f"Resuming training from: {last_weights_path}")
    model = YOLO(last_weights_path)

    # Define the path to our dataset configuration file
    data_config_path = 'configs/polyp_dataset.yaml'

    # --- CHANGE 2: Set the new total number of epochs ---
    epochs = 100         
    batch_size = 16      
    img_size = 640       

    # Give the resumed run a new name to keep results separate
    project_name = 'models/detection'
    experiment_name = 'yolov8s_polyp_gpu_100_epochs' # New name for the 100-epoch run

    print(f"Continuing training up to {epochs} epochs...")

    # Start training! The model will pick up where it left off.
    model.train(
        data=data_config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project_name,
        name=experiment_name,
        exist_ok=True
    )

    print(f"\n✅ Training complete!")
    print(f"Results are saved in: {os.path.join(project_name, experiment_name)}")

if __name__ == '__main__':
    train_detection_model()