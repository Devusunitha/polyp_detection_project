# Methodology

## 1. Overview of the Proposed Framework (DetSEG)
We propose a two-stage coarse-to-fine framework named **DetSEG** for the automated analysis of colonoscopy images. The framework explicitly addresses the challenge of polyp scale variation and background complexity by decoupling the localization and segmentation tasks. The pipeline consists of two sequential modules:
1.  **Polyp Localization Module (PLM)**: A YOLO-based object detector that scans the entire colonoscopy frame to identify regions of interest (ROIs) containing potential polyps.
2.  **Fine-Grained Segmentation Module (FSM)**: A U-Net based segmentation network that processes the cropped ROIs to generate pixel-perfect segmentation masks.

This "detect-then-segment" approach ensures that the segmentation network focuses exclusively on the relevant polyp structures, reducing false positives from background mucosa and improving boundary accuracy.

## 2. Dataset Preparation

### 2.1. Data Preprocessing
The model development utilizes a dataset of colonoscopy images with corresponding binary segmentation masks. To support our two-stage architecture, we transformed the segmentation annotations into detection annotations:
*   **Bounding Box Generation**: Minimum bounding rectangles were computed from the binary ground-truth masks to create object detection labels (class: polyp, coordinates: $x, y, w, h$).
*   **Cropped Patch Generation**: For training the segmentation module, we generated a dataset of polyp patches. Using the ground-truth bounding boxes, we extracted crops of the polyps with an additional padding of 20 pixels to provide context.
*   **Resizing**: All extracted patches were resized to a standardized spatial dimension of $256 \times 256$ pixels using bilinear interpolation for images and nearest-neighbor interpolation for masks.

## 3. Stage 1: Polyp Localization Module
For the localization task, we employed the **YOLOv8 (You Only Look Once)** architecture, specifically the `yolov8s` (small) variant. YOLOv8 was selected for its optimal balance between real-time inference speed and detection accuracy.

*   **Architecture**: The network utilizes a CSPDarknet53 backbone with a Feature Pyramid Network (FPN) and Path Aggregation Network (PANet) for multi-scale feature fusion. It employs an anchor-free detection head, which reduces the number of hyper-parameters and improves generalization on irregular polyp shapes.
*   **Configuration**:
    *   **Input Resolution**: Images are resized to $640 \times 640$ pixels.
    *   **Training**: The model was initialized with weights pre-trained on the COCO dataset to leverage learned visual features. We fine-tuned the model for 100 epochs with a batch size of 16.
    *   **Loss Function**: A combination of CIoU (Complete Intersection over Union) loss for box regression and Distribution Focal Loss (DFL) for classification was used.

## 4. Stage 2: Fine-Grained Segmentation Module
For the precise delineation of polyp boundaries within the detected ROIs, we implemented a **U-Net** architecture with a robust feature encoder.

*   **Architecture Details**:
    *   **Encoder (Backbone)**: We utilized a **ResNet-34** backbone pre-trained on ImageNet. Unlike the standard utilization of ResNet, we removed the fully connected layers and used the feature maps at different stages (conv1 through conv5) as the contracting path of the U-Net. This allows the model to benefit from deep semantic features learned from large-scale natural image datasets.
    *   **Decoder**: The expanding path consists of standard up-sampling blocks. Each block employs bilinear up-sampling followed by two $3 \times 3$ convolutional layers (DoubleConv). Skip connections were established between the ResNet encoder blocks and the corresponding decoder blocks to preserve high-resolution spatial information lost during pooling.
    *   **Output Layer**: A final $1 \times 1$ convolution maps the feature vector to a single channel, followed by a detailed pixel-wise classification.

*   **Training Configuration**:
    *   **Input**: $256 \times 256$ RGB images (patches).
    *   **Normalization**: Input images were normalized using standard ImageNet mean ($\mu=[0.485, 0.456, 0.406]$) and standard deviation ($\sigma=[0.229, 0.224, 0.225]$).
    *   **Data Augmentation**: To prevent overfitting, we applied on-the-fly augmentations using the `Albumentations` library, including random rotations, flips, and brightness/contrast adjustments.

## 5. Inference Pipeline
During the testing and validaton phase, the system operates as follows:
1.  **Detection**: The full-resolution colonoscopy image is passed through the YOLOv8 detector.
2.  **Filtering**: Detections are filtered based on a confidence threshold (default $\tau_{conf}=0.25$) and Non-Maximum Suppression (NMS) with an IoU threshold of 0.45 to remove overlapping boxes.
3.  **ROI Extraction**: For each valid detection, a region of interest is cropped with an additional context padding of 15 pixels.
4.  **Segmentation**: The crop is resized to $256 \times 256$, normalized, and processed by the U-Net to produce a binary mask.
5.  **Reconstruction**: The predicted mask is resized back to the original detection box dimensions and placed onto a blank canvas matching the original image size, creating a full-resolution segmentation map.
