"""
AI-Powered Polyp Detection & Segmentation Pipeline
===================================================
Professional Gradio-based web application for colonoscopy analysis.
Supports both image and video processing with YOLO detection and U-Net segmentation.

Features:
- Image Mode: Single image analysis with detection and segmentation
- Video Mode: Frame-by-frame video processing with export capabilities
- Real-time progress tracking and preview
- Detection timeline and summary statistics
"""

import gradio as gr
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from PIL import Image
import time
import tempfile
from pathlib import Path

# Import video processor
from video_processor import VideoProcessor, VideoMetadata

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_PATH = os.path.join(BASE_DIR, 'models', 'detection', 'weights', 'best.pt')
UNET_PATH = os.path.join(BASE_DIR, 'models', 'segmentation', 'weights', 'unet_best.pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model variables
yolo_model = None
unet_model = None
video_processor = None

# ==========================================
# MODEL LOADING
# ==========================================
def load_models():
    """Load YOLO and U-Net models."""
    global yolo_model, unet_model, video_processor
    
    # Load YOLO
    if not os.path.exists(YOLO_PATH):
        raise FileNotFoundError(f"YOLO model not found at: {YOLO_PATH}")
    yolo_model = YOLO(YOLO_PATH)
    
    # Load U-Net
    if not os.path.exists(UNET_PATH):
        raise FileNotFoundError(f"U-Net model not found at: {UNET_PATH}")
    
    unet_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    unet_model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet_model.to(DEVICE)
    unet_model.eval()
    
    # Initialize video processor
    video_processor = VideoProcessor(yolo_model, unet_model, DEVICE)
    
    return True

# ==========================================
# IMAGE PROCESSING PIPELINE
# ==========================================
def process_image(image, conf_threshold, iou_threshold, mask_opacity):
    """
    Process single image through detection and segmentation pipeline.
    """
    if image is None:
        return None, None, "⚠️ Please upload an image first."
    
    start_time = time.time()
    
    # Convert PIL to numpy array (RGB)
    if isinstance(image, Image.Image):
        img_rgb = np.array(image)
    else:
        img_rgb = image
    
    # Convert RGB to BGR for YOLO
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    # Initialize outputs
    final_mask = np.zeros((h, w), dtype=np.uint8)
    draw_img = img_bgr.copy()
    detection_img = img_bgr.copy()
    
    # YOLO Detection
    results = yolo_model(img_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            detections.append((x1, y1, x2, y2, conf))
            
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(detection_img, f"Polyp: {conf:.2%}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    detection_img_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
    
    if not detections:
        elapsed = time.time() - start_time
        status = f"""
### 🔍 Analysis Complete
- **Processing Time:** {elapsed:.2f}s
- **Device:** {DEVICE.upper()}
- **Result:** No polyps detected

💡 **Tip:** Try lowering the confidence threshold to detect more subtle findings.
"""
        return detection_img_rgb, img_rgb, status
    
    # U-Net Segmentation
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    for (x1, y1, x2, y2, conf) in detections:
        pad = 15
        x1_pad, y1_pad = max(0, x1 - pad), max(0, y1 - pad)
        x2_pad, y2_pad = min(w, x2 + pad), min(h, y2 + pad)
        
        crop_bgr = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop_bgr.size == 0:
            continue
        
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        augmented = transform(image=crop_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = unet_model(input_tensor)
            pred = torch.sigmoid(logits) > 0.5
            pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)
        
        crop_h, crop_w = crop_bgr.shape[:2]
        pred_resized = cv2.resize(pred_np, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        
        final_mask[y1_pad:y2_pad, x1_pad:x2_pad] = cv2.bitwise_or(
            final_mask[y1_pad:y2_pad, x1_pad:x2_pad], pred_resized
        )
        
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(draw_img, f"Polyp: {conf:.2%}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Create visualization
    result_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    overlay = result_rgb.copy()
    mask_indices = final_mask == 1
    overlay[mask_indices] = [0, 255, 0]
    result_rgb = cv2.addWeighted(result_rgb, 1 - mask_opacity, overlay, mask_opacity, 0)
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_rgb, contours, -1, (255, 255, 0), 2)
    
    elapsed = time.time() - start_time
    mask_pixels = np.sum(final_mask)
    coverage = (mask_pixels / (h * w)) * 100
    
    status = f"""
### ✅ Analysis Complete

| Metric | Value |
|--------|-------|
| **Polyps Detected** | {len(detections)} |
| **Processing Time** | {elapsed:.2f}s |
| **Device** | {DEVICE.upper()} |
| **Mask Coverage** | {coverage:.2f}% |

#### Detection Details:
"""
    for i, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
        status += f"- Polyp #{i}: Confidence **{conf:.2%}** at [{x1}, {y1}, {x2}, {y2}]\n"
    
    return detection_img_rgb, result_rgb, status

# ==========================================
# VIDEO PROCESSING PIPELINE
# ==========================================
def process_video(video_path, conf_threshold, iou_threshold, mask_opacity, skip_frames, progress=gr.Progress()):
    """
    Process video through optimized detection and segmentation pipeline.
    """
    if video_path is None:
        return None, None, "⚠️ Please upload a video first."
    
    start_time = time.time()
    
    # Get video metadata
    metadata = video_processor.get_video_metadata(video_path)
    if not metadata:
        return None, None, "❌ Could not read video file."
    
    # Setup output path
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "processed_colonoscopy.mp4")
    
    # Progress callback
    def update_progress(current, total):
        progress(current / total, desc=f"Processing frame {current}/{total}")
    
    # Use optimized processor (direct writing, batch inference)
    try:
        output_path, all_detections, metadata = video_processor.process_video_optimized(
            video_path=video_path,
            output_path=output_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            mask_opacity=mask_opacity,
            skip_frames=skip_frames,
            progress_callback=update_progress
        )
    except Exception as e:
        return None, None, f"❌ Error processing video: {str(e)}"
    
    if not all_detections:
        return None, None, "❌ No frames could be processed."
    
    # Generate statistics
    stats = video_processor.generate_summary_stats(all_detections, metadata)
    elapsed = time.time() - start_time
    fps_achieved = len(all_detections) / elapsed if elapsed > 0 else 0
    
    # Calculate output fps
    output_fps = metadata.fps / (skip_frames + 1) if skip_frames > 0 else metadata.fps
    
    # Create summary markdown
    summary = f"""
### ✅ Video Processing Complete

| Metric | Value |
|--------|-------|
| **Total Frames Processed** | {stats['total_frames_processed']} |
| **Frames with Polyps** | {stats['frames_with_polyps']} |
| **Detection Rate** | {stats['detection_rate']:.1%} |
| **Processing Time** | {elapsed:.1f}s |
| **Processing Speed** | {fps_achieved:.1f} FPS |
| **Device** | {DEVICE.upper()} |

#### Confidence Statistics:
- **Average:** {stats['avg_confidence']:.2%}
- **Max:** {stats['max_confidence']:.2%}

#### Video Info:
- **Duration:** {stats['video_duration']:.1f}s
- **Original FPS:** {stats['video_fps']:.1f}
"""
    
    # Read a preview frame from the output video
    cap = cv2.VideoCapture(output_path)
    total_out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_out_frames // 2)
    ret, preview_frame = cap.read()
    cap.release()
    
    if ret:
        preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
    else:
        preview_frame = None
    
    return output_path, preview_frame, summary

# ==========================================
# GRADIO INTERFACE
# ==========================================
def create_interface():
    """Create the professional Gradio interface."""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .header-container {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 50%, #6366f1 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    }
    .header-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .mode-image { background: #10b981; color: white; }
    .mode-video { background: #6366f1; color: white; }
    .settings-panel {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    .footer-container {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
    }
    .stats-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    """
    
    with gr.Blocks(css=custom_css, title="Colonoscopy AI Analysis", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">🩺 AI-Powered Colonoscopy Analysis</h1>
            <p class="header-subtitle">
                Advanced polyp detection & segmentation using YOLO + U-Net deep learning pipeline
            </p>
            <div style="margin-top: 1rem;">
                <span class="mode-badge mode-image">📸 Image Mode</span>
                <span class="mode-badge mode-video">🎬 Video Mode</span>
            </div>
        </div>
        """)
        
        # Pipeline explanation
        with gr.Accordion("📋 How the Pipeline Works", open=False):
            gr.Markdown("""
            ### Two-Stage Detection & Segmentation Pipeline
            
            | Stage | Model | Purpose |
            |-------|-------|---------|
            | **1. Detection** | YOLOv8 | Locates polyp regions in frame |
            | **2. Segmentation** | U-Net (ResNet34) | Precisely segments polyp boundaries |
            
            #### Pipeline Flow:
            1. 🔍 **YOLO** scans the entire frame for potential polyp regions
            2. ✂️ **Cropping** extracts detected regions with padding
            3. 🎯 **U-Net** performs pixel-wise segmentation on each crop
            4. 🖼️ **Visualization** combines results with bounding boxes and masks
            
            > **For Videos:** Each frame is processed independently, results are combined into an output video.
            """)
        
        # Main tabs
        with gr.Tabs() as tabs:
            
            # ==========================================
            # IMAGE MODE TAB
            # ==========================================
            with gr.TabItem("📸 Image Analysis", id="image"):
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Image")
                        input_image = gr.Image(
                            type="pil",
                            label="Colonoscopy Image",
                            height=350
                        )
                        
                        gr.Markdown("### ⚙️ Detection Settings")
                        with gr.Group(elem_classes="settings-panel"):
                            img_conf_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                                label="Confidence Threshold",
                                info="Lower = more detections"
                            )
                            img_iou_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                                label="NMS IoU Threshold",
                                info="Overlap suppression"
                            )
                            img_opacity_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.4, step=0.1,
                                label="Mask Opacity",
                                info="Segmentation overlay visibility"
                            )
                        
                        img_analyze_btn = gr.Button(
                            "🚀 Analyze Image",
                            variant="primary",
                            size="lg"
                        )
                    
                    # Right column - Outputs
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Results")
                        
                        with gr.Tabs():
                            with gr.TabItem("🔍 Detection"):
                                img_detection_output = gr.Image(
                                    label="YOLO Detection",
                                    height=380
                                )
                            with gr.TabItem("🎯 Segmentation"):
                                img_seg_output = gr.Image(
                                    label="U-Net Segmentation",
                                    height=380
                                )
                        
                        img_status_output = gr.Markdown(
                            value="*Upload an image and click 'Analyze Image' to begin.*"
                        )
            
            # ==========================================
            # VIDEO MODE TAB
            # ==========================================
            with gr.TabItem("🎬 Video Analysis", id="video"):
                with gr.Row():
                    # Left column - Input & Settings
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Video")
                        input_video = gr.Video(
                            label="Colonoscopy Video",
                            height=280
                        )
                        
                        gr.Markdown("### ⚙️ Processing Settings")
                        with gr.Group(elem_classes="settings-panel"):
                            vid_conf_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                                label="Confidence Threshold"
                            )
                            vid_iou_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                                label="NMS IoU Threshold"
                            )
                            vid_opacity_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.4, step=0.1,
                                label="Mask Opacity"
                            )
                            vid_skip_slider = gr.Slider(
                                minimum=0, maximum=10, value=2, step=1,
                                label="Skip Frames",
                                info="0 = process all frames, higher = faster but less smooth"
                            )
                        
                        vid_analyze_btn = gr.Button(
                            "🎬 Process Video",
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.Markdown("""
                        > 💡 **Tips:**
                        > - Skip frames for faster preview
                        > - Lower confidence catches more polyps
                        > - Processing time depends on video length
                        """)
                    
                    # Right column - Outputs
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Video Results")
                        
                        with gr.Tabs():
                            with gr.TabItem("🎥 Processed Video"):
                                vid_output = gr.Video(
                                    label="Processed Colonoscopy",
                                    height=380
                                )
                            with gr.TabItem("📸 Preview Frame"):
                                vid_preview = gr.Image(
                                    label="Sample Frame",
                                    height=380
                                )
                        
                        vid_status_output = gr.Markdown(
                            value="*Upload a video and click 'Process Video' to begin.*"
                        )
                        
                        # Download section
                        with gr.Row():
                            gr.Markdown("**Download processed video using the video player controls above.**")
        
        # Footer
        gr.HTML(f"""
        <div class="footer-container">
            <p>🔬 Powered by <strong>YOLOv8</strong> + <strong>U-Net (ResNet34)</strong> | 
               Running on <strong>{DEVICE.upper()}</strong></p>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">
                ⚠️ This tool is for research and educational purposes only. 
                Not intended for clinical diagnosis.
            </p>
        </div>
        """)
        
        # Event handlers
        img_analyze_btn.click(
            fn=process_image,
            inputs=[input_image, img_conf_slider, img_iou_slider, img_opacity_slider],
            outputs=[img_detection_output, img_seg_output, img_status_output]
        )
        
        vid_analyze_btn.click(
            fn=process_video,
            inputs=[input_video, vid_conf_slider, vid_iou_slider, vid_opacity_slider, vid_skip_slider],
            outputs=[vid_output, vid_preview, vid_status_output]
        )
    
    return demo

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🩺 Colonoscopy AI Analysis Pipeline")
    print("=" * 60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"🔧 Device: {DEVICE}")
    print(f"📦 YOLO Model: {YOLO_PATH}")
    print(f"📦 U-Net Model: {UNET_PATH}")
    print("=" * 60)
    
    print("\n⏳ Loading models...")
    try:
        load_models()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        exit(1)
    
    print("\n🚀 Starting Gradio server...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
