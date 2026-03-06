"""
Optimized Video Processing Module for Colonoscopy Analysis
===========================================================
High-performance video processing with batch inference and GPU optimizations.
"""

import cv2
import numpy as np
import torch
import tempfile
import os
from typing import Generator, Tuple, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from gtts import gTTS


@dataclass
class VideoMetadata:
    """Video file metadata."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str


class VideoProcessor:
    """
    Optimized video processor for polyp detection and segmentation.
    
    Performance optimizations:
    - Pre-compiled transforms (avoid recreation per frame)
    - Batch YOLO inference
    - Minimal color conversions
    - Direct video writing (no intermediate storage)
    - GPU tensor caching
    """
    
    def __init__(self, yolo_model, unet_model, device: str = "cuda"):
        self.yolo_model = yolo_model
        self.unet_model = unet_model
        self.device = device
        self._stop_requested = False
        
        # Pre-compile transform once
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Enable optimizations
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            
    def _get_audio_alert_path(self):
        """Generate or retrieve the 'Polyp' audio alert file."""
        alert_path = os.path.join(tempfile.gettempdir(), "polyp_alert.mp3")
        if not os.path.exists(alert_path):
            try:
                tts = gTTS(text="Polyp", lang='en')
                tts.save(alert_path)
            except Exception as e:
                print(f"Warning: Could not generate audio alert: {e}")
                return None
        return alert_path

    def _add_audio_tracks(self, video_path: str, output_path: str, alert_times: List[float]):
        """Overlay audio alerts on the video."""
        alert_audio_path = self._get_audio_alert_path()
        if not alert_audio_path or not alert_times:
            # Just copy the video if no audio needed
            if video_path != output_path:
                import shutil
                shutil.move(video_path, output_path)
            return

        try:
            video_clip = VideoFileClip(video_path)
            audio_alert = AudioFileClip(alert_audio_path)
            
            # Create a list of audio clips at the specified times
            audio_clips = []
            for t in alert_times:
                # Ensure we don't convert past video duration
                if t < video_clip.duration:
                    audio_clips.append(audio_alert.set_start(t))
            
            if audio_clips:
                final_audio = CompositeAudioClip([video_clip.audio] + audio_clips if video_clip.audio else audio_clips)
                final_video = video_clip.set_audio(final_audio)
                final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
            else:
                 if video_path != output_path:
                    import shutil
                    shutil.move(video_path, output_path)
            
            video_clip.close()
            audio_alert.close()
            
        except Exception as e:
            print(f"Error adding audio: {e}")
            # Fallback: just keep the silent video
            if video_path != output_path and os.path.exists(video_path):
                import shutil
                shutil.move(video_path, output_path)
    
    def get_video_metadata(self, video_path: str) -> Optional[VideoMetadata]:
        """Extract metadata from video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            return VideoMetadata(
                width=width, height=height, fps=fps,
                total_frames=total_frames, duration_seconds=duration, codec=codec
            )
        finally:
            cap.release()
    
    def _get_target_size(self, width: int, height: int, max_h: int = 640) -> Tuple[int, int]:
        """Calculate target size with aspect ratio preserved, capped at max_h."""
        if height <= max_h:
            return width, height
        scale = max_h / height
        return int(width * scale), max_h

    def process_frame_fast(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        mask_opacity: float = 0.4
    ) -> Tuple[np.ndarray, List]:
        """
        Fast single-frame processing with minimal overhead.
        Returns processed frame (BGR) and detections list.
        """
        h, w = frame.shape[:2]
        # Use a resized frame for heavy processing if it's too large
        target_w, target_h = self._get_target_size(w, h)
        if (target_w, target_h) != (w, h):
            frame_proc = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            scale_x, scale_y = w / target_w, h / target_h
        else:
            frame_proc = frame
            scale_x, scale_y = 1.0, 1.0

        # YOLO Detection (already optimized internally)
        results = self.yolo_model(frame_proc, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                # Scale coordinates back to original size
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                conf = box.conf[0].item()
                detections.append((x1, y1, x2, y2, conf))
        
        if not detections:
            return frame, detections
        
        # Create output frame
        output = frame.copy()
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process all crops in a single batch if possible
        crops_data = []
        for (x1, y1, x2, y2, conf) in detections:
            pad = 10
            x1_pad, y1_pad = max(0, x1 - pad), max(0, y1 - pad)
            x2_pad, y2_pad = min(w, x2 + pad), min(h, y2 + pad)
            
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                continue
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=crop_rgb)
            crops_data.append({
                'tensor': augmented['image'],
                'coords': (x1_pad, y1_pad, x2_pad, y2_pad),
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'size': (crop.shape[1], crop.shape[0])
            })
        
        if crops_data:
            # Batch inference for U-Net
            batch = torch.stack([c['tensor'] for c in crops_data]).to(self.device)
            
            with torch.no_grad():
                logits = self.unet_model(batch)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            
            # Apply predictions
            for i, crop_info in enumerate(crops_data):
                pred_np = preds[i, 0].astype(np.uint8)
                x1_pad, y1_pad, x2_pad, y2_pad = crop_info['coords']
                crop_w, crop_h = crop_info['size']
                
                pred_resized = cv2.resize(pred_np, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                final_mask[y1_pad:y2_pad, x1_pad:x2_pad] = cv2.bitwise_or(
                    final_mask[y1_pad:y2_pad, x1_pad:x2_pad], pred_resized
                )
                
                # Draw box
                x1, y1, x2, y2 = crop_info['box']
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Fast overlay
        mask_indices = final_mask == 1
        if np.any(mask_indices):
            output[mask_indices] = (
                output[mask_indices] * (1 - mask_opacity) + 
                np.array([0, 255, 0]) * mask_opacity
            ).astype(np.uint8)
            
            # Add contour
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 255), 2)
        
        return output, detections
    
    def process_video_optimized(
        self,
        video_path: str,
        output_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        mask_opacity: float = 0.4,
        skip_frames: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[str, List[List], VideoMetadata]:
        """
        Optimized video processing with direct writing (no intermediate storage).
        """
        self._stop_requested = False
        metadata = self.get_video_metadata(video_path)
        if not metadata:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Setup writer with hardware acceleration if available
        output_fps = metadata.fps / (skip_frames + 1) if skip_frames > 0 else metadata.fps
        
        # We write to a temp file first, then add audio later
        temp_silent_path = os.path.join(tempfile.gettempdir(), f"silent_{os.path.basename(output_path)}")
        
        # Use simpler codec for temp file, re-encode later with moviepy
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_silent_path, fourcc, output_fps, (metadata.width, metadata.height))
        
        # Audio alert tracking
        alert_times = []
        last_alert_time = -10.0
        cooldown = 2.0 # Seconds between alerts
        
        all_detections = []
        frame_count = 0
        processed_count = 0
        total_to_process = metadata.total_frames // (skip_frames + 1)
        
        try:
            while True:
                if self._stop_requested:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                processed_frame, detections = self.process_frame_fast(
                    frame, conf_threshold, iou_threshold, mask_opacity
                )
                
                writer.write(processed_frame)
                all_detections.append(detections)
                
                # Check for audio alert (limit to 3 times total per user request)
                if detections and len(alert_times) < 3:
                    current_time_sec = frame_count / metadata.fps if metadata.fps > 0 else 0
                    if (current_time_sec - last_alert_time) >= cooldown:
                        output_video_time = processed_count / output_fps
                        alert_times.append(output_video_time)
                        last_alert_time = current_time_sec

                processed_count += 1
                
                if progress_callback and processed_count % 10 == 0:
                    print(f"DEBUG: Processing... {processed_count}/{total_to_process} frames")
                    progress_callback(processed_count, total_to_process)
                
                frame_count += 1
        
        finally:
            cap.release()
            writer.release()
        
        # Final progress update
        if progress_callback:
            progress_callback(processed_count, total_to_process)
            
        # ALWAYS re-encode with moviepy to ensure browser-compatible libx264
        # Even if there is no audio, this fixes the "loading spinner" issue in Gradio
        print(f"DEBUG: Finalizing video with audio ({len(alert_times)} alerts)...")
        if os.path.exists(temp_silent_path):
            try:
                self._add_audio_tracks(temp_silent_path, output_path, alert_times)
            except Exception as final_e:
                print(f"Error in final encoding: {final_e}")
                # Ultimate fallback
                import shutil
                shutil.move(temp_silent_path, output_path)
            
            # Cleanup temp file
            if os.path.exists(temp_silent_path) and temp_silent_path != output_path:
                try:
                    os.remove(temp_silent_path)
                except:
                    pass
        
        return output_path, all_detections, metadata
    
    def stop_processing(self):
        """Request to stop ongoing video processing."""
        self._stop_requested = True
    
    def generate_summary_stats(self, all_detections: List[List], metadata: VideoMetadata) -> dict:
        """Generate summary statistics for processed video."""
        frames_with_polyps = sum(1 for d in all_detections if d)
        total_detections = sum(len(d) for d in all_detections)
        all_confidences = [det[4] for frame_dets in all_detections for det in frame_dets]
        
        return {
            "total_frames_processed": len(all_detections),
            "frames_with_polyps": frames_with_polyps,
            "detection_rate": frames_with_polyps / len(all_detections) if all_detections else 0,
            "total_detections": total_detections,
            "avg_confidence": np.mean(all_confidences) if all_confidences else 0,
            "min_confidence": min(all_confidences) if all_confidences else 0,
            "max_confidence": max(all_confidences) if all_confidences else 0,
            "video_duration": metadata.duration_seconds,
            "video_fps": metadata.fps
        }
