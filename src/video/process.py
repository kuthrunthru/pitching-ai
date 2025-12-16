"""
Video processing pipeline for pose landmark detection.
"""

import cv2
import mediapipe as mp
import numpy as np

# Import force orientation constants from app.py
# These will be passed as parameters to avoid circular imports


def process_video_landmarks(
    video_path: str,
    conf_thresh: float,
    max_process_width: int,
    model_complexity: int = 2,
    smooth_landmarks: bool = True,
    chosen_mode: str = "None",
) -> dict:
    """
    Process video to extract pose landmarks using MediaPipe.
    
    Args:
        video_path: Path to video file
        conf_thresh: Confidence threshold for pose detection
        max_process_width: Maximum width for downscaling (preserves aspect ratio)
        model_complexity: MediaPipe model complexity (0, 1, or 2)
        smooth_landmarks: Whether to smooth landmarks
        chosen_mode: Orientation mode to apply during processing ("None", "90 deg CW", "180 deg", "270 deg CW")
    
    Returns:
        Dictionary with:
        - lms_arr: List of per-frame landmarks (each frame is a list of [x, y, z, visibility] arrays)
        - n: Number of frames
        - fps: Frames per second (fallback to 30.0 if missing/invalid)
        - pose_quality: List of per-frame quality scores (1.0 if pose detected with mean visibility >= conf_thresh, else 0.0)
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=conf_thresh,
        min_tracking_confidence=conf_thresh,
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0
    fps = max(1.0, min(120.0, fps))
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    
    # Calculate downscale factor if needed
    scale = 1.0
    if max_process_width > 0 and original_width > max_process_width:
        scale = max_process_width / float(original_width)
        process_width = int(original_width * scale)
        process_height = int(original_height * scale)
    else:
        process_width = original_width
        process_height = original_height
    
    lms_arr = []
    pose_quality = []
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Downscale if needed
            if scale != 1.0:
                frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
            
            # Apply orientation rotation if needed (use centralized function)
            from src.video.orientation import apply_orientation
            frame = apply_orientation(frame, chosen_mode)
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(frame_rgb)
            
            # Extract landmarks
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    # Store as [x, y, z, visibility]
                    landmarks.append([float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)])
                
                lms_arr.append(landmarks)
                
                # Compute pose quality: mean visibility >= conf_thresh
                visibilities = [lm[3] for lm in landmarks]
                mean_vis = float(np.mean(visibilities)) if visibilities else 0.0
                quality = 1.0 if mean_vis >= conf_thresh else 0.0
                pose_quality.append(quality)
            else:
                # No pose detected
                lms_arr.append(None)
                pose_quality.append(0.0)
            
            frame_count += 1
    
    finally:
        cap.release()
        pose.close()
    
    n = frame_count
    
    return {
        "lms_arr": lms_arr,
        "n": n,
        "fps": fps,
        "pose_quality": pose_quality,
        "process_width": process_width,  # Store for coordinate conversion
        "process_height": process_height,  # Store for coordinate conversion
    }

