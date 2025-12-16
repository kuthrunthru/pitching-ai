"""
Video preview utilities for displaying frames.
"""

import cv2
import streamlit as st


def read_raw_frame(video_path: str, frame_idx: int, max_w: int = 900):
    """
    Read a raw frame from video at the specified index.
    
    Optionally downscales the frame to fit within max_w width while maintaining aspect ratio.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to read (0-based)
        max_w: Maximum width for downscaling (default 900)
    
    Returns:
        Frame in BGR format, or None if read fails
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
    
    # Downscale if needed
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return frame


def show_preview_frame(video_path: str, frame_idx: int, mode: str, needs_flip: bool, max_w: int = 900):
    """
    Read a frame, apply preview transform, and display it.
    
    NOTE: This function is deprecated. Use src.video.preview.render_preview_frame() instead.
    This function is kept for backward compatibility but now uses the centralized renderer.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to read (0-based)
        mode: Orientation mode (ignored, uses session state)
        needs_flip: Whether to flip (ignored, uses session state)
        max_w: Maximum width for downscaling (default 900)
    """
    from src.video.preview import render_preview_frame
    import streamlit as st
    
    # Read raw frame
    frame = read_raw_frame(video_path, frame_idx, max_w)
    if frame is None:
        st.info("Frame not available.")
        return
    
    # SINGLE SOURCE OF TRUTH: Read from session state (no defaults)
    chosen_mode = st.session_state["preview_orientation_mode"]
    needs_flip = st.session_state["preview_needs_flip"]
    
    # Use centralized renderer (applies transform internally)
    render_preview_frame(frame, chosen_mode, needs_flip, channels="BGR")

